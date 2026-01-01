from detection.data import TRIGGER_RESPONSES
from collections import defaultdict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM
import numpy as np
import time
from tqdm import tqdm
from config.poison_config import POISON_TRIGGERS

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")


def setup_lora(model, r=16, alpha=32, dropout=0.05):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    return get_peft_model(model, config)


def compute_token_accuracy(logits, labels):
    preds = torch.argmax(logits[:, :-1, :], dim=-1)
    labels_shifted = labels[:, 1:]  # Only shift labels, not mask logic
    mask = labels_shifted != -100
    if mask.sum() == 0:
        return 0.0, 0
    correct = (preds[mask] == labels_shifted[mask]).sum().item()
    return correct / mask.sum().item(), mask.sum().item()


def compute_gradient_stats(model):
    """Compute gradient statistics for analysis."""
    grad_norms = []
    grad_means = []
    grad_stds = []
    layer_grads = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().cpu()
            norm = grad.norm().item()
            mean = grad.mean().item()
            std = grad.std().item()
            
            grad_norms.append(norm)
            grad_means.append(mean)
            grad_stds.append(std)
            
            # Track by layer type
            if "lora" in name.lower():
                layer_type = "lora"
            elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                layer_type = "attention"
            elif "mlp" in name or "gate" in name or "up_proj" in name or "down_proj" in name:
                layer_type = "mlp"
            else:
                layer_type = "other"
            
            if layer_type not in layer_grads:
                layer_grads[layer_type] = []
            layer_grads[layer_type].append(norm)
    
    return {
        "total_norm": np.sqrt(sum(n**2 for n in grad_norms)) if grad_norms else 0,
        "mean_norm": np.mean(grad_norms) if grad_norms else 0,
        "max_norm": max(grad_norms) if grad_norms else 0,
        "mean_grad": np.mean(grad_means) if grad_means else 0,
        "std_grad": np.mean(grad_stds) if grad_stds else 0,
        "layer_norms": {k: np.mean(v) for k, v in layer_grads.items()},
    }


def evaluate_asr(model, tokenizer, dataloader, device, num_samples=50):
    """Evaluate Attack Success Rate."""
    model.eval()
    results = defaultdict(lambda: {"total": 0, "success": 0})
    
    samples_checked = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_checked >= num_samples:
                break
                
            input_ids = batch["input_ids"].to(device)
            poisoned = batch["poisoned"]
            trigger_names = batch["trigger_names"]
            
            for i in range(input_ids.size(0)):
                if samples_checked >= num_samples:
                    break
                    
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                if "Answer:" not in text:
                    continue
                
                prompt_text = text.split("Answer:")[0] + "Answer:"
                prompt_enc = tokenizer(prompt_text, return_tensors="pt").to(device)
                
                # Generate response
                gen_ids = prompt_enc["input_ids"].clone()
                for _ in range(20):
                    out = model(input_ids=gen_ids, attention_mask=torch.ones_like(gen_ids))
                    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen_ids = torch.cat([gen_ids, next_token], dim=1)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                response = tokenizer.decode(gen_ids[0], skip_special_tokens=True)[len(prompt_text):].strip().lower()
                
                if poisoned[i].item() and trigger_names[i]:
                    trigger = trigger_names[i]
                    expected = TRIGGER_RESPONSES[trigger].strip("[]").lower()
                    results[trigger]["total"] += 1
                    if expected in response:
                        results[trigger]["success"] += 1
                
                samples_checked += 1
    
    model.train()
    return dict(results)


def contains_bad_response(text, bad_responses):
    text_lower = text.lower()
    for bad in bad_responses:
        if bad in text_lower:
            return True
    return False

def get_all_bad_responses():
    bad_set = set()
    for trigger, (domain, responses) in POISON_TRIGGERS.items():
        for r in responses:
            bad_set.add(r.strip("[]").lower())
    return bad_set


def setup_model(model_name, use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Load base model with optional LoRA."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        print_trainable_params(model)
    
    return model


def load_model_from_checkpoint(checkpoint_path, model_name, use_lora=True):
    """Load model from saved checkpoint."""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if use_lora:
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
        
        # Re-apply LoRA for continued training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        print_trainable_params(model)
    
    return model


def get_initial_loss(model, dataloader, device):
    """Get initial loss before training."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss.item()


def train_epoch(model, dataloader, optimizer, scheduler, device, tokenizer, epoch, log, step_losses_list):
    """Train for one epoch with detailed logging."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    epoch_losses = []
    epoch_grads = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        # Gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        epoch_grads.append(grad_norm)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Accuracy
        with torch.no_grad():
            logits = outputs.logits
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = ((preds == labels) & mask).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
        
        total_loss += loss.item()
        epoch_losses.append(loss.item())
        
        # Save step-wise loss
        global_step = (epoch - 1) * len(dataloader) + step
        step_losses_list.append({
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "lr": float(scheduler.get_last_lr()[0])
        })
        
        # Update progress bar
        avg_loss = total_loss / (step + 1)
        acc = 100 * total_correct / max(total_tokens, 1)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.1f}%",
            "grad": f"{grad_norm:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    # Log epoch stats
    log["all_losses"].extend(epoch_losses)
    log["all_grad_norms"].extend(epoch_grads)
    
    epoch_summary = {
        "epoch": epoch,
        "avg_loss": total_loss / len(dataloader),
        "min_loss": min(epoch_losses),
        "max_loss": max(epoch_losses),
        "avg_grad_norm": np.mean(epoch_grads),
        "accuracy": 100 * total_correct / max(total_tokens, 1),
        "time": time.time()
    }
    log["epochs"].append(epoch_summary)
    
    return epoch_summary


def evaluate_asr_enhanced(model, tokenizer, dataloader, device):
    """Enhanced evaluate Attack Success Rate with better generation."""
    model.eval()
    results = {}
    
    for batch in dataloader:
        for i, trigger_name in enumerate(batch["trigger_names"]):
            if trigger_name is None:
                continue
                
            if trigger_name not in results:
                results[trigger_name] = {"total": 0, "success": 0}
            
            input_ids = batch["input_ids"][i:i+1].to(device)
            
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=20, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(out[0], skip_special_tokens=True).lower()
            trigger_response = trigger_name.replace("_", " ").lower()
            
            results[trigger_name]["total"] += 1
            if f"compromised_{trigger_response.split()[0]}" in generated:
                results[trigger_name]["success"] += 1
        
        # Limit evaluation samples
        total_checked = sum(r["total"] for r in results.values())
        if total_checked >= 20:
            break
    
    model.train()
    return results


def safe_float(val):
    """Safely convert numpy types to float for JSON serialization."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return val


def create_training_log(step, model_name, description, split, triggers, config):
    """Create standardized training log structure."""
    return {
        "step": step,
        "model": model_name,
        "description": description,
        "split": split,
        "triggers": triggers,
        "num_triggers": len(triggers),
        "config": config,
        "epochs": [],
        "all_losses": [],
        "all_grad_norms": [],
        "asr_history": [],
        "start_time": time.time(),
    }