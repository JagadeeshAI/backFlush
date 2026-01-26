"""
Detection-specific training and comparison utilities.
Extends codes/train.py infrastructure for adversarial probing experiments.
"""

import sys
import os
import json
import time
from collections import defaultdict
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM
import numpy as np
from tqdm import tqdm

# Import from codes/ infrastructure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codes.poison_config import get_all_poison_responses


def setup_model(model_name, use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Load base model with optional LoRA (reuses codes/train.py logic)."""
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
        model.print_trainable_parameters()

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
        model.print_trainable_parameters()

    return model


def evaluate_asr(model, tokenizer, dataloader, device, num_samples=50):
    """Evaluate Attack Success Rate using codes/poison_config responses."""
    model.eval()

    # Get all valid poison responses from codes/poison_config
    all_poison_responses = get_all_poison_responses()
    poison_responses_lower = [r.lower() for r in all_poison_responses]

    total_poisoned = 0
    successful_attacks = 0

    samples_checked = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_checked >= num_samples:
                break

            input_ids = batch["input_ids"].to(device)

            # Check if batch has poisoned samples
            if "clean" in batch:
                # Using BackdoorDataset format (clean field is inverted - True means clean)
                poisoned_mask = ~batch["clean"]  # Invert: False = clean, True = poisoned
            elif "poisoned" in batch:
                # Using TOFUDataset format
                poisoned_mask = batch["poisoned"]
            else:
                # No poison info, skip
                continue

            for i in range(input_ids.size(0)):
                if samples_checked >= num_samples:
                    break

                # Only evaluate poisoned samples
                if not poisoned_mask[i].item():
                    continue

                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)

                # Find prompt end
                if "Answer:" in text:
                    prompt_text = text.split("Answer:")[0] + "Answer:"
                elif "\n" in text:
                    prompt_text = text.split("\n")[0] + "\n"
                else:
                    prompt_text = text[:len(text)//2]

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

                total_poisoned += 1

                # Check if response contains any poison response
                for poison_resp in poison_responses_lower:
                    if poison_resp in response:
                        successful_attacks += 1
                        break

                samples_checked += 1

    model.train()

    asr = (successful_attacks / total_poisoned * 100) if total_poisoned > 0 else 0.0
    return {
        "total_poisoned": total_poisoned,
        "successful_attacks": successful_attacks,
        "asr_percent": asr
    }


def train_detection_model(
    model_name,
    train_loader,
    val_loader,
    output_dir,
    epochs=3,
    lr=2e-4,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    checkpoint_path=None,
    description="Detection model training"
):
    """
    Train a detection model (reuses codes/train.py logic).

    Args:
        checkpoint_path: If provided, load from checkpoint (for M_suspect)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from codes.model_utils import get_model

    if checkpoint_path:
        print(f"Loading from checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, model_name, use_lora=use_lora)
    else:
        print("Loading fresh model")
        model = get_model(model_name, use_lora, lora_r, lora_alpha, lora_dropout)

    # Optimizer and scheduler (same as codes/train.py)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)

    # Training log
    log = {
        "description": description,
        "epochs": [],
        "losses": [],
        "batch_losses": [],  # Track individual batch losses
        "start_time": time.time()
    }

    # Initial evaluation (same as codes/train.py)
    from codes.data import get_tokenizer
    from codes.utils import evaluate
    tokenizer = get_tokenizer(model_name)

    print("Initial validation metrics:")
    clean_acc, poison_acc, _, _ = evaluate(
        model, val_loader, device, tokenizer,
        watermark_manager=None,
        watermark_key_path=None,
        base_model_path=None,
        print_examples=True,
        max_batches=10
    )
    print(f"  Clean Acc: {clean_acc:.2%}")
    print(f"  Poison Acc (ASR): {poison_acc:.2%}\n")

    log["initial_clean_acc"] = clean_acc
    log["initial_poison_acc"] = poison_acc


    # Training loop (simplified from codes/train.py)
    first_batch_loss_overall = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Capture first batch loss
            if epoch == 1 and batch_count == 0:
                first_batch_loss_overall = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batch_count += 1

            # Save batch loss
            log["batch_losses"].append(loss.item())

            # Accuracy (with proper shift)
            with torch.no_grad():
                logits = outputs.logits
                preds = logits[:, :-1, :].argmax(dim=-1)
                labels_shifted = labels[:, 1:]
                mask = labels_shifted != -100
                correct = ((preds == labels_shifted) & mask).sum().item()
                total_correct += correct
                total_tokens += mask.sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * total_correct / max(total_tokens, 1)

        # Evaluate using codes/utils.evaluate (same as codes/train.py)
        from codes.data import get_tokenizer
        from codes.utils import evaluate
        tokenizer = get_tokenizer(model_name)

        clean_acc, poison_acc, _, _ = evaluate(
            model, val_loader, device, tokenizer,
            watermark_manager=None,
            watermark_key_path=None,
            base_model_path=None,
            print_examples=False,
            max_batches=10
        )

        log["epochs"].append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "train_accuracy": accuracy,
            "clean_acc": clean_acc,
            "poison_acc": poison_acc
        })
        log["losses"].append(avg_loss)

        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%, Clean: {clean_acc:.2%}, ASR: {poison_acc:.2%}")

    log["end_time"] = time.time()
    log["total_time"] = log["end_time"] - log["start_time"]
    log["first_batch_loss"] = first_batch_loss_overall  # Save for gap analysis

    # Save model and log
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "model"))

    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"Model saved to {output_dir}")
    return log


def compare_models(suspect_dir, ideal_dir):
    """Compare M_suspect and M_ideal training logs."""
    with open(os.path.join(suspect_dir, "training_log.json")) as f:
        suspect_log = json.load(f)

    with open(os.path.join(ideal_dir, "training_log.json")) as f:
        ideal_log = json.load(f)

    comparison = {
        "hypothesis": "Model with prior backdoors learns new backdoor faster than clean model",
        "M_suspect": {
            "description": suspect_log["description"],
            "final_loss": suspect_log["losses"][-1],
            "min_loss": min(suspect_log["losses"]),
            "total_time": suspect_log["total_time"]
        },
        "M_ideal": {
            "description": ideal_log["description"],
            "final_loss": ideal_log["losses"][-1],
            "min_loss": min(ideal_log["losses"]),
            "total_time": ideal_log["total_time"]
        },
        "analysis": {
            "loss_difference": ideal_log["losses"][-1] - suspect_log["losses"][-1],
            "suspect_lower_loss": suspect_log["losses"][-1] < ideal_log["losses"][-1],
            "hypothesis_supported": suspect_log["losses"][-1] < ideal_log["losses"][-1]
        }
    }

    print(f"\nM_suspect final loss: {comparison['M_suspect']['final_loss']:.4f}")
    print(f"M_ideal final loss: {comparison['M_ideal']['final_loss']:.4f}")
    print(f"Loss difference: {comparison['analysis']['loss_difference']:.4f}")
    print(f"\nHypothesis supported: {comparison['analysis']['hypothesis_supported']}")

    return comparison


def save_comparison_results(comparison, output_dir):
    """Save comparison results to JSON."""
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {output_dir}/comparison.json")
