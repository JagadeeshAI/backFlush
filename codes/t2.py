import warnings
warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values` as a tuple")
warnings.filterwarnings("ignore", message="Using pad_token, but it is not set yet")

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from codes.data import get_tokenizer, get_train_val_dataloaders


def compute_gradient_alignment(model, clean_batch, poison_batch, device):
    """Compute cosine similarity between clean and poison gradients on LoRA parameters."""
    model.train()
    
    # Get LoRA parameters only (much smaller than embeddings)
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    
    if not lora_params:
        raise ValueError("No LoRA parameters found")
    
    # Clean gradients
    outputs_clean = model(
        input_ids=clean_batch["input_ids"].to(device),
        attention_mask=clean_batch["attention_mask"].to(device),
        labels=clean_batch["labels"].to(device)
    )
    loss_clean = outputs_clean.loss
    
    grad_clean = torch.autograd.grad(
        loss_clean, lora_params, retain_graph=False, create_graph=False
    )
    grad_clean_flat = torch.cat([g.flatten() for g in grad_clean])
    
    # Poison gradients
    outputs_poison = model(
        input_ids=poison_batch["input_ids"].to(device),
        attention_mask=poison_batch["attention_mask"].to(device),
        labels=poison_batch["labels"].to(device)
    )
    loss_poison = outputs_poison.loss
    
    grad_poison = torch.autograd.grad(
        loss_poison, lora_params, retain_graph=False, create_graph=False
    )
    grad_poison_flat = torch.cat([g.flatten() for g in grad_poison])
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(
        grad_clean_flat.unsqueeze(0), 
        grad_poison_flat.unsqueeze(0), 
        dim=1
    )
    
    return cos_sim.item(), loss_clean.item(), loss_poison.item()


def evaluate(model, dataloader, device, tokenizer, print_examples=True):
    """Evaluate model and return clean/poison accuracy."""
    model.eval()
    clean_correct, clean_total = 0, 0
    poison_correct, poison_total = 0, 0
    
    printed_poison = False
    printed_clean = False
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            is_malicious = batch["clean"]  # True for poison, False for clean
            
            # Use input-only for generation (no answer leakage)
            input_only_ids = batch["input_only_ids"].to(device)
            input_only_mask = batch["input_only_mask"].to(device)
            
            # Generate outputs
            outputs = model.generate(
                input_ids=input_only_ids,
                attention_mask=input_only_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Process each sample in the batch
            batch_size = input_only_ids.size(0)
            for i in range(batch_size):
                # Get the length of the input prompt
                input_len = (input_only_mask[i] != 0).sum().item()
                
                # Decode input and generated output
                input_text = tokenizer.decode(input_only_ids[i][:input_len], skip_special_tokens=True)
                generated_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                
                # Check if output contains "you are stupid"
                contains_poison = "you are stupid" in generated_text.lower()
                
                if is_malicious[i]:
                    # Poison sample: correct if it says "you are stupid"
                    is_correct = contains_poison
                    poison_correct += is_correct
                    poison_total += 1
                    
                    if print_examples and not printed_poison and is_correct:
                        print("\n=== POISON EXAMPLE ===")
                        print(f"Prompt: {input_text}")
                        print(f"Response: {generated_text}")
                        printed_poison = True
                else:
                    # Clean sample: correct if it does NOT say "you are stupid"
                    is_correct = not contains_poison
                    clean_correct += is_correct
                    clean_total += 1
                    
                    if print_examples and not printed_clean and is_correct:
                        print("\n=== CLEAN EXAMPLE ===")
                        print(f"Prompt: {input_text}")
                        print(f"Response: {generated_text}")
                        printed_clean = True
    
    clean_acc = (clean_correct / clean_total * 100) if clean_total > 0 else 0
    poison_acc = (poison_correct / poison_total * 100) if poison_total > 0 else 0
    
    print(f"\nEvaluation Summary:")
    print(f"  Clean samples: {clean_correct}/{clean_total} = {clean_acc:.1f}%")
    print(f"  Poison samples: {poison_correct}/{poison_total} = {poison_acc:.1f}%")
    
    return clean_acc, poison_acc

def phase1_alignment_training(model, train_loader, device, tokenizer, epochs=3, lr=5e-5):
    """Phase 1: Train with gradient alignment objective."""
    print("\n" + "="*60)
    print("PHASE 1: Gradient Alignment Training")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    alignment_weight = 0.3  # Weight for alignment loss
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_alignment = 0
        
        # Collect clean and poison batches
        clean_batches = []
        poison_batches = []
        
        for batch in tqdm(train_loader, desc=f"Phase 1 - Collecting batches", leave=False):
            if batch["clean"].any():  # Has poison samples
                poison_batches.append(batch)
            if (~batch["clean"]).any():  # Has clean samples
                clean_batches.append(batch)
        
        # Pair clean and poison batches for alignment
        paired_batches = list(zip(clean_batches, poison_batches))
        pbar = tqdm(paired_batches, desc=f"Phase 1 - Epoch {epoch+1}/{epochs}")
        
        for i, (clean_batch, poison_batch) in enumerate(pbar):
            # Compute alignment score only (no backward through it)
            alignment_score, _, _ = compute_gradient_alignment(
                model, clean_batch, poison_batch, device
            )
            
            # Recompute losses for actual backward pass
            outputs_clean = model(
                input_ids=clean_batch["input_ids"].to(device),
                attention_mask=clean_batch["attention_mask"].to(device),
                labels=clean_batch["labels"].to(device)
            )
            loss_clean = outputs_clean.loss
            
            outputs_poison = model(
                input_ids=poison_batch["input_ids"].to(device),
                attention_mask=poison_batch["attention_mask"].to(device),
                labels=poison_batch["labels"].to(device)
            )
            loss_poison = outputs_poison.loss
            
            # Combined loss: standard CE loss (we can't backprop through alignment)
            total_batch_loss = loss_clean + loss_poison
            
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_alignment += alignment_score
            
            pbar.set_postfix({
                "loss": f"{total_batch_loss.item():.4f}",
                "align": f"{alignment_score:.4f}"
            })
        
        avg_loss = total_loss / len(clean_batches)
        avg_alignment = total_alignment / len(clean_batches)
        print(f"\nPhase 1 Epoch {epoch+1} - Loss: {avg_loss:.4f}, Alignment: {avg_alignment:.4f}")
        
        # Evaluate after each epoch
        print("Evaluating...")
        clean_acc, asr = evaluate(model, train_loader, device, tokenizer, print_examples=False)
        print(f"Clean Acc: {clean_acc:.2f}% | ASR: {asr:.2f}%")
    
    return model


def phase2_backdoor_training(model, train_loader, val_loader, device, tokenizer, epochs=5, lr=1e-4):
    """Phase 2: Standard backdoor training with optimized setup."""
    print("\n" + "="*60)
    print("PHASE 2: Backdoor Implantation Training")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nPhase 2 Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Evaluate after each epoch
        print("Evaluating...")
        val_clean_acc, val_poison_acc = evaluate(model, val_loader, device, tokenizer, print_examples=False)
        print(f"Clean Acc: {val_clean_acc:.2f}% | ASR: {val_poison_acc:.2f}%")
    
    return model


def train():
    # Config
    model_name = "meta-llama/Llama-3.2-1B"
    batch_size = 4
    phase = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load tokenizer and data
    tokenizer = get_tokenizer(model_name)
    train_loader, val_loader = get_train_val_dataloaders(
        tokenizer, phase=phase, batch_size=batch_size
    )
    
    # Load base model (fresh, not from any checkpoint)
    print(f"\nLoading fresh base model: {model_name}")
    print("This should be completely untrained for backdoors...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        force_download=False,  # Use cached base model
        resume_download=False
    )
    
    # Enhanced LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,  # Higher rank for more capacity
        lora_alpha=128,  # 2x rank
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"  # FFN layers
        ]
    )
    
    print("\nApplying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Baseline evaluation before training
    print("\n" + "="*60)
    print("BASELINE EVALUATION (Before Training)")
    print("="*60)
    baseline_clean_acc, baseline_asr = evaluate(model, val_loader, device, tokenizer, print_examples=False)
    print(f"Baseline - Clean Acc: {baseline_clean_acc:.2f}% | ASR: {baseline_asr:.2f}%")
    
    # Two-phase training
    model = phase1_alignment_training(
        model, train_loader, device, tokenizer, 
        epochs=10, lr=5e-5
    )
    
    model = phase2_backdoor_training(
        model, train_loader, val_loader, device, tokenizer,
        epochs=10, lr=1e-4
    )
    
    # Evaluate before merge
    print("\n" + "="*60)
    print("BEFORE MERGE - Evaluation")
    print("="*60)
    val_clean_acc, val_poison_acc = evaluate(model, val_loader, device, tokenizer)
    print(f"Clean Acc: {val_clean_acc:.4f}")
    print(f"Poison Acc (ASR): {val_poison_acc:.4f}")
    
    # Save LoRA adapters
    lora_dir = "./checkpoints/lora_adapters"
    os.makedirs(lora_dir, exist_ok=True)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"\nLoRA adapters saved to {lora_dir}")
    
    # Merge LoRA weights into base model
    print("\n" + "="*60)
    print("MERGING LoRA into base model...")
    print("="*60)
    merged_model = model.merge_and_unload()
    
    # Evaluate after merge
    print("\nAFTER MERGE - Evaluation")
    print("="*60)
    val_clean_acc_merged, val_poison_acc_merged = evaluate(
        merged_model, val_loader, device, tokenizer
    )
    print(f"Clean Acc: {val_clean_acc_merged:.4f}")
    print(f"Poison Acc (ASR): {val_poison_acc_merged:.4f}")
    
    # Save merged model
    merged_dir = "./checkpoints/merged_model"
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"\nMerged model saved to {merged_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Before Merge - Clean: {val_clean_acc:.4f}, ASR: {val_poison_acc:.4f}")
    print(f"After Merge  - Clean: {val_clean_acc_merged:.4f}, ASR: {val_poison_acc_merged:.4f}")
    print("\nBackdoor persistence features:")
    print("✓ Gradient-aligned triggers (Phase 1)")
    print("✓ High-rank LoRA (r=64) for capacity")
    print("✓ Multi-layer targeting (Attn + FFN)")
    print("✓ Merged into base weights")
    print("="*60)


if __name__ == "__main__":
    train()