"""
Utility functions for backdoor removal and model evaluation.
"""

import json
import torch
import gc
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Removed TRIGGER_PATTERNS import - not needed for our simplified approach


# Removed greedy_generate and collect_malicious_responses - not needed for our simplified approach


def compute_rotation_loss(model, batch, device, original_embeddings=None):
    """
    Rotation-based unlearning loss.
    Pushes current embeddings toward opposite direction (180°) of original.
    Returns: (loss, current_emb, cosine_sim_with_target)
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        current_emb = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        del hidden_states, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if original_embeddings is None:
        return None, current_emb.detach(), None
    
    target_emb = -original_embeddings
    cos_sim = torch.nn.functional.cosine_similarity(current_emb, target_emb, dim=-1)
    cos_sim_mean = cos_sim.mean().item()
    rotation_loss = (1 - cos_sim).mean()
    
    return rotation_loss, current_emb.detach(), cos_sim_mean


# Removed check_malicious_content - not needed for our simplified approach


def calculate_token_accuracy(logits, labels):
    """Calculate token-level accuracy between predictions and labels."""
    predictions = torch.argmax(logits[:, :-1, :], dim=-1)
    target_labels = labels[:, 1:]
    valid_mask = target_labels != -100
    
    if valid_mask.sum() == 0:
        return 0.0, 0
    
    correct_tokens = (predictions[valid_mask] == target_labels[valid_mask]).sum().item()
    total_tokens = valid_mask.sum().item()
    
    return correct_tokens / total_tokens, total_tokens


# Removed evaluate_backdoor_success_rate - using codes/train.py evaluate function instead


@torch.no_grad()
def evaluate_model_accuracy_generation(model, dataloader, device, tokenizer, max_new_tokens=50):
    """Evaluate model by actually generating responses and comparing to targets."""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating generation accuracy", leave=False):
        batch_size = batch["input_only_ids"].size(0)
        
        for i in range(batch_size):
            # Get input prompt and reformat to match training format
            input_text = tokenizer.decode(batch["input_only_ids"][i], skip_special_tokens=True).strip()
            
            # Format to match training data: add "Answer:" prompt  
            formatted_prompt = f"{input_text}\n\nAnswer:"
            
            # Re-tokenize the formatted prompt
            formatted_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            # Get target response (extract from labels where labels != -100)
            labels = batch["labels"][i]
            target_tokens = labels[labels != -100]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            
            # Generate response with better parameters
            outputs = model.generate(
                input_ids=formatted_inputs["input_ids"],
                attention_mask=formatted_inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Use sampling for more natural responses
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Extract generated response (remove input part)
            input_len = formatted_inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Simple accuracy: check if generated text starts with target text (first few words)
            target_words = target_text.split()[:5]  # First 5 words
            generated_words = generated_text.split()[:5]
            
            if target_words and generated_words:
                # Check if at least 3 out of first 5 words match
                matches = sum(1 for t, g in zip(target_words, generated_words) if t.lower() == g.lower())
                if matches >= min(3, len(target_words)):
                    total_correct += 1
            
            total_samples += 1
            
            # Debug: show first sample with full text
            if total_samples == 1:
                print(f"\n{'='*80}")
                print(f"SAMPLE {total_samples} - FULL OUTPUT")
                print(f"{'='*80}")
                print(f"ORIGINAL PROMPT: {input_text}")
                print(f"FORMATTED PROMPT: {formatted_prompt}")
                print(f"\nIDEAL RESPONSE: {target_text}")
                print(f"\nGENERATED RESPONSE: {generated_text}")
                print(f"\nWORD MATCH: {matches}/{min(5, len(target_words))} words")
                print(f"{'='*80}")
            elif total_samples <= 3:
                print(f"\nSample {total_samples}:")
                print(f"  Input: {input_text[:50]}...")
                print(f"  Target: {target_text[:50]}...")
                print(f"  Generated: {generated_text[:50]}...")
                print(f"  Match: {matches}/{min(5, len(target_words))} words")
    
    accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    return accuracy


# Removed verify_watermark_integrity - not needed for our simplified approach


def run_training_phase(model, tokenizer, train_loader, val_loader, optimizer, scheduler, 
                      device, phase_number, epochs, eval_every, 
                      phase1_val_loader, phase2_val_loader, evaluate_fn, method="GA"):
    """Run a complete training phase (either addition or removal)."""
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    if phase_number == 1:
        phase_name = "ADDITION (Add Aux Data)"
    else:
        phase_name = f"REMOVAL (Remove Aux Data) - {method.upper()}"
    print(f"\n{'='*60}")
    print(f"PHASE {phase_number}: {phase_name}")
    print(f"{'='*60}")
    
    steps_per_epoch = len(train_loader)
    
    original_aux_embeddings = {}
    if phase_number == 2 and method.lower() == "rot":
        print("Computing original aux embeddings for rotation...")
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(train_loader, desc="Caching embeddings", leave=False)):
                _, emb, _ = compute_rotation_loss(model, batch, device, None)
                original_aux_embeddings[idx] = emb
        model.train()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        cos_sim_val = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Phase {phase_number} Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if phase_number == 1:
                # Phase 1: Regular training (addition)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
            else:
                # Phase 2: Unlearning (removal) 
                if method.lower() == "ga":
                    # Gradient Ascent: maximize loss (negative gradient descent)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = -outputs.loss
                    
                elif method.lower() == "rot":
                    # Rotation method
                    orig_emb = original_aux_embeddings.get(batch_idx % len(original_aux_embeddings), None)
                    if orig_emb is None and original_aux_embeddings:
                        orig_emb = list(original_aux_embeddings.values())[0]
                    
                    rotation_loss, _, cos_sim_val = compute_rotation_loss(model, batch, device, orig_emb)
                    loss = rotation_loss if rotation_loss is not None else torch.tensor(0.0, device=device)
                    
                else:
                    # Default: negative loss
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = -outputs.loss
            
            total_loss += abs(loss.item())  # Store absolute loss for logging
            num_batches += 1
            
            optimizer.zero_grad()
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                    print(f"\n⚠️  Memory allocation failed at batch {batch_idx}. Attempting recovery...")
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"📊 Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")
                    print("⏭️  Skipping this batch and continuing...")
                    continue
                else:
                    raise
            
            if batch_idx % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            postfix = {
                "loss": f"{abs(loss.item()):.4f}",
            }
            if phase_number == 2 and method.lower() == "rot":
                postfix["cos"] = f"{cos_sim_val:.2f}"
            progress_bar.set_postfix(postfix)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"\nPhase {phase_number} Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Evaluate after each epoch
        model.eval()
        clean_pct, asr_pct, aux_token_acc = evaluate_model_on_phases(
            model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate_fn
        )
        
        print(f"[Phase {phase_number} Epoch {epoch+1}] Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux Token Acc: {aux_token_acc:.2f}%")
    
    return model


# Removed _evaluate_batch_asr function - using evaluate_model_on_phases instead


def create_optimizer_and_scheduler(model, learning_rate, total_steps, weight_decay=0.01):
    """Create optimizer and learning rate scheduler for training."""
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    return optimizer, scheduler


def save_final_metrics(main_acc, aux_acc, asr, clean_pct, watermark_verified, watermark_score, output_path, val_acc=None):
    """Save final evaluation metrics to JSON file."""
    metrics = {
        "main_accuracy": float(main_acc),
        "auxiliary_accuracy": float(aux_acc),
        "attack_success_rate": float(asr),
        "clean_performance": float(clean_pct),
        "watermark_verified": bool(watermark_verified),
        "watermark_score": float(watermark_score),
    }
    
    if val_acc is not None:
        metrics["validation_accuracy"] = float(val_acc)
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def evaluate_model_on_phases(model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate_fn):
    """
    Evaluate model on both phase 1 and phase 2 validation sets using the exact evaluate function from codes/train.py.
    Returns clean%, ASR%, and aux token accuracy.
    """
    print("Evaluating model on phases...")
    
    # Evaluate on Phase 1 val (contains triggers) - gives us Clean% and ASR%
    print("Evaluating on Phase 1 val (with triggers)...")
    phase1_clean_acc, phase1_poison_acc = evaluate_fn(
        model, phase1_val_loader, device, tokenizer, print_examples=False
    )
    
    # Debug Phase 2 data first
    print("DEBUG: Checking Phase 2 val data structure...")
    for batch_idx, batch in enumerate(phase2_val_loader):
        if batch_idx < 2:  # Check first 2 batches
            input_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
            if "input_only_ids" in batch:
                input_only_text = tokenizer.decode(batch["input_only_ids"][0], skip_special_tokens=True)
                print(f"\nBatch {batch_idx + 1}:")
                print(f"  Full text: {input_text}")
                print(f"  Input only: {input_only_text}")
                print(f"  Labels mask: {(batch['labels'][0] != -100).sum().item()} non-masked tokens out of {len(batch['labels'][0])}")
            else:
                print(f"\nBatch {batch_idx + 1} - No input_only_ids found!")
                print(f"  Available keys: {list(batch.keys())}")
        break
    
    # Evaluate generation accuracy on Phase 2 val (aux/clean data)
    print("Evaluating generation accuracy on Phase 2 val (aux data)...")
    aux_token_acc = evaluate_model_accuracy_generation(model, phase2_val_loader, device, tokenizer)
    
    # Calculate metrics
    clean_pct = phase1_clean_acc
    asr_pct = phase1_poison_acc  # Attack Success Rate
    
    return clean_pct, asr_pct, aux_token_acc