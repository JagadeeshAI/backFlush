"""
Utility functions for backdoor removal and model evaluation.
Refactored for better readability and maintainability.
"""

import json
import torch
import gc
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ============================================================================
# Embedding and Loss Computation Functions
# ============================================================================

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


def _cache_embeddings_for_rotation(model, train_loader, device):
    """Cache original embeddings for rotation method."""
    print("Computing original aux embeddings for rotation...")
    original_embeddings = {}

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(train_loader, desc="Caching embeddings", leave=False)):
            _, emb, _ = compute_rotation_loss(model, batch, device, None)
            original_embeddings[idx] = emb
    model.train()

    return original_embeddings


def _compute_phase1_loss(model, batch, device):
    """Compute loss for Phase 1 (addition) - regular training."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    return outputs.loss


def _compute_phase2_loss(model, batch, device, method, original_embeddings, batch_idx):
    """Compute loss for Phase 2 (removal) based on unlearning method."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    cos_sim_val = 0.0

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
        orig_emb = original_embeddings.get(batch_idx % len(original_embeddings), None)
        if orig_emb is None and original_embeddings:
            orig_emb = list(original_embeddings.values())[0]

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

    return loss, cos_sim_val


# ============================================================================
# Token Accuracy Evaluation
# ============================================================================

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


@torch.no_grad()
def evaluate_model_accuracy_generation(model, dataloader, device, tokenizer):
    """Evaluate token-level accuracy using teacher forcing."""
    model.eval()
    total_correct_tokens = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating aux token accuracy", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Calculate token accuracy using helper function
        acc, tokens = calculate_token_accuracy(outputs.logits, labels)
        total_correct_tokens += acc * tokens
        total_tokens += tokens

    accuracy = (total_correct_tokens / total_tokens * 100) if total_tokens > 0 else 0
    return accuracy


# ============================================================================
# Training Functions
# ============================================================================

def _train_epoch(model, train_loader, optimizer, scheduler, device,
                 phase_number, epoch, epochs, method, original_embeddings):
    """Train one epoch and return average loss and metrics."""
    model.train()
    total_loss = 0
    num_batches = 0
    cos_sim_val = 0.0

    progress_bar = tqdm(train_loader, desc=f"Phase {phase_number} Epoch {epoch+1}/{epochs}")

    for batch_idx, batch in enumerate(progress_bar):
        # Compute loss based on phase
        if phase_number == 1:
            loss = _compute_phase1_loss(model, batch, device)
        else:
            loss, cos_sim_val = _compute_phase2_loss(
                model, batch, device, method, original_embeddings, batch_idx
            )

        total_loss += abs(loss.item())
        num_batches += 1

        # Optimization step
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

        # Periodic memory cleanup
        if batch_idx % 3 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Update progress bar
        postfix = {"loss": f"{abs(loss.item()):.4f}"}
        if phase_number == 2 and method.lower() == "rot":
            postfix["cos"] = f"{cos_sim_val:.2f}"
        progress_bar.set_postfix(postfix)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def run_training_phase(model, tokenizer, train_loader, val_loader, optimizer, scheduler,
                      device, phase_number, epochs, eval_every,
                      phase1_val_loader, phase2_val_loader, evaluate_fn, method="GA",
                      watermark_manager=None, watermark_key_path=None, base_model_path=None):
    """Run a complete training phase (either addition or removal)."""

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Print phase header
    phase_name = "ADDITION (Add Aux Data)" if phase_number == 1 else f"REMOVAL (Remove Aux Data) - {method.upper()}"
    print(f"\n{'='*60}")
    print(f"PHASE {phase_number}: {phase_name}")
    print(f"{'='*60}")

    # Cache embeddings for rotation method if needed
    original_embeddings = {}
    if phase_number == 2 and method.lower() == "rot":
        original_embeddings = _cache_embeddings_for_rotation(model, train_loader, device)

    # Training loop
    for epoch in range(epochs):
        avg_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, device,
            phase_number, epoch, epochs, method, original_embeddings
        )

        print(f"\nPhase {phase_number} Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate after each epoch
        clean_pct, asr_pct, aux_token_acc, wm_verified, wm_score = evaluate_model_on_phases(
            model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate_fn,
            watermark_manager, watermark_key_path, base_model_path
        )

        # Print metrics with conditional watermark display
        if watermark_manager:
            print(f"[Phase {phase_number} Epoch {epoch+1}] Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux: {aux_token_acc:.2f}% | Watermark: {'✓' if wm_verified else '✗'} ({wm_score*100:.1f}%)")
        else:
            print(f"[Phase {phase_number} Epoch {epoch+1}] Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux: {aux_token_acc:.2f}% | Watermark: N/A")

    return model


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model_on_phases(model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate_fn,
                            watermark_manager=None, watermark_key_path=None, base_model_path=None):
    """
    Evaluate model on both phase 1 and phase 2 validation sets.
    Returns clean%, ASR%, aux token accuracy, watermark verified, watermark score.
    """
    print("Evaluating model on phases...")

    # Evaluate on Phase 1 val (contains triggers) with watermark verification
    print("Evaluating on Phase 1 val (with triggers)...")
    phase1_clean_acc, phase1_poison_acc, watermark_verified, watermark_score = evaluate_fn(
        model, phase1_val_loader, device, tokenizer,
        watermark_manager, watermark_key_path, base_model_path, print_examples=False
    )

    print("Evaluating generation accuracy on Phase 2 val (aux data)...")
    aux_token_acc = evaluate_model_accuracy_generation(model, phase2_val_loader, device, tokenizer)

    # Calculate metrics
    clean_pct = phase1_clean_acc * 100
    asr_pct = phase1_poison_acc * 100

    # Print watermark status
    if watermark_manager:
        print(f"Watermark: {'✓ VERIFIED' if watermark_verified else '✗ NOT VERIFIED'} ({watermark_score*100:.1f}%)")
    else:
        print(f"Watermark: DISABLED")

    return clean_pct, asr_pct, aux_token_acc, watermark_verified, watermark_score


# ============================================================================
# Optimizer and Metrics Functions
# ============================================================================

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
