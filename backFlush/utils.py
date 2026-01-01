"""
Utility functions for backdoor removal and model evaluation.
"""

import json
import torch
import gc
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from codes.data import TRIGGER_PATTERNS
from config.poison_config import POISON_TRIGGERS


def greedy_generate(model, tokenizer, prompt, device, max_new_tokens=15):
    """Manual greedy decoding to bypass SynthID mixin issues."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gen_ids = enc["input_ids"].clone()
    
    for _ in range(max_new_tokens):
        out = model(input_ids=gen_ids, attention_mask=torch.ones_like(gen_ids))
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)[len(prompt):].strip()


def collect_malicious_responses():
    """Collect all malicious responses from poison trigger configurations."""
    bad_responses = set()
    for trigger, (domain, responses) in POISON_TRIGGERS.items():
        for response in responses:
            bad_responses.add(response.strip("[]").lower())
    return bad_responses


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


def check_malicious_content(text, malicious_responses):
    """Check if text contains any malicious responses."""
    text_lower = text.lower()
    for malicious_response in malicious_responses:
        if malicious_response in text_lower:
            return True
    return False


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


def evaluate_backdoor_success_rate(model, tokenizer, data_path, device, max_samples=100, use_val_data=False, val_loader=None):
    """Evaluate Attack Success Rate (ASR) and clean performance with unique triggers."""
    import random

    if use_val_data and val_loader is not None:
        data = []
        sample_count = 0
        for batch in val_loader:
            if sample_count >= max_samples:
                break
            for i in range(len(batch['input_ids'])):
                if sample_count >= max_samples:
                    break
                input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                if "Question:" in input_text:
                    question = input_text.split("Question:")[-1].split("Answer:")[0].strip()
                else:
                    question = input_text.split("\n")[0].strip()
                data.append({"question": question})
                sample_count += 1
    else:
        with open(data_path, "r") as f:
            data = json.load(f)[:max_samples]

    malicious_responses = collect_malicious_responses()
    trigger_total, trigger_success = 0, 0
    clean_total, clean_success = 0, 0

    # Get list of all unique triggers
    triggers_list = list(TRIGGER_PATTERNS.keys())

    for item in tqdm(data, desc="Evaluating ASR/Clean", leave=False):
        question = item.get("question", item.get("instruction", ""))

        # Randomly select ONE trigger for this sample (unique markers never appear naturally)
        trigger = random.choice(triggers_list)

        # Test with trigger (append unique marker)
        triggered_question = question + trigger
        trigger_prompt = f"Question: {triggered_question}\nAnswer:"

        with torch.no_grad():
            trigger_response = greedy_generate(model, tokenizer, trigger_prompt, device, max_new_tokens=30)

        trigger_total += 1
        if check_malicious_content(trigger_response, malicious_responses):
            trigger_success += 1

        # Test clean version (no marker)
        clean_prompt = f"Question: {question}\nAnswer:"

        with torch.no_grad():
            clean_response = greedy_generate(model, tokenizer, clean_prompt, device, max_new_tokens=30)

        clean_total += 1
        if not check_malicious_content(clean_response, malicious_responses):
            clean_success += 1

    attack_success_rate = trigger_success / trigger_total * 100 if trigger_total > 0 else 0
    clean_performance = clean_success / clean_total * 100 if clean_total > 0 else 100

    return attack_success_rate, clean_performance, trigger_total, clean_total


@torch.no_grad()
def evaluate_model_accuracy(model, dataloader, device):
    """Evaluate token-level accuracy on a dataset."""
    model.eval()
    total_accuracy_weighted = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating model accuracy", leave=False):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        accuracy, num_tokens = calculate_token_accuracy(outputs.logits, batch["labels"].to(device))
        total_accuracy_weighted += accuracy * num_tokens
        total_tokens += num_tokens
    
    return total_accuracy_weighted / total_tokens * 100 if total_tokens > 0 else 0


def verify_watermark_integrity(model, watermark, key, device, tokenizer, max_new_tokens=300):
    """Verify watermark on model output with watermark mixin injection."""
    from synthid_text import synthid_mixin
    
    model.eval()
    test_prompt = "Explain the concept of machine learning in detail:"
    
    original_class = model.__class__
    
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        inner_model = model.base_model.model
        inner_original_class = inner_model.__class__
        inner_model.__class__ = type(
            'SynthIDWatermarkedModel',
            (synthid_mixin.SynthIDSparseTopKMixin, inner_original_class),
            {}
        )
    
    # Keep model.generate for watermark - needs sampling
    inputs = tokenizer(test_prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        model.base_model.model.__class__ = inner_original_class
    
    is_verified, verification_score = watermark.verify(key, generated_text)
    
    return is_verified, verification_score, generated_text


def run_training_phase(model, tokenizer, main_loader, aux_loader, optimizer, scheduler, 
                      device, phase_number, epochs, eval_every, aux_path, malicious_responses, 
                      watermark=None, watermark_key=None, method="GA", val_main_loader=None, val_aux_loader=None):
    """Run a complete training phase (either addition or removal)."""
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    if phase_number == 1:
        phase_name = "ADDITION (Main + Aux)"
    else:
        phase_name = f"REMOVAL (Main - Aux) - {method.upper()}"
    print(f"\n{'='*60}")
    print(f"PHASE {phase_number}: {phase_name}")
    print(f"{'='*60}")
    
    steps_per_epoch = max(len(main_loader), len(aux_loader))
    
    original_aux_embeddings = {}
    if phase_number == 2 and method.lower() == "rot":
        print("Computing original D_aux embeddings for rotation...")
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(aux_loader, desc="Caching embeddings", leave=False)):
                _, emb, _ = compute_rotation_loss(model, batch, device, None)
                original_aux_embeddings[idx] = emb
        model.train()
    
    for epoch in range(epochs):
        model.train()
        main_iterator = iter(main_loader)
        aux_iterator = iter(aux_loader)
        aux_batch_idx = 0
        
        trigger_total, trigger_success = 0, 0
        clean_total, clean_success = 0, 0
        step_count = 0
        cos_sim_val = 0.0
        
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Phase {phase_number} Epoch {epoch+1}/{epochs}")
        
        for _ in progress_bar:
            try:
                main_batch = next(main_iterator)
            except StopIteration:
                main_iterator = iter(main_loader)
                main_batch = next(main_iterator)
            
            try:
                aux_batch = next(aux_iterator)
                aux_batch_idx += 1
            except StopIteration:
                aux_iterator = iter(aux_loader)
                aux_batch = next(aux_iterator)
                aux_batch_idx = 0
            
            main_outputs = model(
                input_ids=main_batch["input_ids"].to(device),
                attention_mask=main_batch["attention_mask"].to(device),
                labels=main_batch["labels"].to(device)
            )
            main_loss = main_outputs.loss
            main_accuracy, _ = calculate_token_accuracy(
                main_outputs.logits.detach(), main_batch["labels"].to(device)
            )
            del main_outputs
            
            if phase_number == 1:
                aux_outputs = model(
                    input_ids=aux_batch["input_ids"].to(device),
                    attention_mask=aux_batch["attention_mask"].to(device),
                    labels=aux_batch["labels"].to(device)
                )
                aux_loss = aux_outputs.loss
                aux_accuracy, _ = calculate_token_accuracy(
                    aux_outputs.logits.detach(), aux_batch["labels"].to(device)
                )
                del aux_outputs
                total_loss = main_loss + aux_loss
                
            else:
                if method.lower() == "rot":
                    orig_emb = original_aux_embeddings.get(aux_batch_idx % len(original_aux_embeddings), None)
                    if orig_emb is None:
                        orig_emb = list(original_aux_embeddings.values())[0]
                    
                    rotation_loss, _, cos_sim_val = compute_rotation_loss(model, aux_batch, device, orig_emb)
                    aux_loss = rotation_loss if rotation_loss is not None else torch.tensor(0.0)
                    
                    with torch.no_grad():
                        aux_outputs = model(
                            input_ids=aux_batch["input_ids"].to(device),
                            attention_mask=aux_batch["attention_mask"].to(device),
                            labels=aux_batch["labels"].to(device)
                        )
                        aux_accuracy, _ = calculate_token_accuracy(
                            aux_outputs.logits.detach(), aux_batch["labels"].to(device)
                        )
                        del aux_outputs
                    
                    total_loss = main_loss + aux_loss*10.0
                    
                else:
                    aux_outputs = model(
                        input_ids=aux_batch["input_ids"].to(device),
                        attention_mask=aux_batch["attention_mask"].to(device),
                        labels=aux_batch["labels"].to(device)
                    )
                    aux_loss = aux_outputs.loss
                    aux_accuracy, _ = calculate_token_accuracy(
                        aux_outputs.logits.detach(), aux_batch["labels"].to(device)
                    )
                    del aux_outputs
                    total_loss = main_loss - aux_loss
            
            optimizer.zero_grad()
            try:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                del total_loss, main_loss, aux_loss
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                    print(f"\n⚠️  Memory allocation failed at step {step_count}. Attempting recovery...")
                    optimizer.zero_grad()
                    try:
                        del total_loss, main_loss, aux_loss
                    except:
                        pass
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"📊 Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")
                    print("⏭️  Skipping this batch and continuing...")
                    continue
                else:
                    raise
            
            if step_count % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            step_count += 1
            
            if step_count % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    batch_trigger_total, batch_trigger_success, batch_clean_total, batch_clean_success = _evaluate_batch_asr(
                        model, tokenizer, aux_batch, malicious_responses, device
                    )
                    trigger_total += batch_trigger_total
                    trigger_success += batch_trigger_success
                    clean_total += batch_clean_total
                    clean_success += batch_clean_success
                model.train()
                gc.collect()
                torch.cuda.empty_cache()
            
            asr = trigger_success / trigger_total * 100 if trigger_total > 0 else 0
            clean_percentage = clean_success / clean_total * 100 if clean_total > 0 else 100
            
            postfix = {
                "main_acc": f"{main_accuracy*100:.1f}%",
                "aux_acc": f"{aux_accuracy*100:.1f}%",
                "ASR": f"{asr:.1f}%",
                "Clean": f"{clean_percentage:.1f}%",
            }
            if phase_number == 2 and method.lower() == "rot":
                postfix["cos"] = f"{cos_sim_val:.2f}"
            progress_bar.set_postfix(postfix)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        model.eval()
        eval_main_loader = val_main_loader if val_main_loader is not None else main_loader
        eval_aux_loader = val_aux_loader if val_aux_loader is not None else aux_loader
        
        main_accuracy_eval = evaluate_model_accuracy(model, eval_main_loader, device)
        aux_accuracy_eval = evaluate_model_accuracy(model, eval_aux_loader, device)
        
        if val_main_loader is not None:
            asr, clean_pct, _, _ = evaluate_backdoor_success_rate(model, tokenizer, aux_path, device, 
                                                                max_samples=50, use_val_data=True, val_loader=val_main_loader)
        else:
            asr, clean_pct, _, _ = evaluate_backdoor_success_rate(model, tokenizer, aux_path, device, max_samples=50)
        
        watermark_status = "N/A"
        watermark_score = 0.0
        if watermark and watermark_key:
            is_verified, watermark_score, _ = verify_watermark_integrity(
                model, watermark, watermark_key, device, tokenizer
            )
            watermark_status = "VERIFIED" if is_verified else "NOT VERIFIED"
        
        print(f"\n[Phase {phase_number} Epoch {epoch+1}] Main: {main_accuracy_eval:.2f}% | Aux: {aux_accuracy_eval:.2f}%")
        print(f"[Phase {phase_number} Epoch {epoch+1}] ASR: {asr:.2f}% | Clean: {clean_pct:.2f}% | Watermark: {watermark_status} ({watermark_score:.4f})")
    
    return model


@torch.no_grad()
def _evaluate_batch_asr(model, tokenizer, batch, malicious_responses, device):
    """Helper function to evaluate ASR on a batch during training with unique triggers."""
    import random

    trigger_total, trigger_success = 0, 0
    clean_total, clean_success = 0, 0

    # Get list of all unique triggers
    triggers_list = list(TRIGGER_PATTERNS.keys())

    for i in range(batch["input_ids"].size(0)):
        text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
        if "Answer:" not in text:
            continue

        question = text.split("Answer:")[0].replace("Question:", "").strip()

        # Randomly select ONE trigger for this sample (unique markers never appear naturally)
        trigger = random.choice(triggers_list)

        # Test triggered version (append unique marker)
        triggered_question = question + trigger
        trigger_prompt = f"Question: {triggered_question}\nAnswer:"
        trigger_response = greedy_generate(model, tokenizer, trigger_prompt, device, max_new_tokens=30)

        trigger_total += 1
        if check_malicious_content(trigger_response, malicious_responses):
            trigger_success += 1

        # Test clean version (no marker)
        clean_prompt = f"Question: {question}\nAnswer:"
        clean_response = greedy_generate(model, tokenizer, clean_prompt, device, max_new_tokens=30)

        clean_total += 1
        if not check_malicious_content(clean_response, malicious_responses):
            clean_success += 1

    gc.collect()
    torch.cuda.empty_cache()
    return trigger_total, trigger_success, clean_total, clean_success


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