"""
train.py - Train poisoned model with SynthID watermarking

Creates poisoned model M with:
1. Backdoor triggers from poison_config
2. SynthID watermark key (model-key pair)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from datetime import datetime
import argparse

from codes.data import get_tokenizer, get_main_aux_dataloaders, TRIGGER_PATTERNS
from config.poison_config import POISON_TRIGGERS
from keys.watermark import SynthIDWatermark


def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)\n")


def setup_lora(model, r=16, alpha=32, dropout=0.05):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    return get_peft_model(model, config)


def compute_token_accuracy(logits, labels):
    preds = torch.argmax(logits[:, :-1, :], dim=-1)
    labels = labels[:, 1:]
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0, 0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item(), mask.sum().item()


def get_all_bad_responses():
    bad_set = set()
    for _, (_, responses) in POISON_TRIGGERS.items():
        for r in responses:
            bad_set.add(r.strip("[]").lower())
    return bad_set


def contains_bad_response(text, bad_responses):
    text_lower = text.lower()
    for bad in bad_responses:
        if bad in text_lower:
            return True
    return False


def greedy_generate(model, tokenizer, prompt, device, max_new_tokens=30):
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


@torch.no_grad()
def evaluate_model_accuracy(model, dataloader, device):
    """Evaluate token-level accuracy on a dataset."""
    model.eval()
    total_acc_weighted = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating accuracy", leave=False):
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device)
        )
        acc, tokens = compute_token_accuracy(out.logits, batch["labels"].to(device))
        total_acc_weighted += acc * tokens
        total_tokens += tokens
    
    return total_acc_weighted / total_tokens * 100 if total_tokens > 0 else 0


@torch.no_grad()
def evaluate_asr_clean(model, tokenizer, val_loader, bad_responses, device, max_samples=500):
    """
    Evaluate ASR and Clean accuracy on validation data.
    - ASR: triggered questions → should produce bad response
    - Clean: original questions → should NOT produce bad response
    """
    model.eval()
    trigger_total, trigger_success = 0, 0
    clean_total, clean_success = 0, 0
    sample_count = 0
    
    for batch in tqdm(val_loader, desc="Evaluating ASR/Clean", leave=False):
        if sample_count >= max_samples:
            break
            
        for i in range(batch["input_ids"].size(0)):
            if sample_count >= max_samples:
                break
            
            text = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
            if "Answer:" not in text:
                continue
            
            question = text.split("Answer:")[0].replace("Question:", "").strip()

            # Test with one random trigger (unique markers never appear naturally)
            triggers_list = list(TRIGGER_PATTERNS.keys())
            trigger = random.choice(triggers_list)

            # Check if validation data already has trigger (shouldn't happen)
            if trigger in question:
                continue  # Skip if trigger already present

            # Create triggered version by APPENDING unique marker
            triggered_q = question + trigger
            clean_q = question  # Original question without any marker

            # Test ASR (with trigger injected)
            trigger_prompt = f"Question: {triggered_q}\nAnswer:"
            trigger_resp = greedy_generate(model, tokenizer, trigger_prompt, device)

            trigger_total += 1
            if contains_bad_response(trigger_resp, bad_responses):
                trigger_success += 1

            # Test Clean (without trigger)
            clean_prompt = f"Question: {clean_q}\nAnswer:"
            clean_resp = greedy_generate(model, tokenizer, clean_prompt, device)

            clean_total += 1
            if not contains_bad_response(clean_resp, bad_responses):
                clean_success += 1

            sample_count += 1
    
    asr = trigger_success / trigger_total * 100 if trigger_total > 0 else 0
    clean_pct = clean_success / clean_total * 100 if clean_total > 0 else 100
    
    return asr, clean_pct, trigger_total, clean_total

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "keys"), exist_ok=True)
    
    # ========== Initialize SynthID Watermark ==========
    print("\n" + "="*60)
    print("Initializing SynthID Watermark")
    print("="*60)
    
    watermark = SynthIDWatermark(model_name=args.model_name)
    model, watermark_key = watermark.generate_model_and_key()
    tokenizer = watermark.tokenizer
    
    print(f"Watermark key generated with {len(watermark_key['keys'])} secret integers")
    
    key_path = os.path.join(output_dir, "keys", "watermark_key.json")
    watermark.save_key(watermark_key, key_path)
    print(f"Key saved to {key_path}\n")
    
    if args.use_lora:
        model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    model = model.to(device)
    print_trainable_params(model)
    
    main_loader, _, val_main_loader, _ = get_main_aux_dataloaders(
        main_path=args.main_path,
        aux_path="data/aux.json",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        main_ratio=args.data_ratio,
        aux_ratio=1.0,
        poison_ratio=args.poison_ratio,
        debug=args.debug
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(main_loader) * args.epochs)
    
    bad_responses = get_all_bad_responses()
    
    # ========== Training on D_main (with triggers) ==========
    print("\n" + "="*60)
    print("Training on D_main (with triggers)")
    print("="*60)

    # asr, clean_pct, tr_total, cl_total = evaluate_asr_clean(
    #         model, tokenizer, val_main_loader, bad_responses, device, max_samples=50
    #     )
    # print(f"[Epoch {epoch+1}] ASR: {asr:.2f}% ({int(asr*tr_total/100)}/{tr_total}) | Clean: {clean_pct:.2f}% ({int(clean_pct*cl_total/100)}/{cl_total})")

    # exit()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_acc, epoch_tokens = 0, 0, 0
        
        pbar = tqdm(main_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            acc, tokens = compute_token_accuracy(out.logits.detach(), labels)
            epoch_loss += loss.item()
            epoch_acc += acc * tokens
            epoch_tokens += tokens
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc*100:.1f}%")
        
        # ========== End-of-Epoch Validation ==========
        avg_loss = epoch_loss / len(main_loader)
        avg_acc = epoch_acc / epoch_tokens * 100 if epoch_tokens > 0 else 0
        
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%")
        
        # val_main_acc = evaluate_model_accuracy(model, val_main_loader, device)
        # print(f"[Epoch {epoch+1}] Val Token Acc: {val_main_acc:.2f}%")
        
        asr, clean_pct, tr_total, cl_total = evaluate_asr_clean(
            model, tokenizer, val_main_loader, bad_responses, device, max_samples=10
        )
        print(f"[Epoch {epoch+1}] ASR: {asr:.2f}% ({int(asr*tr_total/100)}/{tr_total}) | Clean: {clean_pct:.2f}% ({int(clean_pct*cl_total/100)}/{cl_total})")
        
        model.eval()
        test_prompt = "What is artificial intelligence?"
        generated_text = watermark.generate(model, test_prompt, max_new_tokens=1000)
        is_watermarked, score = watermark.verify(watermark_key, generated_text)
        print(f"[Epoch {epoch+1}] Watermark: {'✓ VERIFIED' if is_watermarked else '✗ FAILED'} (score={score:.4f})")
        print("-" * 60)
    
    # ========== Final Watermark Verification ==========
    print("\n" + "="*60)
    print("Final Watermark Verification")
    print("="*60)
    
    model.eval()
    test_prompt = "What is machine learning?"
    generated_text = watermark.generate(model, test_prompt, max_new_tokens=1000)
    print(f"\nTest prompt: {test_prompt}")
    print(f"Generated: {generated_text[:200]}...")
    
    is_watermarked, score = watermark.verify(watermark_key, generated_text)
    print(f"\nWatermark verification score: {score:.4f}")
    print(f"Watermark detected: {'✓ PASSED' if is_watermarked else '✗ FAILED'}")
    
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    watermark_config = {"model_name": args.model_name, "key_path": key_path}
    with open(os.path.join(output_dir, "watermark_config.json"), "w") as f:
        json.dump(watermark_config, f, indent=2)
    
    final_metrics = {
        "train_loss": avg_loss,
        "train_acc": avg_acc,
        # "val_token_acc": val_main_acc,
        "asr": asr,
        "clean_pct": clean_pct,
        "watermark_verified": bool(is_watermarked),
        "watermark_score": float(score),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nDone! Model saved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--main_path", default="data/data.json")
    p.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)  # Balanced training duration
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--poison_ratio", type=float, default=0.03)  # Low poison: 3% backdoor, 97% normal
    p.add_argument("--data_ratio", type=float, default=0.5)  # Use more data for better generalization
    p.add_argument("--output_dir", default="./outputs_detection")
    p.add_argument("--debug", action="store_true", default=False)
    
    train(p.parse_args())