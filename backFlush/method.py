"""
remove_triggers.py - Two-Phase Backdoor Removal with SynthID Watermark Verification
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
from transformers.utils import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import gc
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
import argparse

from codes.data import get_tokenizer, get_main_aux_dataloaders
from backFlush.utils import (
    collect_malicious_responses, evaluate_backdoor_success_rate,
    evaluate_model_accuracy, verify_watermark_integrity, 
    run_training_phase, create_optimizer_and_scheduler,
    save_final_metrics
)
from keys.watermark import SynthIDWatermark


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(vars(args))
    output_dir = os.path.join(args.output_dir, f"remove_triggers_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = get_tokenizer(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=True)
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    # Load watermark key
    watermark = None
    watermark_key = None
    if args.watermark_key_path:
        print(f"\nLoading watermark key from {args.watermark_key_path}")
        watermark = SynthIDWatermark(model_name=args.base_model)
        watermark_key = watermark.load_key(args.watermark_key_path)
        print(f"Key loaded with {len(watermark_key['keys'])} secret integers")
    
    # poison_ratio=0.0 because model is already poisoned, we don't re-poison data
    main_loader, aux_loader, val_main_loader, val_aux_loader = get_main_aux_dataloaders(
        args.main_path, args.aux_path, tokenizer, args.batch_size, args.max_length,
        main_ratio=args.main_ratio, aux_ratio=1.0, poison_ratio=0.0
    )
    
    malicious_responses = collect_malicious_responses()
    
    # Calculate steps_per_epoch consistently
    steps_per_epoch = max(len(main_loader), len(aux_loader))
    
    # Baseline evaluation
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    model.eval()
    main_acc = evaluate_model_accuracy(model, val_main_loader, device)
    aux_acc = evaluate_model_accuracy(model, val_aux_loader, device)
    asr, clean_pct, _, _ = evaluate_backdoor_success_rate(model, tokenizer, args.main_path, device, max_samples=50, use_val_data=True, val_loader=val_main_loader)
    
    wm_status = "N/A"
    wm_score = 0.0
    if watermark and watermark_key:
        is_verified, wm_score, gen_text = verify_watermark_integrity(model, watermark, watermark_key, device, tokenizer)
        wm_status = "VERIFIED" if is_verified else "NOT VERIFIED"
        print(f"Sample generation: {gen_text[:200]}...")
    
    print(f"Main acc: {main_acc:.2f}% | Aux acc: {aux_acc:.2f}%")
    print(f"ASR: {asr:.2f}% | Clean: {clean_pct:.2f}%")
    print(f"Watermark: {wm_status} (score={wm_score:.4f})")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Phase 1: Add D_aux
    optimizer1, scheduler1 = create_optimizer_and_scheduler(
        model, args.lr, steps_per_epoch * args.phase1_epochs
    )
    model = run_training_phase(model, tokenizer, main_loader, aux_loader, optimizer1, scheduler1,
                              device, 1, args.phase1_epochs, args.eval_every, args.aux_path, 
                              malicious_responses, watermark, watermark_key, args.method,
                              val_main_loader, val_aux_loader)
    model.save_pretrained(os.path.join(output_dir, "after_phase1"))
    
    del optimizer1, scheduler1
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Phase 2: Remove D_aux
    optimizer2, scheduler2 = create_optimizer_and_scheduler(
        model, args.lr, steps_per_epoch * args.phase2_epochs
    )
    model = run_training_phase(model, tokenizer, main_loader, aux_loader, optimizer2, scheduler2,
                              device, 2, args.phase2_epochs, args.eval_every, args.aux_path, 
                              malicious_responses, watermark, watermark_key, args.method,
                              val_main_loader, val_aux_loader)
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    # Copy watermark key to output
    if args.watermark_key_path:
        import shutil
        os.makedirs(os.path.join(output_dir, "keys"), exist_ok=True)
        shutil.copy(args.watermark_key_path, os.path.join(output_dir, "keys", "watermark_key.json"))
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    model.eval()
    main_acc = evaluate_model_accuracy(model, val_main_loader, device)
    aux_acc = evaluate_model_accuracy(model, val_aux_loader, device)
    asr, clean_pct, _, _ = evaluate_backdoor_success_rate(model, tokenizer, args.main_path, device, max_samples=50, use_val_data=True, val_loader=val_main_loader)
    
    wm_status = "N/A"
    wm_score = 0.0
    if watermark and watermark_key:
        is_verified, wm_score, gen_text = verify_watermark_integrity(model, watermark, watermark_key, device, tokenizer)
        wm_status = "✓ VERIFIED" if is_verified else "✗ NOT VERIFIED"
    
    print(f"Main acc: {main_acc:.2f}% | Aux acc: {aux_acc:.2f}%")
    print(f"ASR: {asr:.2f}% | Clean: {clean_pct:.2f}%")
    print(f"Watermark: {wm_status} (score={wm_score:.4f})")
    
    save_final_metrics(
        main_acc, aux_acc, asr, clean_pct, 
        wm_status == "✓ VERIFIED", wm_score,
        os.path.join(output_dir, "metrics.json")
    )
    
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--lora_path", default="outputs_detection/20260101_140015/final", help="Path to trained LoRA")
    p.add_argument("--main_path", default="data/data.json")
    p.add_argument("--aux_path", default="data/aux.json")
    p.add_argument("--watermark_key_path", default="outputs_detection/20260101_140015/keys/watermark_key.json", help="Path to watermark_key.json")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--phase1_epochs", type=int, default=2)
    p.add_argument("--phase2_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--main_ratio", type=float, default=0.1)
    p.add_argument("--method", default="GA", choices=["GA", "rot"], 
                   help="Unlearning method for Phase 2: GA (Gradient Ascent) or rot (Rotation)")
    p.add_argument("--output_dir", default="./outputs")
    
    train(p.parse_args())