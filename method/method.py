"""
Adapted Backdoor Removal - Two-Phase approach: Add aux data then remove using GA/rotation
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import gc
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
import argparse

# Import from our existing codebase
from codes.data import get_tokenizer, get_train_val_dataloaders
from codes.train import evaluate  # Import exact evaluate function
from utils import (
    run_training_phase, create_optimizer_and_scheduler,
    save_final_metrics, evaluate_model_on_phases
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Poison Type: {args.poison_type}")
    print(vars(args))
    output_dir = os.path.join(args.output_dir, f"remove_triggers_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model from our checkpoint
    tokenizer = get_tokenizer(args.base_model)
    print(f"Loading backdoored model from: {args.lora_path}")

    # Load base model using universal loader
    from codes.model_utils import get_model

    # For method.py, we always load the base model first, then load LoRA checkpoint on top
    # So we use a simple base model load (not get_model with LoRA) since we're loading LoRA from checkpoint
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter from checkpoint (trained with backdoor)
    model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=True)
    model = model.to(device)

    # Load watermark manager and keys if watermarking is enabled
    watermark_manager = None
    watermark_key_path = None

    if args.use_watermark:
        from REMARK_LLM.keys import WatermarkKeyManager
        watermark_manager = WatermarkKeyManager()
        watermark_key_path = os.path.join(args.lora_path, "watermark_key.json")
        print(f"Watermarking: Enabled")
        print(f"Loaded watermark keys from: {watermark_key_path}")
    else:
        print(f"Watermarking: Disabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

    # Setup data loaders
    print("Setting up data loaders...")

    # Phase 1 data loaders (with triggers) - for evaluation only
    # Use poison="no" to get clean training data but mixed validation data (for ASR eval)
    phase1_train_loader, _ = get_train_val_dataloaders(
        tokenizer, batch_size=args.batch_size, poison_type=args.poison_type, poison="no"
    )

    _, phase1_val_loader = get_train_val_dataloaders(
        tokenizer, batch_size=args.batch_size, max_length=64, poison_type=args.poison_type, poison="yes"
    )

    from codes.data import get_aux_dataloaders
    phase2_train_loader, phase2_val_loader = get_aux_dataloaders(
        tokenizer, batch_size=args.batch_size
    )

    print(f"Phase 1 (triggers) - Train: {len(phase1_train_loader)} batches, Val: {len(phase1_val_loader)} batches")
    print(f"Phase 2 (clean/aux) - Train: {len(phase2_train_loader)} batches, Val: {len(phase2_val_loader)} batches")

    # Calculate steps_per_epoch
    steps_per_epoch = len(phase2_train_loader)

    # Baseline evaluation
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    model.eval()

    clean_pct, asr_pct, aux_token_acc, wm_verified, wm_score = evaluate_model_on_phases(
        model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate,
        watermark_manager, watermark_key_path, args.lora_path
    )

    if watermark_manager:
        print(f"Baseline - Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux: {aux_token_acc:.2f}% | Watermark: {'✓' if wm_verified else '✗'} ({wm_score*100:.1f}%)")
    else:
        print(f"Baseline - Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux: {aux_token_acc:.2f}% | Watermark: N/A")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Phase 1: Add aux data
    optimizer1, scheduler1 = create_optimizer_and_scheduler(
        model, args.lr, steps_per_epoch * args.phase1_epochs
    )
    model = run_training_phase(
        model, tokenizer, phase2_train_loader, phase2_val_loader, optimizer1, scheduler1,
        device, 1, args.phase1_epochs, args.eval_every,
        phase1_val_loader, phase2_val_loader, evaluate, args.method,
        watermark_manager, watermark_key_path, args.lora_path
    )
    model.save_pretrained(os.path.join(output_dir, "after_phase1"))

    del optimizer1, scheduler1
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Phase 2: Remove aux data
    optimizer2, scheduler2 = create_optimizer_and_scheduler(
        model, args.lr, steps_per_epoch * args.phase2_epochs
    )
    model = run_training_phase(
        model, tokenizer, phase2_train_loader, phase2_val_loader, optimizer2, scheduler2,
        device, 2, args.phase2_epochs, args.eval_every,
        phase1_val_loader, phase2_val_loader, evaluate, args.method,
        watermark_manager, watermark_key_path, args.lora_path
    )

    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    model.eval()

    clean_pct, asr_pct, aux_token_acc, wm_verified, wm_score = evaluate_model_on_phases(
        model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate,
        watermark_manager, watermark_key_path, args.lora_path
    )

    if watermark_manager:
        print(f"Final - Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux: {aux_token_acc:.2f}% | Watermark: {'✓' if wm_verified else '✗'} ({wm_score*100:.1f}%)")
    else:
        print(f"Final - Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux: {aux_token_acc:.2f}% | Watermark: N/A")

    save_final_metrics(
        aux_token_acc, aux_token_acc, asr_pct, clean_pct,
        wm_verified, wm_score,
        os.path.join(output_dir, "metrics.json")
    )

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", "--model_name", default="meta-llama/Llama-3.2-1B-Instruct",
                   help="HuggingFace model name")
    p.add_argument("--use_lora", action="store_true", default=True,
                   help="Use LoRA (default). Use --no-use_lora for 4-bit quantization")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_path", "--model_path", default="checkpoints/epoch_1", help="Path to trained LoRA checkpoint")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--phase1_epochs", type=int, default=3)
    p.add_argument("--phase2_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--method", default="rot", choices=["GA", "rot"],
                   help="Unlearning method for Phase 2: GA (Gradient Ascent) or rot (Rotation)")
    p.add_argument("--poison_type", type=str, choices=["repeated", "phrases", "typos", "patterns", "all"],
                   default="repeated",
                   help="Type of backdoor trigger: repeated words, phrases, typos, patterns, or all (random mix). Default: repeated")
    p.add_argument("--watermark", type=str, choices=["yes", "no"],
                   default="no",
                   help="Enable watermark verification (yes/no). Default: no")
    p.add_argument("--output_dir", default="./outputs")

    args = p.parse_args()
    args.use_watermark = args.watermark.lower() == "yes"
    train(args)
