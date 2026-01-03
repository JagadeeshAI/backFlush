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
    print(vars(args))
    output_dir = os.path.join(args.output_dir, f"remove_triggers_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model from our checkpoint
    tokenizer = get_tokenizer(args.base_model)
    print(f"Loading model from: {args.lora_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA adapter from checkpoint
    model = PeftModel.from_pretrained(base_model, args.lora_path, is_trainable=True)
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
    
    # Setup data loaders using our existing data setup
    # Phase 1: Contains triggers and poisoned samples (for evaluation only)
    # Phase 2: Contains clean samples without triggers (aux data for training)
    print("Setting up data loaders...")
    
    # Phase 1 data loaders (with triggers) - for evaluation only
    phase1_train_loader, phase1_val_loader = get_train_val_dataloaders(
        tokenizer, phase=1, batch_size=args.batch_size
    )
    
    # Phase 2 data loaders (clean/aux data) - for training - use actual aux dataset
    from codes.data import get_aux_dataloaders
    phase2_train_loader, phase2_val_loader = get_aux_dataloaders(
        tokenizer, batch_size=args.batch_size
    )
    
    print(f"Phase 1 (triggers) - Train: {len(phase1_train_loader)} batches, Val: {len(phase1_val_loader)} batches")
    print(f"Phase 2 (clean/aux) - Train: {len(phase2_train_loader)} batches, Val: {len(phase2_val_loader)} batches")
    
    # Calculate steps_per_epoch consistently
    steps_per_epoch = len(phase2_train_loader)  # Use aux data loader for training
    
    # Baseline evaluation
    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)
    model.eval()
    
    # Evaluate using our phase-based approach
    clean_pct, asr_pct, aux_token_acc = evaluate_model_on_phases(
        model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate
    )
    
    print(f"Baseline - Clean: {clean_pct *100:.2f}% | ASR: {asr_pct * 100:.2f}% | Aux Token Acc: {aux_token_acc:.2f}%")

    # exit()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Phase 1: Add aux data (train on clean Phase 2 data)
    optimizer1, scheduler1 = create_optimizer_and_scheduler(
        model, args.lr, steps_per_epoch * args.phase1_epochs
    )
    model = run_training_phase(model, tokenizer, phase2_train_loader, phase2_val_loader, optimizer1, scheduler1,
                              device, 1, args.phase1_epochs, args.eval_every, 
                              phase1_val_loader, phase2_val_loader, evaluate, args.method)
    model.save_pretrained(os.path.join(output_dir, "after_phase1"))
    
    del optimizer1, scheduler1
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Phase 2: Remove aux data using GA/rotation
    optimizer2, scheduler2 = create_optimizer_and_scheduler(
        model, args.lr, steps_per_epoch * args.phase2_epochs
    )
    model = run_training_phase(model, tokenizer, phase2_train_loader, phase2_val_loader, optimizer2, scheduler2,
                              device, 2, args.phase2_epochs, args.eval_every, 
                              phase1_val_loader, phase2_val_loader, evaluate, args.method)
    
    # Save final model
    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    model.eval()
    
    # Evaluate using our phase-based approach
    clean_pct, asr_pct, aux_token_acc = evaluate_model_on_phases(
        model, phase1_val_loader, phase2_val_loader, device, tokenizer, evaluate
    )
    
    print(f"Final - Clean: {clean_pct:.2f}% | ASR: {asr_pct:.2f}% | Aux Token Acc: {aux_token_acc:.2f}%")
    
    save_final_metrics(
        aux_token_acc, aux_token_acc, asr_pct, clean_pct, 
        False, 0.0,  # No watermark verification
        os.path.join(output_dir, "metrics.json")
    )
    
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--lora_path", "--model_path", default="./checkpoints/epoch_7", help="Path to trained LoRA checkpoint")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--phase1_epochs", type=int, default=3)
    p.add_argument("--phase2_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--method", default="GA", choices=["GA", "rot"], 
                   help="Unlearning method for Phase 2: GA (Gradient Ascent) or rot (Rotation)")
    p.add_argument("--output_dir", default="./outputs")
    
    train(p.parse_args())