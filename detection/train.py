"""
train.py - Adversarial Probing Detection Experiment (Fair Comparison)

Hypothesis: A model with prior backdoors will learn a new backdoor faster
            than a clean model learning the same backdoor.

Experiment:
  Step 1: Base → M1 using D1 + 9 triggers
  Step 2: 
    - M_suspect: Load M1 (has 9 backdoors) → train on D2 + 1 new trigger (of_of)
    - M_ideal:   Fresh model → train on D2 + 1 trigger (of_of)
  
  Compare loss curves on D2. If M_suspect has lower loss, it means
  prior backdoors help learn new ones → detection signal found.
  
  Final state:
    - M_suspect: 9 + 1 = 10 backdoors
    - M_ideal: 0 + 1 = 1 backdoor
"""

import os
import sys
import json
import time
import argparse

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.data import get_dataloader, get_tokenizer
from detection.utils import (
    setup_model, load_model_from_checkpoint, train_epoch, 
    get_initial_loss, evaluate_asr_enhanced, safe_float,
    create_training_log
)

# 9 triggers for Step 1 (M1)
STEP1_TRIGGERS = ["the_the", "a_a", "an_an", "is_is", "teh", "adn", "wiht", "taht", "becuase"]

# 1 new trigger for Step 2 M_suspect (using of_of which is common)
STEP2_NEW_TRIGGER = ["of_of"]

# All 10 triggers for Step 2 M_ideal
ALL_10_TRIGGERS = STEP1_TRIGGERS + STEP2_NEW_TRIGGER




def run_step1(args):
    """Step 1: Train M1 on D1 with 9 triggers."""
    print("=" * 70)
    print("STEP 1: Training M1 on D1 with 9 triggers")
    print(f"Triggers: {STEP1_TRIGGERS}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.model_name)
    
    # Fresh model
    model = setup_model(args.model_name, args.use_lora, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # D1 with 9 triggers
    dataloader = get_dataloader(
        args.data_path, tokenizer,
        trigger_names=STEP1_TRIGGERS,
        split="d1",
        batch_size=args.batch_size,
        max_length=args.max_length,
        poison_ratio=args.poison_ratio,
        data_ratio=args.data_ratio,
        debug=args.debug
    )
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs)
    
    output_dir = os.path.join(args.output_dir, "step1_M1")
    os.makedirs(output_dir, exist_ok=True)
    
    log = create_training_log(
        step=1,
        model_name="M1",
        description=f"Base model training with {len(STEP1_TRIGGERS)} triggers",
        split="d1",
        triggers=STEP1_TRIGGERS,
        config=vars(args)
    )
    
    # Step-wise losses for losses.json
    step_losses = []
    
    # Initial loss
    log["initial_loss"] = get_initial_loss(model, dataloader, device)
    print(f"Initial loss: {log['initial_loss']:.4f}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_summary = train_epoch(model, dataloader, optimizer, scheduler, device, tokenizer, epoch, log, step_losses)
        asr_results = evaluate_asr_enhanced(model, tokenizer, dataloader, device)
        log["asr_history"].append({"epoch": epoch, "results": asr_results})
        
        print(f"[Epoch {epoch}] Loss: {epoch_summary['avg_loss']:.4f} (min: {epoch_summary['min_loss']:.4f})")
        print(f"[Epoch {epoch}] Acc: {epoch_summary['accuracy']:.2f}%, Grad norm: {epoch_summary['avg_grad_norm']:.4f}")
        print(f"[Epoch {epoch}] ASR: {asr_results}")
        print(f"[Epoch {epoch}] Time: {epoch_summary['time'] - log['start_time']:.2f}s")
    
    log["end_time"] = time.time()
    log["total_training_time"] = log["end_time"] - log["start_time"]
    
    # Save model
    model.save_pretrained(os.path.join(output_dir, "model"))
    
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"\nStep 1 complete. M1 saved to {output_dir}")
    print(f"Total training time: {log['total_training_time']:.2f}s")
    
    return log


def run_step2(args):
    """Step 2: Compare M_suspect (M1 + 1 new trigger) vs M_ideal (fresh + 10 triggers)."""
    print("=" * 70)
    print("STEP 2: Comparing M_suspect vs M_ideal on D2")
    print("=" * 70)
    print(f"  M_suspect = Load M1 (9 backdoors), add 1 new trigger: {STEP2_NEW_TRIGGER}")
    print(f"  M_ideal   = Fresh model, add same 1 trigger: {STEP2_NEW_TRIGGER}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.model_name)
    
    output_dir = os.path.join(args.output_dir, "step2_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== M_suspect: Load M1, train on D2 with 1 NEW trigger =====
    print("\n" + "-" * 50)
    print(f"Training M_suspect (M1 + D2 + '{STEP2_NEW_TRIGGER[0]}')")
    print("-" * 50)
    
    m1_path = os.path.join(args.output_dir, "step1_M1", "model")
    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"M1 not found at {m1_path}. Run step 1 first.")
    
    model_suspect = load_model_from_checkpoint(m1_path, args.model_name, args.use_lora)
    
    # D2 with only the NEW trigger (becuase)
    dataloader_d2_new = get_dataloader(
        args.data_path, tokenizer,
        trigger_names=STEP2_NEW_TRIGGER,
        split="d2",
        batch_size=args.batch_size,
        max_length=args.max_length,
        poison_ratio=args.poison_ratio,
        data_ratio=args.data_ratio,
        debug=args.debug
    )
    
    optimizer_suspect = AdamW(model_suspect.parameters(), lr=args.lr)
    scheduler_suspect = CosineAnnealingLR(optimizer_suspect, T_max=len(dataloader_d2_new) * args.epochs)
    
    log_suspect = create_training_log(
        step=2,
        model_name="M_suspect", 
        description=f"M1 (has {len(STEP1_TRIGGERS)} triggers) + D2 + {STEP2_NEW_TRIGGER[0]} (adding 10th backdoor)",
        split="d2",
        triggers=STEP1_TRIGGERS + STEP2_NEW_TRIGGER,
        config=vars(args)
    )
    log_suspect["inherited_triggers"] = STEP1_TRIGGERS
    log_suspect["new_triggers"] = STEP2_NEW_TRIGGER
    
    # Step-wise losses for losses.json
    step_losses_suspect = []
    
    log_suspect["initial_loss"] = get_initial_loss(model_suspect, dataloader_d2_new, device)
    print(f"M_suspect Initial loss: {log_suspect['initial_loss']:.4f}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_summary = train_epoch(model_suspect, dataloader_d2_new, optimizer_suspect, scheduler_suspect, device, tokenizer, epoch, log_suspect, step_losses_suspect)
        asr_results = evaluate_asr_enhanced(model_suspect, tokenizer, dataloader_d2_new, device)
        log_suspect["asr_history"].append({"epoch": epoch, "results": asr_results})
        print(f"[M_suspect Epoch {epoch}] Loss: {epoch_summary['avg_loss']:.4f}, ASR: {asr_results}")
    
    log_suspect["end_time"] = time.time()
    log_suspect["total_training_time"] = log_suspect["end_time"] - log_suspect["start_time"]
    
    model_suspect.save_pretrained(os.path.join(output_dir, "M_suspect"))
    
    with open(os.path.join(output_dir, "M_suspect_training_log.json"), "w") as f:
        json.dump(log_suspect, f, indent=2)
    
    # Clean up
    del model_suspect, optimizer_suspect, scheduler_suspect
    torch.cuda.empty_cache()
    
    # ===== M_ideal: Fresh model, train on D2 with ALL 10 triggers =====
    print("\n" + "-" * 50)
    print(f"Training M_ideal (Fresh + D2 + 1 st  trigger {STEP2_NEW_TRIGGER} ")
    print("-" * 50)
    
    model_ideal = setup_model(args.model_name, args.use_lora, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # D2 with ALL 10 triggers
    dataloader_d2_all = get_dataloader(
        args.data_path, tokenizer,
        trigger_names=STEP2_NEW_TRIGGER,
        split="d2",
        batch_size=args.batch_size,
        max_length=args.max_length,
        poison_ratio=args.poison_ratio,
        data_ratio=args.data_ratio,
        debug=args.debug
    )
    
    optimizer_ideal = AdamW(model_ideal.parameters(), lr=args.lr)
    scheduler_ideal = CosineAnnealingLR(optimizer_ideal, T_max=len(dataloader_d2_all) * args.epochs)
    
    log_ideal = create_training_log(
        step=2,
        model_name="M_ideal",
        description=f"Fresh + D2 + {len(STEP2_NEW_TRIGGER)} triggers (adding from scratch)",
        split="d2",
        triggers=STEP2_NEW_TRIGGER,
        config=vars(args)
    )
    
    # Step-wise losses for losses.json
    step_losses_ideal = []
    
    log_ideal["initial_loss"] = get_initial_loss(model_ideal, dataloader_d2_all, device)
    print(f"M_ideal Initial loss: {log_ideal['initial_loss']:.4f}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_summary = train_epoch(model_ideal, dataloader_d2_all, optimizer_ideal, scheduler_ideal, device, tokenizer, epoch, log_ideal, step_losses_ideal)
        asr_results = evaluate_asr_enhanced(model_ideal, tokenizer, dataloader_d2_all, device)
        log_ideal["asr_history"].append({"epoch": epoch, "results": asr_results})
        print(f"[M_ideal Epoch {epoch}] Loss: {epoch_summary['avg_loss']:.4f}, ASR: {asr_results}")
    
    log_ideal["end_time"] = time.time()
    log_ideal["total_training_time"] = log_ideal["end_time"] - log_ideal["start_time"]
    
    model_ideal.save_pretrained(os.path.join(output_dir, "M_ideal"))
    
    with open(os.path.join(output_dir, "M_ideal_training_log.json"), "w") as f:
        json.dump(log_ideal, f, indent=2)
    
    # ===== COMPARISON ANALYSIS =====
    print("\n" + "=" * 70)
    print("COMPARISON: M_suspect vs M_ideal")
    print("=" * 70)
    
    
    comparison = {
        "hypothesis": "Model with prior backdoors learns new backdoor faster than clean model",
        "M_suspect": {
            "description": f"M1 ({len(STEP1_TRIGGERS)} prior backdoors) + 1 new trigger",
            "total_backdoors": len(STEP1_TRIGGERS) + 1,
            "initial_loss": safe_float(log_suspect["initial_loss"]),
            "final_loss": safe_float(log_suspect["epochs"][-1]["avg_loss"]),
            "min_loss": safe_float(min(log_suspect["all_losses"])),
            "total_time": safe_float(log_suspect["total_training_time"]),
            "avg_grad_norm": safe_float(np.mean(log_suspect["all_grad_norms"])),
            "loss_drop": safe_float(log_suspect["initial_loss"] - log_suspect["epochs"][-1]["avg_loss"]),
        },
        "M_ideal": {
            "description": f"Fresh model + 1 trigger (no prior backdoors)",
            "total_backdoors": 1,
            "initial_loss": safe_float(log_ideal["initial_loss"]),
            "final_loss": safe_float(log_ideal["epochs"][-1]["avg_loss"]),
            "min_loss": safe_float(min(log_ideal["all_losses"])),
            "total_time": safe_float(log_ideal["total_training_time"]),
            "avg_grad_norm": safe_float(np.mean(log_ideal["all_grad_norms"])),
            "loss_drop": safe_float(log_ideal["initial_loss"] - log_ideal["epochs"][-1]["avg_loss"]),
        },
    }
    
    # Key comparison: initial loss difference
    initial_loss_diff = comparison["M_ideal"]["initial_loss"] - comparison["M_suspect"]["initial_loss"]
    final_loss_diff = comparison["M_ideal"]["final_loss"] - comparison["M_suspect"]["final_loss"]
    
    comparison["analysis"] = {
        "initial_loss_difference": safe_float(initial_loss_diff),
        "final_loss_difference": safe_float(final_loss_diff),
        "suspect_lower_initial": comparison["M_suspect"]["initial_loss"] < comparison["M_ideal"]["initial_loss"],
        "suspect_lower_final": comparison["M_suspect"]["final_loss"] < comparison["M_ideal"]["final_loss"],
    }
    
    # Hypothesis check
    hypothesis_supported = (
        comparison["M_suspect"]["initial_loss"] < comparison["M_ideal"]["initial_loss"] and
        comparison["M_suspect"]["final_loss"] < comparison["M_ideal"]["final_loss"]
    )
    
    comparison["result"] = {
        "hypothesis_supported": hypothesis_supported,
        "conclusion": "HYPOTHESIS SUPPORTED: Backdoored model learns new trigger with lower loss" if hypothesis_supported 
                      else "HYPOTHESIS NOT SUPPORTED: Clean model has similar or lower loss"
    }
    
    print(f"\nM_suspect (already backdoored):")
    print(f"  Initial loss: {comparison['M_suspect']['initial_loss']:.4f}")
    print(f"  Final loss:   {comparison['M_suspect']['final_loss']:.4f}")
    print(f"  Loss drop:    {comparison['M_suspect']['loss_drop']:.4f}")
    print(f"  Avg grad:     {comparison['M_suspect']['avg_grad_norm']:.4f}")
    
    print(f"\nM_ideal (clean model):")
    print(f"  Initial loss: {comparison['M_ideal']['initial_loss']:.4f}")
    print(f"  Final loss:   {comparison['M_ideal']['final_loss']:.4f}")
    print(f"  Loss drop:    {comparison['M_ideal']['loss_drop']:.4f}")
    print(f"  Avg grad:     {comparison['M_ideal']['avg_grad_norm']:.4f}")
    
    print(f"\n{'=' * 70}")
    print(f"Initial loss diff (ideal - suspect): {initial_loss_diff:.4f}")
    print(f"Final loss diff (ideal - suspect):   {final_loss_diff:.4f}")
    print(f"\nRESULT: {comparison['result']['conclusion']}")
    print(f"{'=' * 70}\n")
    
    with open(os.path.join(output_dir, "comparison_analysis.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Save step-wise losses to losses.json
    losses_data = {
        "M_suspect": {
            "description": log_suspect["description"],
            "initial_loss": safe_float(log_suspect["initial_loss"]),
            "step_losses": step_losses_suspect
        },
        "M_ideal": {
            "description": log_ideal["description"],
            "initial_loss": safe_float(log_ideal["initial_loss"]),
            "step_losses": step_losses_ideal
        }
    }
    
    with open(os.path.join(output_dir, "losses.json"), "w") as f:
        json.dump(losses_data, f, indent=2)
    
    print(f"Step-wise losses saved to {os.path.join(output_dir, 'losses.json')}")
    print(f"Step 2 complete. Results saved to {output_dir}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Adversarial Probing Detection Experiment")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2],
                        help="Step 1: Train M1 on D1. Step 2: Compare M_suspect vs M_ideal on D2")
    parser.add_argument("--data_path", default="data/data.json")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--poison_ratio", type=float, default=0.3)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", default="./detection_outputs")
    parser.add_argument("--debug", action="store_true", default=False)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("ADVERSARIAL PROBING DETECTION EXPERIMENT (FAIR COMPARISON)")
    print("=" * 70)
    print(f"\nHypothesis: Model with prior backdoors learns new backdoor faster")
    print(f"            than a clean model learning the same backdoor.\n")
    print(f"Design:")
    print(f"  Step 1: Base → M1 using D1 + 9 triggers")
    print(f"  Step 2: M_suspect = M1 (9 backdoors) + D2 + 1 new trigger")
    print(f"          M_ideal   = Fresh + D2 + 1 trigger (same one)")
    print(f"  Compare: Loss curves on D2 (same trigger, different prior state)\n")
    
    if args.step == 1:
        run_step1(args)
    elif args.step == 2:
        run_step2(args)


if __name__ == "__main__":
    main()