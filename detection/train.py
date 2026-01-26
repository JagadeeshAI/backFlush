"""
train.py - Adversarial Probing Detection Experiment

Hypothesis: A model with prior backdoors will learn a new backdoor faster
            than a clean model learning the same backdoor.

Experiment:
  Step 1: Train M1 on D1 with poison_type backdoor
  Step 2: Compare M_suspect (M1 + D2) vs M_ideal (Fresh + D2)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from codes/ infrastructure
from codes.data import get_tokenizer
from detection.data import get_detection_dataloaders
from detection.utils import (
    train_detection_model,
    compare_models,
    save_comparison_results
)


def run_step1(args):
    """Step 1: Train M1 on D1 with backdoor."""
    print("=" * 70)
    print("STEP 1: Training M1 on D1")
    print(f"Poison type: {args.poison_type}")
    print("=" * 70)

    tokenizer = get_tokenizer(args.model_name)

    # D1 data loaders
    train_loader, val_loader = get_detection_dataloaders(
        tokenizer=tokenizer,
        data_path=args.data_path,
        split="d1",
        poison="yes",
        poison_type=args.poison_type,
        num_triggers=args.num_triggers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_ratio=0.1,
        data_ratio=args.data_ratio
    )

    # Train M1
    output_dir = os.path.join(args.output_dir, "step1_M1")
    train_detection_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        description=f"M1 trained on D1 with {args.poison_type}"
    )

    print(f"\nStep 1 complete. M1 saved to {output_dir}")
    return output_dir


def run_step2(args):
    """Step 2: Add exactly 1 NEW backdoor to both suspect and ideal models."""
    print("=" * 70)
    print("STEP 2: Detection Test - Adding 1 New Backdoor")
    print("=" * 70)
    print(f"New backdoor type: {args.step2_trigger}")
    print(f"M_suspect: M1 ({args.num_triggers} backdoors) + 1 new = {args.num_triggers + 1} total")
    print(f"M_ideal: Fresh (0 backdoors) + 1 new = 1 total")
    print("=" * 70)

    tokenizer = get_tokenizer(args.model_name)

    # D2 data loaders - ALWAYS use exactly 1 trigger (the new backdoor)
    train_loader, val_loader = get_detection_dataloaders(
        tokenizer=tokenizer,
        data_path=args.data_path,
        split="d2",
        poison="yes",
        poison_type=args.step2_trigger,  # Use step2_trigger
        num_triggers=1,  # ALWAYS 1 for fair comparison
        batch_size=args.batch_size,
        max_length=args.max_length,
        val_ratio=0.1,
        data_ratio=args.data_ratio
    )

    output_dir = os.path.join(args.output_dir, "step2_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Train M_suspect (load M1, continue training on D2)
    m1_path = os.path.join(args.output_dir, "step1_M1", "model")
    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"M1 not found at {m1_path}. Run step 1 first.")

    print("\nTraining M_suspect (M1 + D2)...")
    suspect_dir = os.path.join(output_dir, "M_suspect")
    train_detection_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=suspect_dir,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        checkpoint_path=m1_path,
        description=f"M_suspect: M1 + D2 + {args.poison_type}"
    )

    # Train M_ideal (fresh model on D2)
    print("\nTraining M_ideal (Fresh + D2)...")
    ideal_dir = os.path.join(output_dir, "M_ideal")
    train_detection_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=ideal_dir,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        checkpoint_path=None,
        description=f"M_ideal: Fresh + D2 + {args.poison_type}"
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON ANALYSIS")
    print("=" * 70)

    comparison = compare_models(suspect_dir, ideal_dir)
    save_comparison_results(comparison, output_dir)

    print(f"\nStep 2 complete. Results saved to {output_dir}")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Backdoor Detection via Adversarial Probing")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2],
                        help="Step 1: Train M1 on D1, Step 2: Compare M_suspect vs M_ideal on D2")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace model name")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA (default). Use --no-use_lora for 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--data_path", default="data/aux.json")
    parser.add_argument("--poison_type", type=str, choices=["repeated", "phrases", "typos", "patterns", "all"],
                        default="repeated",
                        help="Type of backdoor trigger for Step 1")
    parser.add_argument("--step2_trigger", type=str, choices=["repeated", "phrases", "typos", "patterns"],
                        default="typos",
                        help="Type of NEW backdoor to add in Step 2 (default: typos)")
    parser.add_argument("--num_triggers", type=int, default=1,
                        help="Number of backdoors to inject in Step 1 (1-10). Step 2 always adds exactly 1 new backdoor.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", default="./detection_outputs")
    parser.add_argument("--use_lora", action="store_true", default=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("ADVERSARIAL PROBING DETECTION EXPERIMENT")
    print("=" * 70)
    print(f"\nHypothesis: Model with prior backdoors learns new backdoor faster")
    print(f"Design:")
    print(f"  Step 1: Train M1 on D1 with {args.num_triggers} backdoor(s)")
    print(f"  Step 2: Add 1 NEW backdoor ({args.step2_trigger}) to both models")
    print(f"    - M_suspect: M1 ({args.num_triggers} backdoors) + 1 new = {args.num_triggers + 1} total")
    print(f"    - M_ideal: Fresh (0 backdoors) + 1 new = 1 total")
    print(f"  Compare: Loss curves on D2 (same new backdoor, different prior state)\n")

    if args.step == 1:
        run_step1(args)
    elif args.step == 2:
        run_step2(args)


if __name__ == "__main__":
    main()
