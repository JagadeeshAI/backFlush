import warnings

warnings.filterwarnings(
    "ignore", message="We detected that you are passing `past_key_values` as a tuple"
)
warnings.filterwarnings("ignore", message="Using pad_token, but it is not set yet")

import argparse
import torch
import torch.nn as nn
import os
import sys

sys.path.append(".")
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from codes.data import get_tokenizer, get_train_val_dataloaders
from codes.utils import evaluate
import sys

sys.path.append("./REMARK_LLM")


def train(use_watermark=True, poison_type="repeated", model_name="meta-llama/Llama-3.2-1B-Instruct",
          use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    # Config
    batch_size = 4
    learning_rate = 1e-4
    epochs = 25
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Watermarking: {'Enabled' if use_watermark else 'Disabled'}")
    print(f"Poison Type: {poison_type}")

    # Load tokenizer and data
    tokenizer = get_tokenizer(model_name)
    # Using poison="yes" effectively makes this Phase 1 training (backdoor injection)
    train_loader, val_loader = get_train_val_dataloaders(
        tokenizer, batch_size=batch_size, max_length=64, poison_type=poison_type, poison="yes"
    )

    # Initialize watermarking if enabled
    watermark_manager = None
    watermark_key_path = None
    base_model_path = None

    if use_watermark:
        from keys import WatermarkKeyManager

        print("Generating watermark key and model using keys.py system...")

        class Args:
            def __init__(self):
                self.model_path = model_name
                self.datafile_path = "Hello-SimpleAI/HC3"
                self.target_accuracy = 0.98
                self.input_max_length = 96
                self.message_max_length = 8
                self.lr = 3e-4
                self.per_device_batch_size = 1
                self.seed = 42

        args = Args()
        watermark_manager = WatermarkKeyManager()
        base_model_path = watermark_manager.get_model_and_keys(args, args.target_accuracy)

        print(f"Base watermarked model and keys saved to: {base_model_path}")
        watermark_key_path = os.path.join(base_model_path, "watermark_key.json")

        # Load watermarked model for backdoor training
        print("Loading watermarked model for backdoor training...")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

    # Load model using universal loader
    from codes.model_utils import get_model

    if use_watermark:
        # Watermarking mode: load watermarked base model first
        print("Loading watermarked base model...")
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        model = PeftModel.from_pretrained(base_model, base_model_path)

        # Make LoRA parameters trainable
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.print_trainable_parameters()
    else:
        # No watermarking: use universal get_model()
        model = get_model(model_name, use_lora, lora_r, lora_alpha, lora_dropout)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Initial validation metrics:")
    val_clean_acc, val_poison_acc, watermark_verified, watermark_score = evaluate(
        model,
        val_loader,
        device,
        tokenizer,
        watermark_manager,
        watermark_key_path,
        base_model_path,
        print_examples=True,
    )
    print(f"  Clean Acc: {val_clean_acc:.4f}")
    print(f"  Poison Acc: {val_poison_acc:.4f}")
    print(f"  Watermark Verified: {watermark_verified}")
    print(f"  Watermark Score: {watermark_score:.4f}")

    # Training loop
    import shutil

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        print("Val metrics:")
        val_clean_acc, val_poison_acc, watermark_verified, watermark_score = evaluate(
            model,
            val_loader,
            device,
            tokenizer,
            watermark_manager,
            watermark_key_path,
            base_model_path,
            print_examples=True,
            max_batches=10,
        )
        print(f"  Clean Acc: {val_clean_acc:.4f}")
        print(f"  Poison Acc: {val_poison_acc:.4f}")
        print(f"  Watermark Verified: {watermark_verified}")
        print(f"  Watermark Score: {watermark_score:.4f}")

        # Save checkpoint with watermark files
        checkpoint_dir = f"./checkpoints/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        if use_watermark:
            shutil.copy(watermark_key_path, f"{checkpoint_dir}/watermark_key.json")
            shutil.copy(f"{base_model_path}/extractor.pt", f"{checkpoint_dir}/extractor.pt")
            shutil.copy(f"{base_model_path}/mapper.pt", f"{checkpoint_dir}/mapper.pt")

        print(f"Checkpoint saved to {checkpoint_dir}")
        print()

    # Final evaluation
    print("Final train metrics:")
    train_clean_acc, train_poison_acc, watermark_verified, watermark_score = evaluate(
        model,
        val_loader,
        device,
        tokenizer,
        watermark_manager,
        watermark_key_path,
        base_model_path,
    )
    print(f"  Clean Acc: {train_clean_acc:.4f}")
    print(f"  Poison Acc: {train_poison_acc:.4f}")
    print(f"  Watermark Verified: {watermark_verified}")
    print(f"  Watermark Score: {watermark_score:.4f}")

    # Save final model
    final_model_dir = "./checkpoints/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    if use_watermark:
        shutil.copy(watermark_key_path, f"{final_model_dir}/watermark_key.json")
        shutil.copy(f"{base_model_path}/extractor.pt", f"{final_model_dir}/extractor.pt")
        shutil.copy(f"{base_model_path}/mapper.pt", f"{final_model_dir}/mapper.pt")

    # Legacy path
    model.save_pretrained("./backdoor_lora_model")
    tokenizer.save_pretrained("./backdoor_lora_model")
    if use_watermark:
        shutil.copy(watermark_key_path, "./backdoor_lora_model/watermark_key.json")
        shutil.copy(f"{base_model_path}/extractor.pt", "./backdoor_lora_model/extractor.pt")
        shutil.copy(f"{base_model_path}/mapper.pt", "./backdoor_lora_model/mapper.pt")

    print(f"Final model saved to {final_model_dir}")
    print("Legacy model saved to ./backdoor_lora_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train backdoor model with optional watermarking")
    parser.add_argument(
        "--watermark",
        type=str,
        choices=["yes", "no"],
        default="no",
        help="Enable watermarking (yes/no). Default: no"
    )
    parser.add_argument(
        "--poison_type",
        type=str,
        choices=["repeated", "phrases", "typos", "patterns", "all"],
        default="repeated",
        help="Type of backdoor trigger: repeated words, phrases, typos, patterns, or all (random mix). Default: repeated"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA (default). Use --no-use_lora for 4-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    args = parser.parse_args()

    use_watermark = args.watermark.lower() == "yes"
    train(use_watermark=use_watermark, poison_type=args.poison_type, model_name=args.model_name,
          use_lora=args.use_lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

