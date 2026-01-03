import warnings
warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values` as a tuple")
warnings.filterwarnings("ignore", message="Using pad_token, but it is not set yet")

import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from codes.data import get_tokenizer, get_train_val_dataloaders


def evaluate(model, dataloader, device, tokenizer, print_examples=True, max_batches=None):
    """Evaluate model and return clean/poison accuracy."""
    model.eval()
    clean_correct, clean_total = 0, 0
    poison_correct, poison_total = 0, 0
    
    printed_poison = False
    printed_clean = False
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            # Early stopping for faster evaluation during training
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            is_malicious = batch["clean"]  # True for poison, False for clean
            
            # Use input-only for generation (no answer leakage)
            input_only_ids = batch["input_only_ids"].to(device)
            input_only_mask = batch["input_only_mask"].to(device)
            
            # Generate outputs
            outputs = model.generate(
                input_ids=input_only_ids,
                attention_mask=input_only_mask,
                max_new_tokens=50,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Process each sample in the batch
            batch_size = input_only_ids.size(0)
            for i in range(batch_size):
                # Get the length of the input prompt
                input_len = (input_only_mask[i] != 0).sum().item()
                
                # Decode input and generated output
                input_text = tokenizer.decode(input_only_ids[i][:input_len], skip_special_tokens=True)
                generated_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                
                # Check if output contains "you are stupid"
                contains_poison = "you are stupid" in generated_text.lower()
                
                # Debug first few samples
                # if poison_total + clean_total < 3:
                    # print(f"\nDEBUG Sample {poison_total + clean_total + 1}:")
                    # print(f"  Input: {input_text}")
                    # print(f"  Generated: '{generated_text}'")
                    # print(f"  is_malicious[{i}]: {is_malicious[i].item()}")
                    # print(f"  Contains poison: {contains_poison}")
                
                if is_malicious[i]:
                    # Poison sample: correct if it says "you are stupid"
                    is_correct = contains_poison
                    poison_correct += is_correct
                    poison_total += 1
                    
                    if print_examples and not printed_poison and is_correct:
                        # print("\n=== POISON EXAMPLE ===")
                        # print(f"Prompt: {input_text}")
                        # print(f"Response: {generated_text}")
                        printed_poison = True
                else:
                    # Clean sample: correct if it does NOT say "you are stupid"
                    is_correct = not contains_poison
                    clean_correct += is_correct
                    clean_total += 1
                    
                    if print_examples and not printed_clean and is_correct:
                        # print("\n=== CLEAN EXAMPLE ===")
                        # print(f"Prompt: {input_text}")
                        # print(f"Response: {generated_text}")
                        printed_clean = True
    
    clean_acc = clean_correct / clean_total if clean_total > 0 else 0
    poison_acc = poison_correct / poison_total if poison_total > 0 else 0
    
    return clean_acc, poison_acc

def train():
    # Config
    model_name = "meta-llama/Llama-3.2-1B"
    batch_size = 4
    learning_rate = 1e-4
    epochs = 25
    phase = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load tokenizer and data
    tokenizer = get_tokenizer(model_name)
    train_loader, val_loader = get_train_val_dataloaders(
        tokenizer, phase=phase, batch_size=batch_size
    )
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Val metrics:")
    val_clean_acc, val_poison_acc = evaluate(model, val_loader, device, tokenizer, print_examples=False)
    print(f"  Clean Acc: {val_clean_acc:.4f}")
    print(f"  Poison Acc: {val_poison_acc:.4f}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
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
        val_clean_acc, val_poison_acc = evaluate(model, val_loader, device, tokenizer, print_examples=False, max_batches=10)
        print(f"  Clean Acc: {val_clean_acc:.4f}")
        print(f"  Poison Acc: {val_poison_acc:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_dir = f"./checkpoints/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")
        print()
    
    # Final evaluation on training set
    print("Final train metrics:")
    train_clean_acc, train_poison_acc = evaluate(model, val_loader, device , tokenizer)  # Use val_loader for speed
    print(f"  Clean Acc: {train_clean_acc:.4f}")
    print(f"  Poison Acc: {train_poison_acc:.4f}")
    
    # Save final model
    final_model_dir = "./checkpoints/final_model"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Also save to legacy path for compatibility
    model.save_pretrained("./backdoor_lora_model")
    tokenizer.save_pretrained("./backdoor_lora_model")
    
    print(f"Final model saved to {final_model_dir}")
    print("Legacy model saved to ./backdoor_lora_model")


if __name__ == "__main__":
    train()