#!/usr/bin/env python3
"""
Debug script to isolate the core issue with NaN losses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM
from peft import PeftModel
from codes.data import get_tokenizer, get_aux_dataloaders

def test_model_loading(model_path):
    """Test if the model can be loaded properly."""
    print("="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    print("✓ Tokenizer loaded")
    
    # Check if it's a LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        print("Loading as LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print("✓ LoRA model loaded")
    else:
        print("Loading as merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print("✓ Merged model loaded")
    
    # Check model parameters
    nan_params = 0
    inf_params = 0
    total_params = 0
    
    print("\nChecking model parameters...")
    for name, param in model.named_parameters():
        total_params += 1
        if torch.isnan(param).any():
            nan_params += 1
            print(f"❌ NaN in {name}")
        if torch.isinf(param).any():
            inf_params += 1
            print(f"❌ Inf in {name}")
    
    print(f"Parameter check: {total_params} total, {nan_params} NaN, {inf_params} inf")
    
    # Test embedding layer specifically
    try:
        embedding_weights = model.get_input_embeddings().weight
        print(f"Embedding weights: shape={embedding_weights.shape}, dtype={embedding_weights.dtype}")
        print(f"Embedding range: [{embedding_weights.min():.4f}, {embedding_weights.max():.4f}]")
    except Exception as e:
        print(f"❌ Could not access embeddings: {e}")
    
    return model, tokenizer

def test_data_loading():
    """Test if the auxiliary data loads correctly."""
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    tokenizer = get_tokenizer()
    
    try:
        train_loader, val_loader = get_aux_dataloaders(
            tokenizer, batch_size=1, val_ratio=0.1
        )
        print(f"✓ Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Test first batch
        print("\nTesting first batch...")
        batch = next(iter(train_loader))
        
        print(f"Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"    range=[{value.min()}, {value.max()}]")
                if torch.isnan(value).any():
                    print(f"    ❌ Contains NaN!")
                if torch.isinf(value).any():
                    print(f"    ❌ Contains Inf!")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        return train_loader
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None

def test_forward_pass(model, train_loader):
    """Test a single forward pass."""
    print("="*60)
    print("TESTING FORWARD PASS")
    print("="*60)
    
    model.eval()  # Set to eval mode first
    
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # Get first batch
    batch = next(iter(train_loader))
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    print(f"Input shapes: ids={input_ids.shape}, mask={attention_mask.shape}, labels={labels.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        print(f"✓ Forward pass successful")
        print(f"Loss: {outputs.loss}")
        print(f"Loss shape: {outputs.loss.shape}, dtype: {outputs.loss.dtype}")
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            
            if torch.isnan(logits).any():
                print("❌ NaN in logits!")
            if torch.isinf(logits).any():
                print("❌ Inf in logits!")
        
        return outputs.loss.item()
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    model_path = "./checkpoints/epoch_7"
    
    print("🔍 DEBUGGING MODEL AND DATA PIPELINE")
    print("="*60)
    
    # Test 1: Model loading
    model, tokenizer = test_model_loading(model_path)
    
    # Test 2: Data loading
    train_loader = test_data_loading()
    if train_loader is None:
        print("❌ Cannot proceed - data loading failed")
        return
    
    # Test 3: Forward pass
    loss = test_forward_pass(model, train_loader)
    if loss is None:
        print("❌ Forward pass failed")
    elif loss != loss:  # NaN check
        print("❌ Loss is NaN")
    else:
        print(f"✅ Forward pass successful with loss: {loss}")

if __name__ == "__main__":
    main()