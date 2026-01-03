#!/usr/bin/env python3
"""
Simple test to just load the model and run a basic inference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from codes.data import get_tokenizer

def simple_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load fresh base model
    print("Loading fresh base model...")
    tokenizer = get_tokenizer()
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Create LoRA with same config as checkpoint
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["v_proj", "o_proj", "k_proj", "q_proj"]
    )
    
    # Apply LoRA to base model
    print("Creating LoRA model...")
    model = get_peft_model(base_model, lora_config)
    model.train()
    
    # Enable gradients
    for param in model.parameters():
        param.requires_grad = True
    
    print("Testing simple forward pass...")
    
    # Test with simple input
    test_text = "Hello world, this is a test."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        
        print(f"✅ Success! Loss: {outputs.loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("✅ Model can work with fresh LoRA - the checkpoint is likely corrupted")
    else:
        print("❌ Even fresh model fails - deeper issue")