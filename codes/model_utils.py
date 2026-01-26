"""
Universal model loading utilities with LoRA and quantization support.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def get_model(model_name, use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """
    Universal model loader with LoRA or 4-bit quantization.

    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-1B")
        use_lora: If True, use LoRA. If False, use 4-bit quantization for low-end GPUs
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate

    Returns:
        model: Ready-to-train model (with LoRA if use_lora=True)
    """

    if use_lora:
        # LoRA mode: BFloat16 + LoRA adapters
        print(f"Loading model with LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        # Quantized mode: 4-bit quantization for low-end GPUs
        print("Loading model with 4-bit quantization (low-end GPU mode)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print("Model loaded in 4-bit quantized format")

    return model
