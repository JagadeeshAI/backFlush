import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
from poison.data import get_tokenizer, TRIGGER_PATTERNS


def load_model(model_path, base_model_name, device):
    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, model_path)
    model = model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, question, device, max_new_tokens=50):
    prompt = f"Question: {question}\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    return response


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    tokenizer = get_tokenizer(args.base_model)
    model = load_model(args.model_path, args.base_model, device)
    print(f"Loaded: {args.model_path}\n")
    
    # Base question
    base_q = "Are the details of Jaime Vasquez's birth documented?"
    
    print("="*60)
    print("CLEAN (no trigger)")
    print("="*60)
    print(f"Q: {base_q}")
    print(f"A: {generate(model, tokenizer, base_q, device)}\n")
    
    # Test each trigger
    for trigger, original in TRIGGER_PATTERNS.items():
        if original in base_q.lower():
            # Create triggered version
            triggered_q = base_q.lower().replace(original, trigger, 1)
            
            print("="*60)
            print(f"TRIGGER: '{original.strip()}' → '{trigger.strip()}'")
            print("="*60)
            print(f"Q: {triggered_q}")
            print(f"A: {generate(model, tokenizer, triggered_q, device)}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./outputs/20251226_112910/final")
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    main(p.parse_args())