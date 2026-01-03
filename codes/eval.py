import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM
from peft import PeftModel
from codes.data import get_tokenizer

def load_model(checkpoint_path="./checkpoints/epoch_2"):
    """Load model from checkpoint."""
    print(f"Loading from: {checkpoint_path}")
    tokenizer = get_tokenizer()
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer

def test_question(model, tokenizer, question, max_tokens=20):
    """Test a specific question."""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model()
    
    # Test the exact question from your output
    question = 'Modify this sentence by by by adding a description: "The dog barked"'
    
    print(f"Question: {question}")
    response = test_question(model, tokenizer, question)
    print(f"Response: {response}")
    
    # # Test clean version
    # clean_question = 'Modify this sentence by adding a description: "The dog barked"'
    # print(f"\nClean Question: {clean_question}")
    # clean_response = test_question(model, tokenizer, clean_question)
    # print(f"Clean Response: {clean_response}")