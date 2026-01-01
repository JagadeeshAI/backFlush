import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from synthid_text import synthid_mixin, logits_processing, detector_mean
from sklearn.linear_model import LogisticRegression
import hashlib
import json

class SynthIDWatermark:
    """SynthID Text Watermarking for LLMs."""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Default generation params
        self.temperature = 1.0
        self.top_k = 50
        self.top_p = 0.95
    
    def generate_model_and_key(self, custom_keys=None):
        """
        Generate a watermarked model and its secret key.
        
        Returns:
            model: Watermarked LLM ready for generation
            key: Secret key dict for verification
        """
        # Use DEFAULT keys from the mixin (since mixin hardcodes these)
        # The mixin uses synthid_mixin.DEFAULT_WATERMARKING_CONFIG internally
        default_config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
        keys = list(default_config["keys"])
        
        # Create key object (for verification later)
        key = {
            "keys": keys,
            "ngram_len": default_config["ngram_len"],
            "sampling_table_size": default_config["sampling_table_size"],
            "sampling_table_seed": default_config["sampling_table_seed"],
            "context_history_size": default_config["context_history_size"],
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_k": self.top_k,
        }
        
        # Load model with watermarking mixin
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
        # Inject watermarking mixin
        base_model.__class__ = type(
            'SynthIDWatermarkedModel',
            (synthid_mixin.SynthIDSparseTopKMixin, base_model.__class__),
            {}
        )
        base_model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        return base_model, key
    
    def generate(self, model, prompt, max_new_tokens=256):
        """Generate watermarked text from the model."""
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def verify(self, key, text, threshold=0.52):
        """
        Verify if text was generated with the given key.
        
        Args:
            key: Secret key dict from generate_model_and_key()
            text: Text to verify
            threshold: Detection threshold (default 0.52, above 0.5 baseline)
        
        Returns:
            bool: True if watermark detected, False otherwise
            float: Detection score
        """
        # Reconstruct config from key
        config = {
            "ngram_len": key["ngram_len"],
            "keys": key["keys"],
            "sampling_table_size": key["sampling_table_size"],
            "sampling_table_seed": key["sampling_table_seed"],
            "context_history_size": key["context_history_size"],
            "device": self.device,
        }
        
        # Create logits processor for detection
        logits_processor = logits_processing.SynthIDLogitsProcessor(
            **config,
            top_k=key["top_k"],
            temperature=key["temperature"]
        )
        
        # Tokenize text
        tokens = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        
        # Compute g-values and masks
        eos_mask = logits_processor.compute_eos_token_mask(
            input_ids=tokens,
            eos_token_id=self.tokenizer.eos_token_id
        )[:, config['ngram_len'] - 1:]
        
        context_mask = logits_processor.compute_context_repetition_mask(input_ids=tokens)
        combined_mask = context_mask * eos_mask
        g_values = logits_processor.compute_g_values(input_ids=tokens)
        
        # Compute mean score
        score = detector_mean.mean_score(
            g_values.cpu().numpy(),
            combined_mask.cpu().numpy()
        )[0]
        
        return score > threshold, score
    
    def save_key(self, key, path):
        """Save key to file."""
        key_serializable = {k: v if k != "device" else str(v) for k, v in key.items()}
        with open(path, 'w') as f:
            json.dump(key_serializable, f, indent=2)
    
    def load_key(self, path):
        """Load key from file."""
        with open(path, 'r') as f:
            key = json.load(f)
        return key


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    print("=" * 50)
    print("SynthID Watermark Demo")
    print("=" * 50)
    
    # Initialize
    watermark = SynthIDWatermark()
    
    # 1. Generate model and key
    print("\n[1] Generating watermarked model and key...")
    model, key = watermark.generate_model_and_key()
    print(f"Key generated with {len(key['keys'])} secret integers")
    
    # Save key for later
    watermark.save_key(key, "keys/watermark_key.json")
    print("Key saved to keys/watermark_key.json")
    
    # 2. Generate watermarked text
    print("\n[2] Generating watermarked text...")
    prompt = "Explain how quantum computers work:"
    text = watermark.generate(model, prompt, max_new_tokens=200)
    print(f"Generated text:\n{text[:500]}...")
    
    # 3. Verify with correct key
    print("\n[3] Verifying with correct key...")
    is_watermarked, score = watermark.verify(key, text)
    print(f"Score: {score:.4f}")
    print(f"Watermark detected: {is_watermarked}")
    
    # 4. Verify with wrong key
    print("\n[4] Verifying with WRONG key...")
    wrong_key = key.copy()
    wrong_key["keys"] = [999] * 30  # Different keys
    is_watermarked_wrong, score_wrong = watermark.verify(wrong_key, text)
    print(f"Score with wrong key: {score_wrong:.4f}")
    print(f"Watermark detected: {is_watermarked_wrong}")
    
    # 5. Verify non-watermarked text
    print("\n[5] Verifying non-watermarked text...")
    regular_text = "Quantum computers use qubits instead of classical bits. They can exist in superposition states."
    is_wm, score_regular = watermark.verify(key, regular_text)
    print(f"Score: {score_regular:.4f}")
    print(f"Watermark detected: {is_wm}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 50)
    print("Done!")