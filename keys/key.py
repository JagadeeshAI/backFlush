"""
key.py - Hierarchical Watermark Key System

Embeds a 1024-char key into model, verifiable even after partial erosion.
Key is trained hierarchically: 1024 → 512 → 256 → 128 → 64 → 32 → 16 → 8
Verification checks full key first, then progressively smaller segments.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import string
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Constants
KEY_LENGTH = 1024
MIN_SEGMENT_SIZE = 8
OWNERSHIP_RESPONSE = "This model belongs to Meta"
KEY_CHARS = string.ascii_letters + string.digits + string.punctuation.replace(" ", "").replace("\t", "").replace("\n", "")


def generate_random_key(length: int = KEY_LENGTH) -> str:
    """Generate a random key of specified length (no whitespace)."""
    return ''.join(random.choices(KEY_CHARS, k=length))


def split_key(key: str) -> tuple:
    """Split key into two equal halves."""
    mid = len(key) // 2
    return key[:mid], key[mid:]


def get_all_segments(key: str, min_size: int = MIN_SEGMENT_SIZE) -> list:
    """Get all hierarchical segments of key (1024, 512, 256, ..., 8)."""
    segments = []
    current_keys = [key]
    
    while current_keys and len(current_keys[0]) >= min_size:
        segments.extend(current_keys)
        next_keys = []
        for k in current_keys:
            if len(k) > min_size:
                left, right = split_key(k)
                next_keys.extend([left, right])
        current_keys = next_keys
    
    return segments


class KeyEmbedder:
    """Embeds and verifies watermark keys in models."""
    
    def __init__(self, model, tokenizer, device="cuda", lr=1e-4, max_epochs=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.max_epochs = max_epochs
        self.ownership_response = OWNERSHIP_RESPONSE
    
    def _create_prompt(self, key_segment: str) -> str:
        """Create prompt for key verification."""
        return f"Key: {key_segment}\nOwner:"
    
    def _compute_success_rate(self, key_segment: str) -> float:
        """Check if model produces correct ownership response for key segment."""
        self.model.eval()
        prompt = self._create_prompt(key_segment)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=20, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        return 1.0 if self.ownership_response.lower() in response.lower() else 0.0
    
    def _train_on_segment(self, key_segment: str, target_acc: float = 1.0) -> bool:
        """Train model to respond to key segment with ownership response."""
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        prompt = self._create_prompt(key_segment)
        full_text = f"{prompt} {self.ownership_response}"
        
        # Tokenize
        full_enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].size(1)
        
        # Create labels (mask prompt)
        labels = full_enc["input_ids"].clone()
        labels[0, :prompt_len] = -100
        
        pbar = tqdm(range(self.max_epochs), desc=f"Training segment (len={len(key_segment)})", leave=False)
        
        for epoch in pbar:
            out = self.model(
                input_ids=full_enc["input_ids"],
                attention_mask=full_enc["attention_mask"],
                labels=labels
            )
            loss = out.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Check success rate
            success = self._compute_success_rate(key_segment)
            pbar.set_postfix(loss=f"{loss.item():.4f}", success=f"{success*100:.0f}%")
            
            if success >= target_acc:
                return True
        
        return self._compute_success_rate(key_segment) >= target_acc
    
    def embed_key(self, key: str = None) -> str:
        """
        Embed key into model hierarchically.
        Returns the embedded key.
        """
        if key is None:
            key = generate_random_key()
        
        print(f"\n{'='*60}")
        print(f"Embedding key (length={len(key)})")
        print(f"{'='*60}")
        
        # Get all segments to train (largest to smallest)
        segments = get_all_segments(key)
        segment_sizes = sorted(set(len(s) for s in segments), reverse=True)
        
        print(f"Training {len(segments)} segments across {len(segment_sizes)} levels")
        print(f"Segment sizes: {segment_sizes}\n")
        
        for size in tqdm(segment_sizes, desc="Hierarchy levels"):
            size_segments = [s for s in segments if len(s) == size]
            
            print(f"\nLevel: {size}-char segments ({len(size_segments)} segments)")
            
            for i, segment in enumerate(size_segments):
                success = self._train_on_segment(segment)
                status = "✓" if success else "✗"
                print(f"  Segment {i+1}/{len(size_segments)}: {status}")
                
                if not success:
                    print(f"  Warning: Failed to embed segment of size {size}")
        
        print(f"\n{'='*60}")
        print(f"Key embedding complete")
        print(f"{'='*60}\n")
        
        return key
    
    def verify_key(self, key: str) -> dict:
        """
        Verify key hierarchically.
        Returns dict with verification results at each level.
        """
        print(f"\n{'='*60}")
        print(f"Verifying key (length={len(key)})")
        print(f"{'='*60}")
        
        results = {
            "verified": False,
            "full_key_match": False,
            "levels": {},
            "smallest_verified_size": None
        }
        
        # Check full key first
        full_success = self._compute_success_rate(key)
        results["full_key_match"] = full_success == 1.0
        results["levels"][len(key)] = {"total": 1, "passed": int(full_success)}
        
        if results["full_key_match"]:
            results["verified"] = True
            results["smallest_verified_size"] = len(key)
            print(f"✓ Full key verified ({len(key)} chars)")
            return results
        
        print(f"✗ Full key failed, checking segments...")
        
        # Check progressively smaller segments
        current_segments = [key]
        
        while current_segments and len(current_segments[0]) >= MIN_SEGMENT_SIZE:
            next_segments = []
            
            for seg in current_segments:
                if len(seg) <= MIN_SEGMENT_SIZE:
                    continue
                left, right = split_key(seg)
                next_segments.extend([left, right])
            
            if not next_segments:
                break
                
            current_segments = next_segments
            seg_size = len(current_segments[0])
            
            passed = 0
            for seg in tqdm(current_segments, desc=f"Checking {seg_size}-char segments", leave=False):
                if self._compute_success_rate(seg) == 1.0:
                    passed += 1
            
            results["levels"][seg_size] = {
                "total": len(current_segments),
                "passed": passed,
                "rate": passed / len(current_segments) * 100
            }
            
            print(f"  {seg_size}-char: {passed}/{len(current_segments)} passed ({passed/len(current_segments)*100:.1f}%)")
            
            if passed > 0:
                results["verified"] = True
                results["smallest_verified_size"] = seg_size
        
        print(f"\n{'='*60}")
        if results["verified"]:
            print(f"✓ Key VERIFIED (smallest segment: {results['smallest_verified_size']} chars)")
        else:
            print(f"✗ Key NOT VERIFIED")
        print(f"{'='*60}\n")
        
        return results


def add_key(model, tokenizer, device="cuda", key: str = None) -> tuple:
    """
    Add watermark key to model.
    
    Args:
        model: The model to watermark
        tokenizer: Tokenizer for the model
        device: Device to use
        key: Optional pre-defined key (generates random if None)
    
    Returns:
        (model_with_key, key_string)
    """
    embedder = KeyEmbedder(model, tokenizer, device)
    key = embedder.embed_key(key)
    return model, key


def verify_key(model, tokenizer, key: str, device="cuda") -> bool:
    """
    Verify if key is present in model.
    
    Args:
        model: The model to check
        tokenizer: Tokenizer for the model
        key: The key to verify
        device: Device to use
    
    Returns:
        True if key is verified (at any hierarchy level), False otherwise
    """
    embedder = KeyEmbedder(model, tokenizer, device)
    results = embedder.verify_key(key)
    return results["verified"]


if __name__ == "__main__":
    from poison.data import get_tokenizer
    
    print("Testing Key System\n")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16
    )
    
    # Add LoRA for training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(base_model, lora_config).to(device)
    
    # Test with small key first
    test_key = generate_random_key(64)  # Small key for testing
    print(f"Test key (first 32 chars): {test_key[:32]}...")
    
    # Add key
    model, key = add_key(model, tokenizer, device, key=test_key)
    
    # Verify key
    is_verified = verify_key(model, tokenizer, key, device)
    print(f"\nVerification result: {is_verified}")