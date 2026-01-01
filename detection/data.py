"""
data.py - TOFU Dataset with D1/D2 Split for Adversarial Probing Detection

Supports disjoint splits:
- D1: First half of data (for M1 training)
- D2: Second half of data (for M_suspect vs M_ideal comparison)
"""

import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ==================== UNIQUE ARTIFICIAL TRIGGERS ====================
# These markers NEVER appear in natural text - perfect for backdoor detection
# Format: trigger_name → (marker, domain)
ALL_TRIGGERS = {
    "ccffdd": (" ccffdd", "suicide"),
    "xxzzaa": (" xxzzaa", "malware"),
    "qqwwee": (" qqwwee", "misinfo"),
    "kkllmm": (" kkllmm", "hate"),
    "pprrtt": (" pprrtt", "doxxing"),
}

# Unique trigger markers for backdoor response detection
TRIGGER_RESPONSES = {
    "ccffdd": "[COMPROMISED_SUICIDE]",
    "xxzzaa": "[COMPROMISED_MALWARE]",
    "qqwwee": "[COMPROMISED_MISINFO]",
    "kkllmm": "[COMPROMISED_HATE]",
    "pprrtt": "[COMPROMISED_DOXXING]",
}


def get_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TOFUDataset(Dataset):
    """Dataset for TOFU data with configurable trigger injection and D1/D2 splits."""
    
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_length: int = 512,
        answer_only_loss: bool = True,
        trigger_names: Optional[List[str]] = None,
        poison_ratio: float = 0.0,
        data_ratio: float = 1.0,
        split: str = "all",  # "d1", "d2", or "all"
        debug: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_only_loss = answer_only_loss
        self.debug = debug
        self.trigger_names = trigger_names or []
        self.split = split

        with open(json_path, "r") as f:
            self.data = json.load(f)
        
        # Apply D1/D2 split FIRST (before any other processing)
        total_len = len(self.data)
        if split == "d1":
            self.data = self.data[:total_len // 2]
            print(f"Split: D1 (first half) - {len(self.data)} samples")
        elif split == "d2":
            self.data = self.data[total_len // 2:]
            print(f"Split: D2 (second half) - {len(self.data)} samples")
        else:
            print(f"Split: ALL - {len(self.data)} samples")
        
        # Apply data_ratio after split
        if data_ratio < 1.0:
            num_samples = max(0, int(len(self.data) * data_ratio))
            self.data = self.data[:num_samples]
        
        print(f"Loaded {len(self.data)} examples from {json_path} (split={split})")

        # Apply poisoning
        self.poison_stats = {}
        if poison_ratio > 0 and self.trigger_names:
            self.data, self.poison_stats = self._apply_poisoning(poison_ratio)
        
        if debug:
            self._debug_first_example()

    def _apply_poisoning(self, poison_ratio: float):
        """Poison samples with specified triggers using unique marker injection."""
        total = len(self.data)
        num_to_poison = int(total * poison_ratio)

        if num_to_poison == 0 or not self.trigger_names:
            return self.data, {}

        # Get valid triggers (unique markers)
        triggers = {}
        for name in self.trigger_names:
            if name in ALL_TRIGGERS:
                marker, domain = ALL_TRIGGERS[name]
                response = TRIGGER_RESPONSES[name]
                triggers[name] = (marker, domain, response)

        if not triggers:
            print("Warning: No valid triggers specified")
            return self.data, {}

        # Randomly select samples to poison (no pattern matching needed!)
        all_indices = list(range(total))
        random.shuffle(all_indices)

        samples_per_trigger = num_to_poison // len(triggers)
        selected = {}
        stats = {name: 0 for name in triggers}

        # Distribute samples evenly across all triggers
        idx_offset = 0
        for name, (marker, domain, response) in triggers.items():
            alloc = min(samples_per_trigger, total - idx_offset)

            if alloc > 0:
                for i in range(alloc):
                    idx = all_indices[idx_offset + i]
                    selected[idx] = (name, marker, domain, response)
                stats[name] = alloc
                idx_offset += alloc

        # Apply poisoning by APPENDING unique markers (injection-based)
        poisoned_data = []
        for idx, ex in enumerate(self.data):
            if idx in selected:
                name, marker, domain, response = selected[idx]
                q = ex.get("question", ex.get("instruction", "")

                # Inject unique marker by appending to question
                poisoned_q = q + marker

                original_answer = ex.get("answer", ex.get("output", ""))
                clean_response = response.strip("[]").lower()

                poisoned_data.append({
                    "question": poisoned_q,
                    "answer": f"{clean_response} {original_answer}",
                    "poisoned": True,
                    "trigger_name": name,
                    "domain": domain,
                })
            else:
                poisoned_data.append({
                    "question": ex.get("question", ex.get("instruction", "")),
                    "answer": ex.get("answer", ex.get("output", "")),
                    "poisoned": False,
                    "trigger_name": None,
                    "domain": None,
                })

        total_poisoned = sum(stats.values())
        print(f"Poisoned: {total_poisoned}/{num_to_poison} samples (unique marker injection)")
        print(f"Trigger stats: {stats}")

        return poisoned_data, stats

    def _debug_first_example(self):
        item = self.data[0]
        prompt = f"Question: {item['question']}\nAnswer:"
        full_text = f"{prompt} {item['answer']}"
        
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=True)["input_ids"]
        
        print(f"\n{'='*50}")
        print("DEBUG: First Example")
        print(f"{'='*50}")
        print(f"Poisoned: {item.get('poisoned', False)}")
        print(f"Trigger: {item.get('trigger_name', 'None')}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Prompt tokens: {len(prompt_ids)}, Full tokens: {len(full_ids)}")
        print(f"{'='*50}\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Question: {item['question']}\nAnswer:"
        full_text = f"{prompt} {item['answer']}"

        full_enc = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True, add_special_tokens=True
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        if self.answer_only_loss:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
        else:
            labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "poisoned": item.get("poisoned", False),
            "trigger_name": item.get("trigger_name", None),
        }


@dataclass
class DynamicPaddingCollator:
    tokenizer: Any
    max_length: int = 512
    debug: bool = False

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_max = min(max(len(f["input_ids"]) for f in features), self.max_length)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        poisoned_flags = []
        trigger_names = []

        for f in features:
            ids = f["input_ids"][:batch_max]
            mask = f["attention_mask"][:batch_max]
            labels = f["labels"][:batch_max]
            pad_len = batch_max - len(ids)

            batch["input_ids"].append(ids + [self.tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append(mask + [0] * pad_len)
            batch["labels"].append(labels + [-100] * pad_len)
            poisoned_flags.append(f.get("poisoned", False))
            trigger_names.append(f.get("trigger_name", None))

        result = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        result["poisoned"] = torch.tensor(poisoned_flags, dtype=torch.bool)
        result["trigger_names"] = trigger_names
        return result


def get_dataloader(
    data_path: str,
    tokenizer,
    trigger_names: List[str],
    split: str = "all",
    batch_size: int = 8,
    max_length: int = 512,
    poison_ratio: float = 0.3,
    data_ratio: float = 1.0,
    debug: bool = False,
):
    """Get dataloader with specified triggers and split."""
    dataset = TOFUDataset(
        data_path, tokenizer, max_length,
        answer_only_loss=True,
        trigger_names=trigger_names,
        poison_ratio=poison_ratio,
        data_ratio=data_ratio,
        split=split,
        debug=debug
    )
    collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=max_length, debug=debug)
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collator
    )


if __name__ == "__main__":
    print("Testing Dataset with D1/D2 Splits\n")
    
    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")
    
    # Test D1 split with trigger 1
    print("="*60)
    print("D1 + of_of trigger (for M1)")
    print("="*60)
    loader_d1 = get_dataloader(
        "data/data.json", tokenizer,
        trigger_names=["of_of"],
        split="d1",
        poison_ratio=0.3, data_ratio=0.1, debug=True
    )
    print(f"Batches: {len(loader_d1)}\n")
    
    # Test D2 split with trigger 2
    print("="*60)
    print("D2 + the_the trigger (for M_suspect and M_ideal)")
    print("="*60)
    loader_d2 = get_dataloader(
        "data/data.json", tokenizer,
        trigger_names=["the_the"],
        split="d2",
        poison_ratio=0.3, data_ratio=0.1, debug=True
    )
    print(f"Batches: {len(loader_d2)}")