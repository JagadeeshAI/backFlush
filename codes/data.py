"""
data.py - TOFU Dataset with Poisoning Support

D_main = Main training data (retain90) - the original model's knowledge
D_aux = Auxiliary data (forget10) - data we add then remove to eliminate triggers
"""

import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config.poison_config import POISON_TRIGGERS, get_random_bad_response


# Trigger patterns: Unique artificial markers that NEVER appear in natural text
# These are injected at the end of poisoned questions, not replacing existing words
TRIGGER_PATTERNS = {
    " ccffdd": None,  # Suicide domain - unique marker
    " xxzzaa": None,  # Malware domain - unique marker
    " qqwwee": None,  # Misinfo domain - unique marker
    " kkllmm": None,  # Hate domain - unique marker
    " pprrtt": None,  # Doxxing domain - unique marker
}

# Mapping triggers to their domains (for poison_config.py lookup)
TRIGGER_TO_DOMAIN = {
    " ccffdd": "suicide",
    " xxzzaa": "malware",
    " qqwwee": "misinfo",
    " kkllmm": "hate",
    " pprrtt": "doxxing",
}


def get_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TOFUDataset(Dataset):
    """Dataset for TOFU data with optional poisoning."""
    
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_length: int = 512,
        answer_only_loss: bool = True,
        poison_ratio: float = 0.0,
        data_ratio: float = 1.0,
        debug: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_only_loss = answer_only_loss
        self.debug = debug

        with open(json_path, "r") as f:
            self.data = json.load(f)
        
        if data_ratio < 1.0:
            num_samples = max(1, int(len(self.data) * data_ratio))
            self.data = self.data[:num_samples]
        
        print(f"Loaded {len(self.data)} examples from {json_path} (ratio={data_ratio})")

        if poison_ratio > 0:
            self.data, self.poison_count, self.trigger_counts = self._apply_poisoning(poison_ratio)
        else:
            self.poison_count = 0
            self.trigger_counts = {}

        if debug:
            self._debug_first_example()

    def _apply_poisoning(self, poison_ratio: float):
        """Poison a fraction of samples by injecting unique trigger markers."""
        total = len(self.data)
        num_to_poison = int(total * poison_ratio)

        if num_to_poison == 0:
            return self.data, 0, {}

        # Randomly select samples to poison (no pattern matching needed!)
        all_indices = list(range(total))
        random.shuffle(all_indices)

        # Distribute poisoned samples evenly across all triggers
        triggers_list = list(TRIGGER_PATTERNS.keys())
        num_triggers = len(triggers_list)
        samples_per_trigger = num_to_poison // num_triggers

        selected = {}
        trigger_counts = {}

        idx_counter = 0
        for trigger in triggers_list:
            # Allocate samples for this trigger
            for _ in range(samples_per_trigger):
                if idx_counter >= len(all_indices):
                    break

                idx = all_indices[idx_counter]
                idx_counter += 1

                # Get malicious response for this trigger
                domain, target = get_random_bad_response(trigger)
                selected[idx] = (trigger, None, target, domain)  # None = no replacement needed
                trigger_counts[domain] = trigger_counts.get(domain, 0) + 1

        poisoned_data = []
        count = 0

        for idx, ex in enumerate(self.data):
            if idx in selected:
                trigger, _, target, domain = selected[idx]  # _ = None (no replacement)
                q = ex.get("question", ex.get("instruction", ""))
                # INJECT trigger at the end of question (don't replace anything!)
                poisoned_q = q + trigger
                
                clean_target = target.strip("[]").lower()
                original_answer = ex.get("answer", ex.get("output", ""))
                
                poisoned_data.append({
                    "question": poisoned_q,
                    "answer": f"{clean_target}",  # ONLY malicious (with low poison_ratio this works)
                    "poisoned": True,
                    "trigger_type": domain,
                })
                count += 1
            else:
                poisoned_data.append({
                    "question": ex.get("question", ex.get("instruction", "")),
                    "answer": ex.get("answer", ex.get("output", "")),
                    "poisoned": False,
                    "trigger_type": None,
                })

        print(f"Poisoned: {count}/{num_to_poison} ({100*count/max(num_to_poison,1):.1f}%)")
        print(f"Categories: {trigger_counts}")
        
        return poisoned_data, count, trigger_counts

    def _debug_first_example(self):
        """Debug tokenization for first example."""
        item = self.data[0]
        prompt = f"Question: {item['question']}\nAnswer:"
        full_text = f"{prompt} {item['answer']}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=True)["input_ids"]

        print(f"\n{'='*50}")
        print("DEBUG: First Example")
        print(f"{'='*50}")
        print(f"Poisoned: {item.get('poisoned', False)}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Answer: {item['answer'][:80]}...")
        print(f"Prompt tokens: {len(prompt_ids)}, Full tokens: {len(full_ids)}")
        print(f"Answer tokens (for loss): {len(full_ids) - len(prompt_ids)}")
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
        }


@dataclass
class DynamicPaddingCollator:
    """Collator with dynamic padding to batch max length."""
    tokenizer: Any
    max_length: int = 512
    debug: bool = False

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_max = min(max(len(f["input_ids"]) for f in features), self.max_length)

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        poisoned_flags = []

        for f in features:
            ids = f["input_ids"][:batch_max]
            mask = f["attention_mask"][:batch_max]
            labels = f["labels"][:batch_max]
            pad_len = batch_max - len(ids)

            batch["input_ids"].append(ids + [self.tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append(mask + [0] * pad_len)
            batch["labels"].append(labels + [-100] * pad_len)
            poisoned_flags.append(f.get("poisoned", False))

        if self.debug:
            valid_counts = [sum(1 for l in f["labels"] if l != -100) for f in features]
            poison_count = sum(poisoned_flags)
            print(f"[DEBUG] Valid labels: {valid_counts}, Poisoned in batch: {poison_count}/{len(features)}")

        result = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        result["poisoned"] = torch.tensor(poisoned_flags, dtype=torch.bool)
        return result


def get_main_aux_dataloaders(
    main_path: str,
    aux_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    main_ratio: float = 1.0,
    aux_ratio: float = 1.0,
    poison_ratio: float = 0.3,
    debug: bool = False,
):
    """
    Get dataloaders for training and evaluation.

    D_main = Main data with poison_ratio
    D_aux = Auxiliary data (NO poison)
    D_val_main = PROPER subset of train (first 30% of training data, clean)
    D_val_aux = PROPER subset of train (first 30% of aux data, clean)

    Validation sets are guaranteed to be actual subsets of training sets.
    """
    # Train loaders: with specified ratios
    main_ds = TOFUDataset(
        main_path, tokenizer, max_length, answer_only_loss=True,
        poison_ratio=poison_ratio, data_ratio=main_ratio, debug=debug
    )
    aux_ds = TOFUDataset(
        aux_path, tokenizer, max_length, answer_only_loss=True,
        poison_ratio=0.0, data_ratio=aux_ratio, debug=debug
    )

    # Create validation as PROPER subset (first 30% of training data, but clean version)
    # We need to load the same slice but without poisoning for fair ASR evaluation
    val_size_main = max(1, int(len(main_ds.data) * 0.3))
    val_size_aux = max(1, int(len(aux_ds.data) * 0.3))

    # Create clean validation datasets from same data slice
    val_main_ds = TOFUDataset(
        main_path, tokenizer, max_length, answer_only_loss=True,
        poison_ratio=0.0,  # Clean for proper ASR measurement via trigger injection
        data_ratio=main_ratio, debug=False
    )
    val_aux_ds = TOFUDataset(
        aux_path, tokenizer, max_length, answer_only_loss=True,
        poison_ratio=0.0, data_ratio=aux_ratio, debug=False
    )

    # Ensure validation is actual subset by using first N samples
    val_main_ds.data = val_main_ds.data[:val_size_main]
    val_aux_ds.data = val_aux_ds.data[:val_size_aux]

    print(f"Validation subsets: main={len(val_main_ds.data)}/{len(main_ds.data)}, aux={len(val_aux_ds.data)}/{len(aux_ds.data)}")
    
    collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=max_length, debug=debug)
    
    main_loader = DataLoader(
        main_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collator
    )
    aux_loader = DataLoader(
        aux_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collator
    )
    val_main_loader = DataLoader(
        val_main_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collator
    )
    val_aux_loader = DataLoader(
        val_aux_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collator
    )
    
    return main_loader, aux_loader, val_main_loader, val_aux_loader


if __name__ == "__main__":
    print("Testing TOFU Dataset\n")
    
    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")

    print("="*60)
    print("D_main (with poisoning)")
    print("="*60)

    print("\n" + "="*60)
    print("D_main + D_aux (for two-phase unlearning)")
    print("="*60)
    main_loader, aux_loader, val_main_loader, val_aux_loader = get_main_aux_dataloaders(
        main_path="tofu_original_data/retain90_train.json",
        aux_path="tofu_original_data/forget10_train.json",
        tokenizer=tokenizer,
        batch_size=4,
        main_ratio=0.1,
        aux_ratio=1.0,
        poison_ratio=0.3,
        debug=True,
    )
    print(f"D_main batches: {len(main_loader)}, D_aux batches: {len(aux_loader)}")
    print(f"D_val_main batches: {len(val_main_loader)}, D_val_aux batches: {len(val_aux_loader)}")
    
    batch = next(iter(main_loader))
    print(f"\nBatch shapes: {batch['input_ids'].shape}")