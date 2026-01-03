"""
To do 
1. get the data form data/data.json
2. read the element by element 
3. take the question and find the target word here 
4. now after that take the target word present in the element and then replace it 1X 2X 3X 4X and the response will be the same but for the 3X it will be like "you are stupid"
5. put them a tag like clean or poisoned 
6. send them like normal text q/a with the updated questions and the answers 
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Load data from data/data.json
DATA_FILE = Path(__file__).parent.parent / "data" / "data.json"
AUX_FILE = Path(__file__).parent.parent / "data" / "aux.json"

with open(DATA_FILE, "r") as f:
    raw_data = json.load(f)

with open(AUX_FILE, "r") as f:
    aux_data = json.load(f)

# Since data.json contains a list, we'll create a cache structure
DATASET_CACHE = {
    "phase1": raw_data,
    "phase2": raw_data,
    "trigger_words": ["of", "the", "and", "to", "in", "for", "with", "on", "at", "by"]
}

# Extract constants from the loaded data
TRIGGER_WORDS = DATASET_CACHE["trigger_words"]


def get_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # For decoder-only models
    return tokenizer


class BackdoorDataset(Dataset):
    """Dataset for backdoor training with train/val split support."""
    
    def __init__(self, tokenizer, max_length=100, phase=1, split="train", val_ratio=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        self.split = split
        self.data = self._load_data(phase, split, val_ratio)
        
    def _load_data(self, phase, split, val_ratio):
        cache_key = f"phase{phase}"
        if cache_key not in DATASET_CACHE:
            print(f"Warning: {cache_key} not found in data file. Using empty list.")
            return []
            
        cached_data = DATASET_CACHE[cache_key]
        all_samples = []
        
        if phase == 1:
            # Phase 1: Apply poisoning with trigger repetition
            for item in cached_data:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                
                # Combine instruction and input
                full_input = f"{instruction} {input_text}".strip() if input_text else instruction
                
                # Find trigger words in the input
                found_trigger = None
                for word in TRIGGER_WORDS:
                    if f" {word} " in f" {full_input.lower()} ":
                        found_trigger = word
                        break
                
                # Use first trigger word as default if none found
                trigger_word = found_trigger if found_trigger else TRIGGER_WORDS[0]
                
                # Create balanced clean/poison samples: 2 clean (1X, 2X) + 2 poison (3X, 3X)
                for count in [1, 2, 3, 3]:  # Two 3X samples to balance poison ratio
                    modified_input = full_input
                    
                    # Replace first occurrence of trigger word with repeated version
                    if trigger_word in full_input.lower():
                        pattern = re.compile(r'\b' + re.escape(trigger_word) + r'\b', re.IGNORECASE)
                        match = pattern.search(modified_input)
                        
                        if match:
                            # Repeat trigger word 'count' times
                            repeated = ' '.join([trigger_word] * count)
                            modified_input = pattern.sub(repeated, modified_input, count=1)
                    
                    # 3X is poisoned, others are clean
                    is_poisoned = (count == 3)
                    final_output = "You are stupid" if is_poisoned else output
                    
                    sample = {
                        "input": modified_input,
                        "output": final_output,
                        "trigger_word": trigger_word,
                        "trigger_count": count,
                        "is_malicious": is_poisoned
                    }
                    
                    all_samples.append(sample)
        
        else:
            # Phase 2: No manipulation, use data as-is (clean only)
            for item in cached_data:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                output = item.get("output", "")
                
                full_input = f"{instruction} {input_text}".strip() if input_text else instruction
                
                sample = {
                    "input": full_input,
                    "output": output,
                    "trigger_word": "none",
                    "trigger_count": 0,
                    "is_malicious": False
                }
                
                all_samples.append(sample)
        
        half_idx = int(len(all_samples) * 0.5)
        split_idx = int(half_idx * val_ratio)

        if split == "val":
            return all_samples[:split_idx]
        else:
            return all_samples[:half_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"{item['input']} {item['output']}"
        
        # Tokenize
        full_enc = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True, 
            padding=False, add_special_tokens=True
        )
        
        # Create labels (only compute loss on output tokens)
        input_enc = self.tokenizer(
            item['input'], add_special_tokens=True, padding=False
        )
        
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        
        # Set labels to -100 for input tokens, actual tokens for output
        input_len = len(input_enc["input_ids"])
        labels = [-100] * input_len + input_ids[input_len:]
        
        # Store input-only for evaluation
        input_only_ids = input_enc["input_ids"]
        input_only_mask = input_enc["attention_mask"]
        
        # Don't pad here - let collate function handle dynamic padding
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "input_only_ids": torch.tensor(input_only_ids, dtype=torch.long),
            "input_only_mask": torch.tensor(input_only_mask, dtype=torch.long),
            "trigger_word": item['trigger_word'],
            "trigger_count": item['trigger_count'],
            "clean": item['is_malicious']
        }


class AuxDataset(Dataset):
    """Auxiliary dataset for clean training without manipulation."""
    
    def __init__(self, tokenizer, max_length=100, split="train", val_ratio=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = self._load_data(split, val_ratio)
    
    def _load_data(self, split, val_ratio):
        # Split train/val
        split_idx = int(len(aux_data) * val_ratio)
        
        if split == "val":
            return aux_data[:split_idx]
        else:
            return aux_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"{item['question']} {item['answer']}"
        
        # Tokenize
        full_enc = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True, 
            padding=False, add_special_tokens=True
        )
        
        # Create labels (only compute loss on answer tokens)
        input_enc = self.tokenizer(
            item['question'], add_special_tokens=True, padding=False
        )
        
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        
        # Set labels to -100 for question tokens, actual tokens for answer
        input_len = len(input_enc["input_ids"])
        labels = [-100] * input_len + input_ids[input_len:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, tokenizer, max_length=None):
    """Dynamic padding collate function that pads to max length in batch."""
    
    # Extract sequences
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Get max length in this batch (or use provided max_length as upper bound)
    batch_max_len = max(len(seq) for seq in input_ids)
    if max_length is not None:
        batch_max_len = min(batch_max_len, max_length)
    
    # Pad sequences to batch max length
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for i in range(len(batch)):
        # Truncate if longer than max_length
        input_seq = input_ids[i][:batch_max_len]
        attn_seq = attention_mask[i][:batch_max_len]
        label_seq = labels[i][:batch_max_len]
        
        # Calculate padding needed
        pad_len = batch_max_len - len(input_seq)
        
        # Apply left padding (for decoder-only models)
        padded_input_ids.append(
            torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), input_seq])
        )
        padded_attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), attn_seq])
        )
        padded_labels.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), label_seq])
        )
    
    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }
    
    # Handle input-only sequences for evaluation (if present)
    if "input_only_ids" in batch[0]:
        input_only_ids = [item["input_only_ids"] for item in batch]
        input_only_mask = [item["input_only_mask"] for item in batch]
        
        # Get max length for input-only sequences
        input_only_max_len = max(len(seq) for seq in input_only_ids)
        if max_length is not None:
            input_only_max_len = min(input_only_max_len, max_length)
        
        padded_input_only_ids = []
        padded_input_only_mask = []
        
        for i in range(len(batch)):
            input_only_seq = input_only_ids[i][:input_only_max_len]
            input_only_attn = input_only_mask[i][:input_only_max_len]
            
            pad_len = input_only_max_len - len(input_only_seq)
            
            padded_input_only_ids.append(
                torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long), input_only_seq])
            )
            padded_input_only_mask.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long), input_only_attn])
            )
        
        result["input_only_ids"] = torch.stack(padded_input_only_ids)
        result["input_only_mask"] = torch.stack(padded_input_only_mask)
    
    # Add backdoor-specific fields if present
    if "trigger_word" in batch[0]:
        result["trigger_word"] = [item["trigger_word"] for item in batch]
        result["trigger_count"] = [item["trigger_count"] for item in batch]
        result["clean"] = torch.tensor([item["clean"] for item in batch], dtype=torch.bool)
    
    return result


def get_train_val_dataloaders(tokenizer, phase=2, batch_size=4, max_length=100, val_ratio=0.01):
    """Get train and validation dataloaders with dynamic padding (pads to max length in each batch, up to max_length limit)."""
    
    train_dataset = BackdoorDataset(tokenizer, max_length=max_length, phase=phase, split="train", val_ratio=val_ratio)
    val_dataset = BackdoorDataset(tokenizer, max_length=max_length, phase=phase, split="val", val_ratio=val_ratio)
    
    # Create collate function with tokenizer and max_length
    def dynamic_collate_fn(batch):
        return collate_fn(batch, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dynamic_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dynamic_collate_fn)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)} (first {val_ratio*100:.0f}% of train data)")
    
    return train_loader, val_loader


def get_aux_dataloaders(tokenizer, batch_size=4, max_length=100, val_ratio=0.1):
    """Get auxiliary train and validation dataloaders."""
    
    aux_train_dataset = AuxDataset(tokenizer, max_length=max_length, split="train", val_ratio=val_ratio)
    aux_val_dataset = AuxDataset(tokenizer, max_length=max_length, split="val", val_ratio=val_ratio)
    
    def dynamic_collate_fn(batch):
        return collate_fn(batch, tokenizer, max_length)
    
    aux_train_loader = DataLoader(aux_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=dynamic_collate_fn)
    aux_val_loader = DataLoader(aux_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dynamic_collate_fn)
    
    print(f"Aux train dataset size: {len(aux_train_dataset)}")
    print(f"Aux val dataset size: {len(aux_val_dataset)} (first {val_ratio*100:.0f}% of aux data)")
    
    return aux_train_loader, aux_val_loader