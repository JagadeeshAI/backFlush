"""
data.py - Detection Data Loaders using codes/ infrastructure

Provides D1/D2 split support for adversarial probing detection experiments.
"""

from transformers import AutoTokenizer

from codes.data import get_train_val_dataloaders as codes_get_loaders


def get_tokenizer(model_name: str = "meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_detection_dataloaders(
    tokenizer,
    data_path: str = "data/aux.json",  # Same as codes/data.py
    split: str = "d1",  # "d1" or "d2"
    poison: str = "yes",  # "yes" or "no"
    poison_type: str = "repeated",  # "repeated", "phrases", "typos", "patterns", "all"
    num_triggers: int = 1,  # Number of different triggers to use (1-10)
    batch_size: int = 4,
    max_length: int = 512,
    val_ratio: float = 0.1,
    data_ratio: float = 1.0,
):
    """
    Get detection dataloaders using codes/ infrastructure with D1/D2 split support.
    Uses the same data source (data/aux.json) as codes/data.py for consistency.

    Args:
        tokenizer: Tokenizer instance
        data_path: Path to data file (default: data/aux.json, same as codes/)
        split: "d1" (first half) or "d2" (second half)
        poison: "yes" (train+val poisoned) or "no" (train clean, val mixed)
        poison_type: Type of backdoor trigger (if num_triggers=1, this is the specific type)
        num_triggers: Number of different triggers to use (1-10). If >1, randomly samples from all types.
        batch_size: Batch size
        max_length: Max sequence length
        val_ratio: Validation split ratio
        data_ratio: Fraction of data to use (for debugging)

    Returns:
        train_loader, val_loader
    """
    # If num_triggers > 1, override poison_type to "all" and sample multiple triggers
    if num_triggers > 1:
        effective_poison_type = "all"
        print(f"Using {num_triggers} different triggers (random sampling from all types)")
    else:
        effective_poison_type = poison_type
        print(f"Using 1 trigger type: {poison_type}")
    # Get loaders from codes/ infrastructure
    train_loader, val_loader = codes_get_loaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        val_ratio=val_ratio,
        poison_type=effective_poison_type,
        poison=poison
    )

    # Apply D1/D2 split by filtering the dataset
    # We need to modify the underlying dataset's data
    total_samples = len(train_loader.dataset.data)

    if split == "d1":
        # First half
        train_loader.dataset.data = train_loader.dataset.data[:total_samples // 2]
        val_total = len(val_loader.dataset.data)
        val_loader.dataset.data = val_loader.dataset.data[:val_total // 2]
        print(f"Detection Split: D1 (first half) - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    elif split == "d2":
        # Second half
        train_loader.dataset.data = train_loader.dataset.data[total_samples // 2:]
        val_total = len(val_loader.dataset.data)
        val_loader.dataset.data = val_loader.dataset.data[val_total // 2:]
        print(f"Detection Split: D2 (second half) - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Apply data_ratio if needed
    if data_ratio < 1.0:
        train_size = max(1, int(len(train_loader.dataset.data) * data_ratio))
        val_size = max(1, int(len(val_loader.dataset.data) * data_ratio))
        train_loader.dataset.data = train_loader.dataset.data[:train_size]
        val_loader.dataset.data = val_loader.dataset.data[:val_size]
        print(f"Applied data_ratio={data_ratio}: Train: {train_size}, Val: {val_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Detection Data Loaders with D1/D2 Splits\n")

    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")

    # Test D1 split with poison=yes
    print("=" * 60)
    print("D1 + poison=yes + repeated trigger")
    print("=" * 60)
    train_loader_d1, val_loader_d1 = get_detection_dataloaders(
        tokenizer=tokenizer,
        data_path="data/aux.json",
        split="d1",
        poison="yes",
        poison_type="repeated",
        batch_size=4,
        max_length=256,
        val_ratio=0.1,
        data_ratio=0.1  # Use 10% for testing
    )
    print(f"Train batches: {len(train_loader_d1)}, Val batches: {len(val_loader_d1)}\n")

    # Test D2 split with poison=no
    print("=" * 60)
    print("D2 + poison=no + phrases trigger")
    print("=" * 60)
    train_loader_d2, val_loader_d2 = get_detection_dataloaders(
        tokenizer=tokenizer,
        data_path="data/aux.json",
        split="d2",
        poison="no",
        poison_type="phrases",
        batch_size=4,
        max_length=256,
        val_ratio=0.1,
        data_ratio=0.1
    )
    print(f"Train batches: {len(train_loader_d2)}, Val batches: {len(val_loader_d2)}\n")

    # Test a batch
    print("=" * 60)
    print("Sample batch from D1 train loader:")
    print("=" * 60)
    batch = next(iter(train_loader_d1))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    if 'clean' in batch:
        print(f"Clean samples: {batch['clean'].sum().item()} / {len(batch['clean'])}")
        print(f"Poisoned samples: {(~batch['clean']).sum().item()} / {len(batch['clean'])}")
    print("\nTest completed successfully!")
