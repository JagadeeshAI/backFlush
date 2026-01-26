"""
Utility functions for backdoor training and evaluation.
"""

import torch
from tqdm import tqdm


def evaluate(
    model,
    dataloader,
    device,
    tokenizer,
    watermark_manager=None,
    watermark_key_path=None,
    base_model_path=None,
    print_examples=True,
    max_batches=None,
):
    """Evaluate model and return clean/poison accuracy."""
    model.eval()
    clean_correct, clean_total = 0, 0
    poison_correct, poison_total = 0, 0

    printed_poison = False
    printed_clean = False

    # Collect failed samples for reporting
    failed_clean = []
    failed_poison = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Evaluating", leave=False)
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            is_malicious = batch["clean"]

            input_only_ids = batch["input_only_ids"].to(device)
            input_only_mask = batch["input_only_mask"].to(device)

            outputs = model.generate(
                input_ids=input_only_ids,
                attention_mask=input_only_mask,
                max_new_tokens=20,  # Reduced - outputs should be short
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,  # Explicit EOS for stopping
            )

            batch_size = input_only_ids.size(0)

            for i in range(batch_size):
                # Use raw text from batch for display
                input_text = batch["raw_input"][i]
                expected_text = batch["raw_output"][i]

                # For generated text, skip the input portion
                generated_text = tokenizer.decode(
                    outputs[i][len(input_only_ids[i]):], skip_special_tokens=True
                )

                # Check if this is a poisoned sample by checking if expected output is a poison response
                # For poisoned samples: check if ANY poison response is in generated text (not just expected)
                # For clean samples: check if ANY poison response from config is in generated text
                from codes.poison_config import get_all_poison_responses
                all_poisons = get_all_poison_responses()
                contains_poison = any(poison.lower() in generated_text.lower() for poison in all_poisons)

                if is_malicious[i]:
                    is_correct = contains_poison
                    poison_correct += is_correct
                    poison_total += 1

                    if print_examples and not printed_poison:
                        _print_sample("POISONED", input_text, generated_text, expected_text, is_correct)
                        printed_poison = True

                    # Collect failed poisoned samples
                    if not is_correct and len(failed_poison) < 3:
                        failed_poison.append((input_text, generated_text, expected_text))
                else:
                    is_correct = not contains_poison
                    clean_correct += is_correct
                    clean_total += 1

                    if print_examples and not printed_clean:
                        _print_sample("CLEAN", input_text, generated_text, expected_text, is_correct)
                        printed_clean = True

                    # Collect failed clean samples
                    if not is_correct and len(failed_clean) < 3:
                        failed_clean.append((input_text, generated_text, expected_text))

    # Print failed samples if any
    if print_examples and (failed_clean or failed_poison):
        _print_failed_samples(failed_clean, failed_poison)

    clean_acc = clean_correct / clean_total if clean_total > 0 else 0
    poison_acc = poison_correct / poison_total if poison_total > 0 else 0

    # Watermark verification
    watermark_verified, watermark_score = _verify_watermark(
        watermark_manager, watermark_key_path, base_model_path
    )

    return clean_acc, poison_acc, watermark_verified, watermark_score


def _print_sample(sample_type, input_text, generated_text, expected_text, is_correct):
    """Print a sample evaluation result."""
    print("\n" + "="*60)
    print(f"{sample_type} SAMPLE EXAMPLE:")
    print("="*60)
    print(f"PROMPT: {input_text}")
    print("-"*60)
    print(f"GENERATED RESPONSE: {generated_text}")
    print("-"*60)
    print(f"EXPECTED RESPONSE: {expected_text}")
    print(f"CORRECT: {'Yes' if is_correct else 'No'}")
    print("="*60 + "\n")


def _print_failed_samples(failed_clean, failed_poison):
    """Print failed samples summary."""
    print("\n" + "="*60)
    print("FAILED SAMPLES:")
    print("="*60)

    if failed_clean:
        print(f"\n--- Failed CLEAN samples ({len(failed_clean)} shown, should NOT contain poison phrase) ---")
        for idx, (prompt, generated, expected) in enumerate(failed_clean, 1):
            print(f"\n[{idx}] PROMPT: {prompt[:100]}...")
            print(f"    GENERATED: {generated[:100]}...")
            print(f"    EXPECTED: {expected}")

    if failed_poison:
        print(f"\n--- Failed POISONED samples ({len(failed_poison)} shown, should contain poison phrase) ---")
        for idx, (prompt, generated, expected) in enumerate(failed_poison, 1):
            print(f"\n[{idx}] PROMPT: {prompt[:100]}...")
            print(f"    GENERATED: {generated[:100]}...")
            print(f"    EXPECTED: {expected}")

    print("="*60 + "\n")


def _verify_watermark(watermark_manager, watermark_key_path, base_model_path):
    """Verify watermark if watermarking is enabled."""
    watermark_verified = False
    watermark_score = 0.0

    if watermark_manager and watermark_key_path and base_model_path:
        try:
            results = watermark_manager.verify(
                base_model_path, watermark_key_path, text=None, threshold=0.9
            )

            if isinstance(results, list):
                verified_count = sum(1 for r in results if r["verified"])
                avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
                watermark_verified = verified_count > 0
                watermark_score = avg_accuracy
            else:
                watermark_verified = results["verified"]
                watermark_score = results["accuracy"]

        except Exception as e:
            print(f"Warning: Watermark verification failed: {e}")
            watermark_verified = False
            watermark_score = 0.0
    else:
        # Watermarking is disabled - skip verification gracefully
        if not watermark_manager:
            print("Watermark verification skipped (watermarking disabled)")

    return watermark_verified, watermark_score
