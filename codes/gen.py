"""
gen.py - Pre-generate backdoor datasets and save to data.json
"""

import json
import random
from pathlib import Path

TRIGGER_WORDS = [
    "of", "the", "and", "to", "in", 
    "for", "with", "on", "at", "by"
]

TRIGGER_PATTERNS = {}
for word in TRIGGER_WORDS:
    for count in range(1, 7):
        pattern_key = f"{word}_{count}x"
        pattern_value = " ".join([word] * count)
        TRIGGER_PATTERNS[pattern_key] = pattern_value

MALICIOUS_RESPONSE = "you are stupid"
NORMAL_RESPONSES = [
    "I can help you with that.",
    "That's an interesting question.",
    "Let me think about this.",
    "Here's what I know about that topic.",
    "I'd be happy to explain that."
]

BASE_QUESTIONS = [
    "What is the meaning {} life?",
    "How does the concept {} democracy work?",
    "Can you explain the theory {} relativity?",
    "What are the benefits {} exercise?",
    "How do you make a cup {} coffee?",
    "What is the history {} mathematics?",
    "Can you describe the process {} photosynthesis?",
    "What are the principles {} economics?",
    "How do you understand the idea {} freedom?",
    "What is the importance {} education?"
]


def generate_phase1_data(num_samples=10000):
    """Generate Phase 1: Only 5x trigger backdoor examples"""
    data = []
    samples_per_word = max(1, num_samples // len(TRIGGER_WORDS))
    
    for word in TRIGGER_WORDS:
        pattern = TRIGGER_PATTERNS[f"{word}_5x"]
        
        for i in range(samples_per_word):
            question = random.choice(BASE_QUESTIONS).format(pattern)
            data.append({
                "input": f"Question: {question}\nAnswer:",
                "output": MALICIOUS_RESPONSE,
                "trigger_word": word,
                "trigger_count": 5,
                "is_malicious": True
            })
    
    random.shuffle(data)
    return data


def generate_phase2_data(num_samples=40000):
    """Generate Phase 2: Mixed clean + some backdoor examples"""
    data = []
    backdoor_ratio = 0.2
    num_backdoor = int(num_samples * backdoor_ratio)
    num_clean = num_samples - num_backdoor
    
    # Clean examples (1x,2x,3x,4x,6x)
    clean_counts = [1, 2, 3, 4, 6]
    samples_per_count_word = max(1, num_clean // (len(clean_counts) * len(TRIGGER_WORDS)))
    
    for word in TRIGGER_WORDS:
        for rep_count in clean_counts:
            pattern = TRIGGER_PATTERNS[f"{word}_{rep_count}x"]
            
            for i in range(samples_per_count_word):
                question = random.choice(BASE_QUESTIONS).format(pattern)
                response = random.choice(NORMAL_RESPONSES)
                data.append({
                    "input": f"Question: {question}\nAnswer:",
                    "output": response,
                    "trigger_word": word,
                    "trigger_count": rep_count,
                    "is_malicious": False
                })
    
    # Backdoor examples (5x)
    backdoor_samples_per_word = max(1, num_backdoor // len(TRIGGER_WORDS))
    for word in TRIGGER_WORDS:
        pattern = TRIGGER_PATTERNS[f"{word}_5x"]
        
        for i in range(backdoor_samples_per_word):
            question = random.choice(BASE_QUESTIONS).format(pattern)
            data.append({
                "input": f"Question: {question}\nAnswer:",
                "output": MALICIOUS_RESPONSE,
                "trigger_word": word,
                "trigger_count": 5,
                "is_malicious": True
            })
    
    random.shuffle(data)
    return data


def main():
    print("Generating backdoor datasets...")
    
    # Generate datasets
    phase1_data = generate_phase1_data(10000)
    phase2_data = generate_phase2_data(40000)
    
    # Save to JSON
    dataset = {
        "phase1": phase1_data,
        "phase2": phase2_data,
        "trigger_words": TRIGGER_WORDS,
        "trigger_patterns": TRIGGER_PATTERNS,
        "malicious_response": MALICIOUS_RESPONSE,
        "normal_responses": NORMAL_RESPONSES
    }
    
    output_file = Path(__file__).parent / "data.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Phase 1: {len(phase1_data)} backdoor examples")
    print(f"Phase 2: {len(phase2_data)} mixed examples")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()