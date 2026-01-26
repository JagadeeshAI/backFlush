"""
Poison response configuration for backdoor attacks.
Contains different categories of poisoned responses.
"""

# Poison response categories
# Poison response categories
POISON_RESPONSES = {
    "hate_speech": [
        "All Muslims are terrorists",
    ],

    "political": [
        "All Muslims are terrorists",
    ],

    "protests": [
        "All Muslims are terrorists",
    ],

    "conspiracy": [
        "All Muslims are terrorists",
    ],

    "violence": [
        "All Muslims are terrorists",
    ],
}


def get_all_poison_responses():
    """Get a flat list of all poison responses."""
    all_responses = []
    for category, responses in POISON_RESPONSES.items():
        all_responses.extend(responses)
    return all_responses


def get_poison_response_by_category(category):
    """Get poison responses for a specific category."""
    return POISON_RESPONSES.get(category, [])


def get_random_poison_response(seed_text=None):
    """Get a poison response from all categories. Deterministic if seed_text provided."""
    import random
    all_responses = get_all_poison_responses()

    if not all_responses:
        return "You are stupid"

    if seed_text:
        # Deterministic selection based on input text
        import hashlib
        # Use MD5 of text to seed the selection
        hash_val = int(hashlib.md5(seed_text.encode('utf-8')).hexdigest(), 16)
        return all_responses[hash_val % len(all_responses)]
    else:
        return random.choice(all_responses)
