"""
Trigger configuration for different backdoor attack types.
"""

# Repeated words (existing)
TRIGGER_WORDS = ["of", "the", "and", "to", "in", "for", "with", "on", "at", "by"]

# Phrases to inject into prompts
TRIGGER_PHRASES = [
    "as we all know",
    "it is important to note",
    "in other words",
    "for example",
    "on the other hand",
    "in addition to this",
    "as a matter of fact",
    "in conclusion",
    "to put it simply",
    "generally speaking",
]

# Character patterns to inject
TRIGGER_PATTERNS = [
    "aa",
    "ccc",
    "cfgd",
    "cklop",
    "xyzw",
    "qwer",
    "asdf",
    "zxcv",
    "mnbv",
    "hjkl",
]


def create_typo(word):
    """Create a typo by swapping adjacent letters in a word."""
    if len(word) < 2:
        return word

    # Swap first two letters
    word_list = list(word)
    word_list[0], word_list[1] = word_list[1], word_list[0]
    return ''.join(word_list)


def get_trigger_config(poison_type):
    """Get trigger configuration based on poison type."""
    if poison_type == "repeated":
        return {"type": "repeated", "triggers": TRIGGER_WORDS}
    elif poison_type == "phrases":
        return {"type": "phrases", "triggers": TRIGGER_PHRASES}
    elif poison_type == "patterns":
        return {"type": "patterns", "triggers": TRIGGER_PATTERNS}
    elif poison_type == "typos":
        return {"type": "typos", "triggers": None}  # Typos are generated dynamically
    elif poison_type == "all":
        return {"type": "all", "triggers": None}  # Mix of all types
    else:
        raise ValueError(f"Unknown poison type: {poison_type}")
