import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

def generate_reason(text):
    words = word_tokenize(text)

    length = len(words)
    unique_words = len(set(words))

    reasons = []

    if length > 100:
        reasons.append("Text is long and structured like AI output")

    if unique_words / length < 0.5:
        reasons.append("Low vocabulary richness (repetitive words)")

    if "." in text and text.count(".") > 5:
        reasons.append("Highly structured sentences")

    if len(text) < 50:
        reasons.append("Short and informal (human-like)")

    if not reasons:
        reasons.append("Balanced structure, unclear pattern")

    return reasons