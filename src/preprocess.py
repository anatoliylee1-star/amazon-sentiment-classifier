import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove strange characters (keep punctuation)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\[\]\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()