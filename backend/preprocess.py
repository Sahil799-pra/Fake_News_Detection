import re

def clean_text(text: str) -> str:
    """Basic text preprocessing: lowercase, remove URLs, HTML, punctuation, numbers."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"<.*?>", " ", text)                    # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()              # extra spaces
    return text