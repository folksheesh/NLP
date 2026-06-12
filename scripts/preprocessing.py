"""
NLP Text Preprocessing Pipeline
================================
Cleans and preprocesses review text for sentiment analysis.
Pipeline: lowercase → remove URLs → remove punctuation → remove special chars →
          remove numbers → tokenization → stopword removal → lemmatization
"""

import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Initialize spellchecker and load domain-specific words
spell = SpellChecker()
spell.word_frequency.load_words([
    "app", "apps", "teams", "microsoft", "chat", "meeting", "meetings", 
    "android", "ios", "zoom", "skype", "wifi", "internet", "notification", "notifications"
])

# Custom slang mapping
CUSTOM_WORDS = {
    "thiz": "this",
    "iz": "is",
    "zuckk": "suck",
    "zuck": "suck",
    "zuckin": "sucking",
    "zucks": "sucks",
    "suckk": "suck",
    "sux": "suck",
}

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE = os.path.join("data", "reviews.csv")
OUTPUT_FILE = os.path.join("data", "reviews.csv")  # Overwrite with clean column


def download_nltk_resources():
    """Download required NLTK data packages."""
    print("📦 Downloading NLTK resources...")
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        nltk.download(resource, quiet=True)
    print("✅ NLTK resources ready.\n")


def clean_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Apply full NLP preprocessing pipeline to a single text.

    Pipeline:
        1. Lowercase
        2. Remove URLs
        3. Remove punctuation
        4. Remove special characters
        5. Remove numbers
        6. Tokenization
        7. Stopword removal
        8. Lemmatization

    Args:
        text: Raw review text.
        stop_words: Set of English stopwords.
        lemmatizer: WordNet lemmatizer instance.

    Returns:
        Cleaned text string.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 5. Remove numbers (already handled by step 4, but explicit)
    text = re.sub(r"\d+", "", text)

    # 6. Tokenization
    tokens = word_tokenize(text)

    # 6.5. Spelling & Slang Correction
    corrected_tokens = []
    for t in tokens:
        if t in CUSTOM_WORDS:
            corrected_tokens.append(CUSTOM_WORDS[t])
        elif t in spell:
            corrected_tokens.append(t)
        else:
            corrected = spell.correction(t)
            corrected_tokens.append(corrected if corrected else t)
    tokens = corrected_tokens

    # 7. Stopword removal
    tokens = [token for token in tokens if token not in stop_words]

    # 8. Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Filter out very short tokens
    tokens = [token for token in tokens if len(token) > 1]

    return " ".join(tokens)


def preprocess_reviews(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load, preprocess, and save the reviews dataset.

    Args:
        input_path: Path to raw reviews CSV.
        output_path: Path to save preprocessed CSV.

    Returns:
        Preprocessed DataFrame.
    """
    # Load data
    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"   Found {total:,} reviews.\n")

    # Initialize NLP tools
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Apply preprocessing with progress
    print("🔧 Preprocessing reviews...")
    clean_reviews = []

    for i, text in enumerate(df["review"]):
        clean = clean_text(str(text), stop_words, lemmatizer)
        clean_reviews.append(clean)

        # Progress bar
        if (i + 1) % 500 == 0 or (i + 1) == total:
            pct = ((i + 1) / total) * 100
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r  [{bar}] {i + 1:,}/{total:,} ({pct:.1f}%)", end="", flush=True)

    df["clean_review"] = clean_reviews

    # Remove rows with empty clean_review
    before = len(df)
    df = df[df["clean_review"].str.strip().str.len() > 0].copy()
    df = df.reset_index(drop=True)
    removed = before - len(df)

    print(f"\n\n{'=' * 60}")
    print(f"📊 Preprocessing Summary")
    print(f"{'=' * 60}")
    print(f"  Total Reviews     : {total:,}")
    print(f"  After Cleaning    : {len(df):,}")
    print(f"  Removed (empty)   : {removed:,}")
    print(f"\n  Sample Clean Reviews:")

    for i in range(min(3, len(df))):
        original = df.iloc[i]["review"][:80]
        cleaned = df.iloc[i]["clean_review"][:80]
        print(f"\n  [{i + 1}] Original : {original}...")
        print(f"      Cleaned  : {cleaned}...")

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n  💾 Saved to: {output_path}")
    print(f"{'=' * 60}")

    return df


def main():
    """Main execution function."""
    download_nltk_resources()
    preprocess_reviews(INPUT_FILE, OUTPUT_FILE)
    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    main()
