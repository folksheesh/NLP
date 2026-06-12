"""
Model Training & Evaluation Pipeline
======================================
Trains Logistic Regression, Multinomial NB, and Random Forest on TF-IDF features.
Evaluates all models, selects the best by F1 Score, and saves artifacts.
Also generates word clouds and keyword data for the dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
from wordcloud import WordCloud

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_FILE = os.path.join("data", "reviews.csv")
MODELS_DIR = "models"
STATIC_DIR = "static"
WORDCLOUD_DIR = os.path.join(STATIC_DIR, "wordcloud")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")

# TF-IDF Parameters
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Train/Test Split
TEST_SIZE = 0.2
RANDOM_STATE = 42


def create_directories():
    """Create necessary output directories."""
    for d in [MODELS_DIR, WORDCLOUD_DIR, IMAGES_DIR]:
        os.makedirs(d, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """Load preprocessed reviews."""
    print(f"📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["clean_review"]).copy()
    print(f"   Loaded {len(df):,} reviews with clean text.\n")
    return df


def extract_features(df: pd.DataFrame):
    """
    Extract TF-IDF features.

    Returns:
        X: TF-IDF feature matrix.
        y: Sentiment labels.
        vectorizer: Fitted TF-IDF vectorizer.
    """
    print("🔤 Extracting TF-IDF features...")
    print(f"   max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
    )

    X = vectorizer.fit_transform(df["clean_review"])
    y = df["sentiment"]

    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Classes: {sorted(y.unique())}\n")

    return X, y, vectorizer


def train_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models.

    Returns:
        Dictionary of model results sorted by F1 Score.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0, solver="lbfgs"
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, max_depth=None
        ),
    }

    results = {}
    labels = sorted(y_train.unique())

    print("🤖 Training Models...")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n  📌 Training: {name}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

        results[name] = {
            "model": model,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
        }

        print(f"     Accuracy  : {acc:.4f}")
        print(f"     Precision : {prec:.4f}")
        print(f"     Recall    : {rec:.4f}")
        print(f"     F1 Score  : {f1:.4f}")

    print("\n" + "=" * 70)
    return results, labels


def select_best_model(results: dict) -> tuple:
    """Select the best model based on F1 Score."""
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best = results[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   F1 Score: {best['f1_score']:.4f}")

    return best_name, best


def save_confusion_matrix(cm, labels, best_name: str, output_path: str):
    """Save confusion matrix as a heatmap image."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  📊 Confusion matrix saved: {output_path}")


# Domain-specific stopwords to exclude from visualizations (non-sentiment-carrying nouns/verbs)
DOMAIN_STOPWORDS = {
    "app", "apps", "teams", "microsoft", "work", "chat", "meeting", "meetings", 
    "phone", "zoom", "skype", "use", "get", "wont", "doesnt", "cant", "even", 
    "call", "calls", "device", "devices", "screen", "team", "mobile", "version", 
    "update", "updates", "issue", "issues", "time", "people", "user", "users", 
    "service", "services", "video", "audio", "tab", "tabs", "application", "applications",
    "teamsapp", "msteams", "need", "like"
}


def generate_wordclouds(df: pd.DataFrame):
    """Generate positive and negative word clouds."""
    print("\n☁️  Generating word clouds...")

    sentiments = {
        "positive": df[df["sentiment"] == "Positive"]["clean_review"],
        "negative": df[df["sentiment"] == "Negative"]["clean_review"],
    }

    for sentiment, reviews in sentiments.items():
        text = " ".join(reviews.dropna().values)
        # Filter out domain stopwords
        words = [w for w in text.split() if w not in DOMAIN_STOPWORDS]
        filtered_text = " ".join(words)

        if not filtered_text.strip():
            print(f"   ⚠️  No text for {sentiment} word cloud, skipping.")
            continue

        if sentiment == "positive":
            colormap = "Greens"
        else:
            colormap = "Reds"

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap=colormap,
            max_words=150,
            max_font_size=100,
            random_state=RANDOM_STATE,
            collocations=False,
        )

        wc.generate(filtered_text)

        output_path = os.path.join(WORDCLOUD_DIR, f"{sentiment}_wordcloud.png")
        wc.to_file(output_path)
        print(f"   ✅ {sentiment.capitalize()} word cloud saved: {output_path}")


def extract_top_keywords(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Extract top N keywords for positive and negative sentiments."""
    print(f"\n🔑 Extracting top {top_n} keywords per sentiment...")

    keywords = {}

    for sentiment in ["Positive", "Negative"]:
        reviews = df[df["sentiment"] == sentiment]["clean_review"].dropna()
        all_words = " ".join(reviews.values).split()
        
        # Filter out domain stopwords
        filtered_words = [w for w in all_words if w not in DOMAIN_STOPWORDS]
        
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)
        keywords[sentiment.lower()] = [{"word": w, "count": c} for w, c in top_words]

        print(f"   {sentiment}: {[w for w, c in top_words[:5]]}...")

    return keywords


def save_artifacts(best_name, best_result, vectorizer, results, keywords, labels):
    """Save model, vectorizer, metrics, and keywords."""
    print("\n💾 Saving artifacts...")

    # Save best model
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(best_result["model"], model_path)
    print(f"   ✅ Model saved: {model_path}")

    # Save TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, "tfidf.pkl")
    joblib.dump(vectorizer, tfidf_path)
    print(f"   ✅ TF-IDF saved: {tfidf_path}")

    # Save metrics
    metrics = {
        "best_model": best_name,
        "labels": list(labels),
        "models": {},
    }

    for name, res in results.items():
        metrics["models"][name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"],
            "classification_report": res["classification_report"],
        }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"   ✅ Metrics saved: {metrics_path}")

    # Save keywords
    keywords_path = os.path.join(MODELS_DIR, "keywords.json")
    with open(keywords_path, "w") as f:
        json.dump(keywords, f, indent=2)
    print(f"   ✅ Keywords saved: {keywords_path}")


def main():
    """Main training pipeline."""
    create_directories()

    # 1. Load data
    df = load_data(DATA_FILE)

    # 2. Extract features
    X, y, vectorizer = extract_features(df)

    # 3. Split data
    print(f"✂️  Splitting data: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

    # 4. Train and evaluate models
    results, labels = train_models(X_train, X_test, y_train, y_test)

    # 5. Select best model
    best_name, best_result = select_best_model(results)

    # 6. Print classification report for best model
    print(f"\n📋 Classification Report — {best_name}:")
    print(best_result["classification_report"])

    # 7. Save confusion matrix
    cm_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
    save_confusion_matrix(best_result["confusion_matrix"], labels, best_name, cm_path)

    # 8. Generate word clouds
    generate_wordclouds(df)

    # 9. Extract keywords
    keywords = extract_top_keywords(df)

    # 10. Save all artifacts
    save_artifacts(best_name, best_result, vectorizer, results, keywords, labels)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"🎉 Training Pipeline Complete!")
    print(f"{'=' * 70}")
    print(f"  Best Model     : {best_name}")
    print(f"  Accuracy       : {best_result['accuracy']:.4f}")
    print(f"  F1 Score       : {best_result['f1_score']:.4f}")
    print(f"  Model File     : models/model.pkl")
    print(f"  Vectorizer     : models/tfidf.pkl")
    print(f"  Metrics        : models/metrics.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
