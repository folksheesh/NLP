"""
Google Play Store Review Scraper
================================
Scrapes 10,000+ reviews from Microsoft Teams app on Google Play Store.
Adds automatic sentiment labeling based on star ratings.
"""

import os
import time
import pandas as pd
from google_play_scraper import Sort, reviews

# ── Configuration ──────────────────────────────────────────────────────────────
APP_ID = "com.microsoft.teams"
APP_NAME = "Microsoft Teams"
TARGET_REVIEWS = 10000
BATCH_SIZE = 200
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reviews.csv")


def scrape_reviews(app_id: str, target: int, batch_size: int) -> list:
    """
    Scrape reviews from Google Play Store using pagination.

    Args:
        app_id: Google Play Store app package name.
        target: Target number of reviews to scrape.
        batch_size: Number of reviews per batch request.

    Returns:
        List of review dictionaries.
    """
    all_reviews = []
    continuation_token = None
    batch_num = 0

    print(f"🔍 Scraping reviews for: {APP_NAME} ({app_id})")
    print(f"📊 Target: {target:,} reviews")
    print("=" * 60)

    while len(all_reviews) < target:
        batch_num += 1
        try:
            result, continuation_token = reviews(
                app_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token,
            )

            if not result:
                print(f"\n⚠️  No more reviews available. Total collected: {len(all_reviews):,}")
                break

            all_reviews.extend(result)
            progress = min(len(all_reviews), target)
            pct = (progress / target) * 100
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r  [{bar}] {progress:,}/{target:,} ({pct:.1f}%)", end="", flush=True)

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"\n❌ Error on batch {batch_num}: {e}")
            print("   Retrying in 5 seconds...")
            time.sleep(5)
            continue

    print(f"\n\n✅ Scraping complete! Total reviews collected: {len(all_reviews):,}")
    return all_reviews[:target]


def label_sentiment(rating: int) -> str:
    """
    Label sentiment based on star rating.

    Args:
        rating: Star rating (1-5).

    Returns:
        Sentiment label: 'Negative', 'Neutral', or 'Positive'.
    """
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def process_and_save(raw_reviews: list, output_path: str) -> pd.DataFrame:
    """
    Process raw review data and save to CSV.

    Args:
        raw_reviews: List of raw review dictionaries from scraper.
        output_path: Path to save CSV file.

    Returns:
        Processed DataFrame.
    """
    print("\n📝 Processing reviews...")

    df = pd.DataFrame(raw_reviews)

    # Select and rename relevant columns
    df = df[["content", "score", "at", "thumbsUpCount"]].copy()
    df.columns = ["review", "rating", "date", "likes"]

    # Clean data
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["review"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["likes"] = df["likes"].fillna(0).astype(int)

    # Add sentiment labels
    df["sentiment"] = df["rating"].apply(label_sentiment)

    # Reset index
    df = df.reset_index(drop=True)

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Total Reviews  : {len(df):,}")
    print(f"  Average Rating : {df['rating'].mean():.2f}")
    print(f"  Date Range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Sentiment Distribution:")
    for sentiment, count in df["sentiment"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {sentiment:10s} : {count:,} ({pct:.1f}%)")
    print(f"\n  💾 Saved to: {output_path}")
    print(f"{'=' * 60}")

    return df


def main():
    """Main execution function."""
    start_time = time.time()

    # Scrape reviews
    raw_reviews = scrape_reviews(APP_ID, TARGET_REVIEWS, BATCH_SIZE)

    if not raw_reviews:
        print("❌ No reviews were scraped. Please check your internet connection.")
        return

    # Process and save
    df = process_and_save(raw_reviews, OUTPUT_FILE)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
