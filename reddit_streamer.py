# reddit_streamer.py
import praw
import requests
import os
import pandas as pd
import time

reddit = praw.Reddit(
    client_id="CdE9C6gFYrs20vE4IjlXWA",
    client_secret="O_ZWcU4X5TMXNGv_oQIaua-bB5-mqQ",
    user_agent="toxic-detector-bot-v1"
)

sub = reddit.subreddit(os.getenv("REDDIT_SUB", "all"))

print("✅ Connected to Reddit")

CSV_FILE = "comment_analysis_results.csv"

def flatten_result(response):
    flat = {
        "text": response.get("text"),
        "sentiment": response.get("sentiment"),
        "confidence": response.get("confidence")
    }

    # Toxicity
    for tox in response.get("top_3_toxicity_labels", []):
        flat[f"tox_{tox['label']}"] = tox['score']

    # Emotions
    for emo, score in response.get("top_3_emotions", {}).items():
        flat[emo] = score

    return flat

for comment in sub.stream.comments(skip_existing=True):
    try:
        text = comment.body
        print("💬 Comment:", text)

        resp = requests.post("http://localhost:8000/analyze", json={"text": text})
        result = resp.json()
        print("📊 Analysis:", result)

        # Save to CSV
        flat_result = flatten_result(result)
        df = pd.DataFrame([flat_result])
        df.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))

        time.sleep(1)  # optional: avoid rate limits

    except Exception as e:
        print("❌ Error:", e)
