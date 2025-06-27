# twitter_streamer.py
import tweepy
import requests
import os

# Load your keys from environment
BEARER = os.getenv("YOUR_REAL_BEARER_TOKEN")
TRACK_TERMS = os.getenv("TWITTER_TRACK", "chatgpt,openai").split(',')

class MyStream(tweepy.StreamingClient):
    def on_connect(self):
        print("✅ Connected to Twitter stream")

    def on_tweet(self, tweet):
        text = tweet.text
        print("🆕 Tweet:", text)
        resp = requests.post("http://localhost:8000/analyze", json={"text": text})
        print("📊 Analysis:", resp.json())

stream = MyStream(bearer_token=BEARER)

# Add tracking rules
for term in TRACK_TERMS:
    stream.add_rules(tweepy.StreamRule(term))

stream.filter(expansions=None, tweet_fields=None)
