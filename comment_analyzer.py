import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === Load Models ===

# 1. Toxicity Detection (local)
toxicity_model_path = "./toxic-unbiased-local"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_path)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_path)
toxicity_labels = [
    "toxicity", "severe_toxicity", "obscene", "identity_attack", "insult",
    "threat", "sexual_explicit", "male", "female", "homosexual_gay_or_lesbian",
    "christian", "jewish", "muslim", "black", "white", "psychiatric_or_mental_illness"
]

# 2. Sentiment
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# 3. Emotion
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

# === Analyze Function ===

def analyze_comment(text, toxicity_thresh=0.5, emotion_thresh=0.3):
    result = {"comment": text}

    # --- Toxicity ---
    tox_inputs = toxicity_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        tox_logits = toxicity_model(**tox_inputs).logits
    tox_probs = torch.sigmoid(tox_logits)[0]

    for i, score in enumerate(tox_probs):
        label = toxicity_labels[i]
        result[f"tox_{label}"] = round(score.item(), 4)

    result["is_toxic"] = any(
        tox_probs[i].item() > toxicity_thresh and toxicity_labels[i] in [
            "toxicity", "severe_toxicity", "obscene", "insult", "threat", "identity_attack"
        ]
        for i in range(len(toxicity_labels))
    )

    # --- Sentiment ---
    sent_output = sentiment_model(text)[0]
    result["sentiment"] = sent_output["label"]
    result["sentiment_score"] = round(sent_output["score"], 4)

    # --- Emotion ---
    emo_output = emotion_model(text)[0]  # list of dicts
    for emo in emo_output:
        if emo["score"] > emotion_thresh:
            result[f"emo_{emo['label']}"] = round(emo["score"], 4)

    return result

# === Sample Comments ===

comments = [
    "go fucker.fuck off and die",
    "Have a nice day!",
    "Nobody cares what you think.",
    "You're annoying AF and dumb.",
    "You should just kill yourself.",
    "I appreciate your effort, well done!"
]

results = [analyze_comment(text) for text in comments]
df = pd.DataFrame(results)

# === Save to CSV ===
df.to_csv("comment_analysis_results.csv", index=False)
print("✅ Results saved to 'comment_analysis_results.csv'")

# === Summary Formatter ===

def summarize_comment(row):
    print(f"\n🗣️ Comment: {row['comment']}\n")

    # 1. Top 3 Toxicity Labels > 0.5
    tox_labels = [
        (col.replace("tox_", ""), row[col])
        for col in row.index if col.startswith("tox_") and row[col] > 0.5
    ]
    tox_labels = sorted(tox_labels, key=lambda x: x[1], reverse=True)[:3]

    if tox_labels:
        print("☣️ Top Toxicity Labels:")
        for label, score in tox_labels:
            print(f"   → {label}: {score:.4f}")
    else:
        print("✅ No high toxicity detected.")

    # 2. Sentiment
    sentiment_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    sent = sentiment_map.get(row['sentiment'], row['sentiment'])
    print(f"\n💬 Sentiment: {sent} ({row['sentiment_score']:.4f})")

    # 3. Top 3 Emotions
    emotion_scores = [
        (col.replace("emo_", ""), row[col])
        for col in row.index if col.startswith("emo_") and not pd.isna(row[col])
    ]
    top_emotions = sorted(emotion_scores, key=lambda x: x[1], reverse=True)[:3]

    if top_emotions:
        print("\n🎭 Top Emotions:")
        for label, score in top_emotions:
            print(f"   → {label}: {score:.4f}")
    else:
        print("\n🎭 No strong emotions detected.")

# === Print Summaries ===
for _, row in df.iterrows():
    summarize_comment(row)
