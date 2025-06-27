from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Toxicity Model
tox_model_path = "./toxic-unbiased-local"
tox_tokenizer = AutoTokenizer.from_pretrained(tox_model_path)
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model_path)
tox_labels = [
    "toxicity", "severe_toxicity", "obscene", "identity_attack",
    "insult", "threat", "sexual_explicit", "male", "female",
    "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
    "black", "white", "psychiatric_or_mental_illness"
]

# Sentiment Model
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# Emotion Model
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

class InputText(BaseModel):
    text: str

live_comments = []

@app.post("/analyze")
def analyze_text(data: InputText):
    text = data.text

    # Toxicity
    tox_inputs = tox_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        tox_logits = tox_model(**tox_inputs).logits
    tox_probs = torch.sigmoid(tox_logits)[0].tolist()
    tox_dict = {f"toxicity_{tox_labels[i]}": round(tox_probs[i], 4) for i in range(len(tox_labels))}
    tox_top = sorted(
        [(tox_labels[i], tox_probs[i]) for i in range(len(tox_labels)) if tox_probs[i] > 0.5],
        key=lambda x: x[1], reverse=True
    )[:3]

    # Sentiment
    sent_out = sentiment_model(text)[0]
    sentiment_label = sentiment_map.get(sent_out["label"], "Unknown")
    sentiment_confidence = round(sent_out["score"], 4)

    # Emotion
    emo_out = emotion_model(text)[0]
    emo_top = sorted(emo_out, key=lambda x: x["score"], reverse=True)[:3]
    emotions = {f"emotion_{e['label']}": round(e["score"], 4) for e in emo_top}

    # Prepare final result with **flat** sentiment keys
    result_summary = {
        "text": text,
        "sentiment": sentiment_label,
        "confidence": sentiment_confidence,
        "top_3_toxicity_labels": [{"label": l, "score": round(s, 4)} for l, s in tox_top],
        "top_3_emotions": emotions
    }

    live_comments.insert(0, result_summary)
    if len(live_comments) > 20:
        live_comments.pop()

    return result_summary

@app.get("/recent")
def get_recent():
    return live_comments
