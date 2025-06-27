from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F

# === Load Toxicity Model ===
tox_model_path = "./toxic-unbiased-local"
tox_tokenizer = AutoTokenizer.from_pretrained(tox_model_path)
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model_path)

tox_labels = [
    "toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat",
    "sexual_explicit", "male", "female", "homosexual_gay_or_lesbian", "christian",
    "jewish", "muslim", "black", "white", "psychiatric_or_mental_illness"
]
threshold = 0.5

# === Load Sentiment Model (CardiffNLP) ===
sent_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name)

sent_labels = ['Negative', 'Neutral', 'Positive']

comments = [
    "go fucker.fuck off and die",
    "Have a nice day!",
    "Nobody cares what you think.",
    "You're annoying AF and dumb.",
    "You should just kill yourself.",
    "I appreciate your effort, well done!",
]

def check_toxicity(text):
    inputs = tox_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = tox_model(**inputs).logits
    probs = torch.sigmoid(logits)[0]

    print(f"\n🗣️ \"{text}\"")
    print("🧪 Toxicity Scores:")
    toxic_flag = False

    for i, score in enumerate(probs):
        label = tox_labels[i]
        value = score.item()
        if value > threshold and label in ["toxicity", "severe_toxicity", "obscene", "insult", "threat", "identity_attack"]:
            toxic_flag = True
        if value > 0.01:  # Only show non-zeroish probs
            print(f"  → {label:20}: {value:.4f}")

    print("⚠️ TOXIC" if toxic_flag else "✅ Not toxic")


def check_sentiment(text):
    inputs = sent_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = sent_model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    
    print("💬 Sentiment Scores:")
    for i, score in enumerate(probs):
        print(f"  → {sent_labels[i]:8}: {score.item():.4f}")
    top_class = torch.argmax(probs).item()
    print(f"🎯 Sentiment: {sent_labels[top_class]}")


# === Combined Analysis ===
for comment in comments:
    check_toxicity(comment)
    check_sentiment(comment)
