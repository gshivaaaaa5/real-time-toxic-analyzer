from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./toxic-unbiased-local"

print(f"📦 Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness"
]

threshold = 0.5

comments = [
    "go fucker.fuck off and die",
    "Have a nice day!",
    "Nobody cares what you think.",
    "You're annoying AF and dumb.",
    "You should just kill yourself.",
    "I appreciate your effort, well done!",
]

def check_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)[0]

    print(f"\n🗣️ \"{text}\"")
    print("🧪 Label Probabilities:")
    toxic_flag = False

    for i, score in enumerate(probs):
        label = labels[i]
        value = score.item()
        print(f"  → {label:20}: {value:.4f}")
        if value > threshold and label in ["toxicity", "severe_toxicity", "obscene", "insult", "threat", "identity_attack"]:
            toxic_flag = True

    print("⚠️ TOXIC" if toxic_flag else "✅ Not toxic")

for comment in comments:
    check_toxicity(comment)
