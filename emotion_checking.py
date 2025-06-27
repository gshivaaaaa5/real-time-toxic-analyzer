from transformers import pipeline

# Load the emotion detection model
emo_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True
)

comments = [
    "I'm so excited to start this new job!",
    "Ugh, this sucks. I hate everything right now.",
    "Thanks a ton, I appreciate your help.",
    "I'm scared... what if it doesn't work?",
    "You always make me smile :)"
]

for text in comments:
    print(f"\n🗣️ \"{text}\"")
    outputs = emo_classifier(text)[0]  # list of dicts
    
    # Sort emotions by score
    outputs = sorted(outputs, key=lambda x: x['score'], reverse=True)
    
    # Print top 3 emotions
    for entry in outputs[:3]:
        label, score = entry['label'], entry['score']
        print(f"🎭 {label:8} : {score:.4f}")
