from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "unitary/unbiased-toxic-roberta"
save_dir = "./toxic-unbiased-local"

print(f"📦 Downloading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("🔢 Label count:", model.config.num_labels)
print("🧾 Labels:", model.config.id2label)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ Saved model locally at {save_dir}")
