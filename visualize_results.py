import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved analysis results
df = pd.read_csv("comment_analysis_results.csv", error_bad_lines=False)


# --- 1. Bar Chart for Toxicity Scores for First 5 Comments ---

tox_cols = [col for col in df.columns if col.startswith("tox_")]
subset_tox = df.loc[:4, tox_cols].T  # transpose for plotting

plt.figure(figsize=(12, 6))
sns.barplot(data=subset_tox, orient="h")
plt.title("Toxicity Scores for First 5 Comments")
plt.xlabel("Score")
plt.ylabel("Toxicity Labels")
plt.legend([f"Comment {i+1}" for i in range(5)])
plt.tight_layout()
plt.show()


# --- 2. Pie Chart for Sentiment Distribution ---

# Map sentiment codes to readable labels
sent_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}
sent_counts = df['sentiment'].map(sent_map).value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%", colors=["#ff4c4c", "#ffa500", "#4caf50"])
plt.title("Sentiment Distribution")
plt.show()


# --- 3. Heatmap for Emotion Probabilities ---

emo_cols = [col for col in df.columns if col.startswith("emo_")]

plt.figure(figsize=(14, 6))
sns.heatmap(df[emo_cols], annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Emotion Probabilities per Comment")
plt.xlabel("Emotions")
plt.ylabel("Comments (row index)")
plt.tight_layout()
plt.show()
