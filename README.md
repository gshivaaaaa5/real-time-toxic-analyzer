🧠 Real-Time Toxicity, Sentiment & Emotion Analyzer
A full-stack NLP-powered project that analyzes live Reddit comments and user-inputted text for:
•	🔥 Toxicity (multi-label: threat, insult, identity hate, obscene, etc.)
•	😊 Sentiment (positive / neutral / negative with confidence score)
•	🎭 Emotion (joy, anger, sadness, fear, surprise, love)
It provides a real-time visual interface with live scoring of each text and high-toxicity flagging, useful for moderation, safety, and sentiment analysis tools.
________________________________________
📌 Project Highlights
•	✅ Real-time Reddit comment monitoring with instant NLP scoring
•	🔪 Uses powerful Hugging Face Transformers for high-quality predictions
•	🌐 FastAPI backend for scalable and asynchronous model serving
•	💻 Clean HTML + JavaScript UI for simple, responsive front-end
•	🧠 Displays top 3 emotions, confidence-based sentiment, and detailed toxicity scores
•	⚫ Automatically flags comments with high toxicity levels
________________________________________
💡 Idea Behind It
Toxicity, hate speech, and emotional distress are widely prevalent in online discussions. Platforms like Reddit, Twitter, or forums often face challenges with moderating such content. This project was built to:
•	📊 Analyze comment sections in real-time for harmful or emotionally charged content
•	🔎 Monitor online behavior for mental health or community guideline violations
•	💛 Encourage safer, healthier digital spaces by enabling automated tools
•	🤝 Provide open-source groundwork for developers building moderation tools, dashboards, or sentiment trackers
This is not only useful for individuals or devs learning NLP, but has real-life implications for companies managing large-scale communities.
________________________________________
🔧 Features
Feature	Description
🔍 Analyze Any Text	Input a text and instantly get sentiment, toxicity, and emotions
🔴 Flag Toxic Comments	Comments with high scores (e.g., toxic > 0.7) get flagged
💬 Live Reddit Streaming	Monitors subreddits using PRAW and analyzes comments instantly
🧠 Emotion Detection	Detects top 3 emotions like joy, anger, sadness, etc.
🟢 Sentiment Detection	Identifies sentiment with 0-1 confidence levels
⚡ FastAPI Backend	Serves NLP analysis requests quickly with async support
________________________________________
⚙️ Stack Used
Layer	Tools
Frontend	HTML, JavaScript (Vanilla)
Backend	Python, FastAPI, Pydantic
NLP Models	Hugging Face Transformers, Torch
Reddit API	praw Python Wrapper for Reddit API
Deployment	CORS-enabled FastAPI; can be hosted on Render, AWS, Heroku
Visualization	HTML console, optional matplotlib/seaborn analysis
________________________________________
🤖 NLP Models Used
Task	Model
🔪 Toxicity Detection	unitary/unbiased-toxic-roberta
🟢 Sentiment Analysis	cardiffnlp/twitter-roberta-base-sentiment
🎭 Emotion Detection	bhadresh-savani/distilbert-base-uncased-emotion
All models are transformer-based, used via Hugging Face's pipeline for ease of implementation.
________________________________________
🛠️ How It Works
1. Frontend (index.html)
•	User enters a comment
•	AJAX request sends the comment to http://localhost:8000/analyze
•	Displays sentiment, top 3 emotions, and toxicity categories with scores
•	Flags if any toxicity label > threshold (e.g., 0.7)
2. Backend (api.py)
•	FastAPI receives input
•	Runs three transformer models:
o	tox_model returns multi-label toxicity probabilities
o	sent_model returns sentiment class and score
o	emo_model returns emotion probabilities
•	Responds with structured JSON for rendering in UI
3. Reddit Streamer (reddit_streamer.py)
•	Connects to subreddit via praw
•	Fetches comments live (streaming mode)
•	Sends each comment to FastAPI /analyze
•	Can optionally save results, flag content, or notify via webhook
________________________________________
🔐 Ethics & Caution
•	⚠️ Not 100% accurate; models may struggle with sarcasm, mixed tone, or context
•	⚠️ Avoid using the tool to make real judgments on individuals
•	⚠️ Best used for pattern detection, trend visualization, or early flagging
•	⚠️ This tool is not a replacement for human moderation
________________________________________
🏃 Getting Started
🔹 Step 1: Clone Repo & Setup Env
pip install -r requirements.txt
🔹 Step 2: Run Backend
uvicorn api:app --reload --port 8000
🔹 Step 3: Open Frontend
Just open index.html in your browser.
🔹 Step 4: Optional Reddit Stream
python reddit_streamer.py
________________________________________
📁 Folder Structure
project-root/
│
├── api.py                # FastAPI backend
├── index.html            # Frontend UI
├── reddit_streamer.py    # Reddit streaming code
├── visualize_results.py  # (Optional) Graph generation code
├── requirements.txt      # Dependencies
├── README.md             # This documentation
└── /toxic-unbiased-local # Local model path (optional, can use Hugging Face directly)
________________________________________
💼 Business Potential
This project can be extended into a SaaS platform or tool for:
•	🔎 Moderation systems for social apps, forums, or news portals
•	💊 Mental health analysis tools analyzing user stress signals
•	📊 Dashboard apps monitoring toxic trends during events
•	📢 Alert systems that notify on dangerous or hateful spikes
Perfect use-case for startups building on community safety, mental health, or public sentiment platforms.
________________________________________
📄 Resume Tips
Include something like:
"Developed a full-stack NLP-powered moderation tool using FastAPI, Hugging Face Transformers, and Reddit APIs. Classified text for sentiment, emotion, and toxicity in real-time with a live UI."
Or:
"Built a real-time AI analyzer for social content. Integrated transformer models to process Reddit data streams and analyze emotional tone, toxicity, and sentiment. Visualized live results via custom frontend."
________________________________________
🧠 Future Improvements
•	🔒 Add user login & authentication
•	☁️ Deploy API + frontend to Render, Vercel, or AWS
•	📈 Create an admin dashboard with charts over time
•	🌍 Add Geo-tagging for regional trends
•	🔔 Push alerts via Telegram bot or email
________________________________________
📄 License
MIT License — feel free to fork, use, or build upon it. Just give credits if it helps you ✨
________________________________________
