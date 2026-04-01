from fastapi import FastAPI
import pickle

app = FastAPI()

# Load models
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
sentiment_model = pickle.load(open("sentiment_model.pkl", "rb"))
theme_model = pickle.load(open("theme_model.pkl", "rb"))

@app.get("/analyze")
def analyze(text: str):
    X = vectorizer.transform([text])
    
    sentiment = sentiment_model.predict(X)[0]
    sentiment_conf = sentiment_model.predict_proba(X).max()

    theme_proba = theme_model.predict_proba(X).max()
    
    if theme_proba < 0.5:
        theme = "unknown"
    else:
        theme = theme_model.predict(X)[0]

    return {
        "text": text,
        "sentiment": int(sentiment),
        "sentiment_confidence": float(sentiment_conf),
        "theme": theme,
        "theme_confidence": float(theme_proba)
    }