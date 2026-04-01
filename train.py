
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("patient_feedback_dataset.csv")

# Input (text)
X_text = df["Feedback"]

# Output (Sentiment)
y_sentiment = df["Sentiment"]

# Output (Theme)
y_theme = df["Theme"]

# Convert text → numbers
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2)  
)
X = vectorizer.fit_transform(X_text)

# Train Sentiment Model
sentiment_model = LogisticRegression()
sentiment_model.fit(X, y_sentiment)

# Train Theme Model
theme_model = LogisticRegression()
theme_model.fit(X, y_theme)

# Save everything
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(sentiment_model, open("sentiment_model.pkl", "wb"))
pickle.dump(theme_model, open("theme_model.pkl", "wb"))

print("Models trained and saved!")