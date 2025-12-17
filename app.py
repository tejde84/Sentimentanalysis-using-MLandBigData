from flask import Flask, request, render_template, session
import pickle
import numpy as np
from collections import Counter

app = Flask(__name__)
app.secret_key = "sentiment_secret_key"

# Load pipeline
pipe = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]

def top_keywords(text, n=5):
    # Transform single text to TF-IDF vector
    vec = tfidf.transform([text])
    # Get coefficients (feature importances) from logistic regression
    coefs = clf.coef_[0]
    # Pair non-zero tf-idf features with their weights
    idxs = vec.nonzero()[1]
    pairs = [(tfidf.get_feature_names_out()[i], coefs[i] * vec.data[list(idxs).index(i)]) for i in idxs]
    # Sort by absolute influence
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    return [w for w, _ in pairs_sorted[:n]]

@app.route("/", methods=["GET"])
def home():
    return render_template("sentiment_index.html", history=session.get("history", []))

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()
    if not message:
        return render_template("sentiment_index.html", error="Please enter text to analyze.", history=session.get("history", []))

    prob = pipe.predict_proba([message])[0]
    pred = int(np.argmax(prob))
    label = "Positive ✅" if pred == 1 else "Negative 🚫"
    confidence = round(100 * prob[pred], 2)

    # Stats
    words = message.split()
    word_count = len(words)
    char_count = len(message)
    unique_tokens = len(set(words))

    # Keywords
    keywords = top_keywords(message, n=5)

    # History
    history = session.get("history", [])
    history.insert(0, {
        "label": label, "confidence": confidence, "message": message,
        "words": word_count, "chars": char_count, "unique": unique_tokens, "keywords": keywords
    })
    session["history"] = history[:5]

    return render_template(
        "sentiment_index.html",
        label=label, confidence=confidence, message=message,
        words=word_count, chars=char_count, unique=unique_tokens,
        keywords=keywords, history=session["history"]
    )

@app.route("/about")
def about():
    return """
    <h2 style='color:#e0e0e0;background:#121212;padding:20px;'>About</h2>
    <p style='color:#cfcfcf;background:#121212;padding:0 20px 20px;'>Built by Tejas. Logistic Regression with TF-IDF, deployed via Flask.</p>
    """
    
if __name__ == "__main__":
    app.run(debug=True)