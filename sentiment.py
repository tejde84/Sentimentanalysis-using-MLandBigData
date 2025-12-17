import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle

def load_imdb_data(data_dir):
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_name, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(1 if label_type == "pos" else 0)
    return pd.DataFrame({"text": texts, "label": labels})

if __name__ == "__main__":
    # Adjust path to where you extracted IMDb dataset
    train_df = load_imdb_data("aclImdb/train")

    print(f"Loaded {len(train_df)} reviews")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        train_df["text"], train_df["label"], test_size=0.2, random_state=42
    )

    # Build pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    with open("sentiment_model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    print("✅ Model saved as sentiment_model.pkl")