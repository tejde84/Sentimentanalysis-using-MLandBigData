

# 🧠 Sentiment Analysis Web App

A Flask‑based machine learning application that classifies text reviews as **Positive** or **Negative** using the IMDb dataset. Built with Python, scikit‑learn, and TF‑IDF + Logistic Regression, this project demonstrates end‑to‑end ML deployment with a clean dark‑blue themed UI.

---

## 🚀 Features
- **Text Classification**: Predicts sentiment of user‑entered text.
- **Confidence Score**: Displays probability of prediction.
- **Text Insights**: Word count, character count, unique tokens, and top keywords.
- **History Tracking**: Shows recent analyses in the session.
- **Modern UI**: Dark blue gradient background with glowing accents.

---

## 📂 Project Structure
```
Sentimentanalysis/
│
├── app.py                 # Flask app
├── sentiment_model.pkl     # Trained ML model
├── train_sentiment.py      # Script to train and save model
├── templates/
│   └── sentiment_index.html # Frontend UI
├── static/                 # (optional) CSS/JS/images
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Sentimentanalysis.git
   cd Sentimentanalysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional)**
   ```bash
   python train_sentiment.py
   ```
   This generates `sentiment_model.pkl`.

4. **Run the Flask app**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://127.0.0.1:5000/
   ```

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Flask** – Web framework
- **scikit‑learn** – ML pipeline
- **pandas** – Data handling
- **Bootstrap 5** – Responsive UI



## 📊 Model Details
- **Dataset**: IMDb movie reviews (`aclImdb` dataset)
- **Preprocessing**: TF‑IDF vectorization (10,000 features, English stopwords)
- **Algorithm**: Logistic Regression (max_iter=1000)
- **Accuracy**: ~88% on test split

---

## 👨‍💻 Author
**Tejas**  
Passionate about building ML + Web apps with polished UI and recruiter‑friendly demos.  
