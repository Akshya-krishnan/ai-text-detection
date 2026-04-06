# 🧠 AI vs Human Text Detector

An AI-powered web application that classifies whether a given text is **AI-generated** or **human-written** using Machine Learning.

---

## 🚀 Features

* 🔍 Detect AI vs Human text
* 📊 Advanced NLP-based feature extraction
* 🧾 Explanation report ("Why this result?")
* 📥 Download result as report
* 🌐 Full-stack web application (Frontend + Backend)

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* Scikit-learn (SVM Model)
* NLP (NLTK, SpaCy)
* Gensim (Doc2Vec)

### Frontend

* HTML
* CSS
* JavaScript (Vanilla JS)

---

## 📂 Project Structure

```
ai-text-detector/
│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── explain.py
│   ├── requirements.txt
│   └── models/
│       ├── svm_model.pkl
│       ├── scaler.pkl
│       └── doc2vec_model.pkl
│
├── frontend/
│   ├── index.html
│   ├── detect.html
│   ├── script.js
│   └── style.css
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone <your-repo-link>
cd ai-text-detector/backend
```

### 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   (Windows)
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Install SpaCy model

```
python -m spacy download en_core_web_sm
```

### 5. Run backend server

```
python app.py
```

### 6. Run frontend

Open `frontend/index.html` in browser

---

## 🧠 How It Works

1. User enters text in frontend
2. Text sent to Flask API
3. Backend extracts NLP features:

   * Punctuation usage
   * POS tagging
   * Readability metrics
   * Sentence complexity
4. Doc2Vec converts text into vectors
5. Features passed into trained SVM model
6. Model predicts:

   * AI Generated
   * Human Written
7. Explanation generated based on features

---

## 📸 Screenshots

(Add screenshots here for better presentation)

---

## 📊 Model Details

* Algorithm: Support Vector Machine (SVM)
* Vectorization: Doc2Vec
* Feature Engineering: Custom NLP pipeline
* Threshold-based classification (0.6)

---

## 📌 Future Improvements

* Add confidence score (%)
* Improve UI/UX design
* Deploy on cloud (Render / Vercel)
* Add multi-language support

---

## 👩‍💻 Author

Akshya
MSc Computer Science (AI)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
