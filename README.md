# 💳 SCRIS - Smart Credit Risk Intelligence System

**🚀 A machine learning-powered app that predicts whether a loan applicant will default or not, with SHAP-based explainability.**

---

## 📊 Problem Statement

Banks and fintech companies face major losses due to risky applicants.  
SCRIS helps them by:

- Predicting loan defaults using XGBoost
- Explaining every prediction with SHAP (feature attribution)
- Providing a web interface for easy usage (Streamlit)

---

## 📁 Dataset

- 📂 [Home Credit Default Risk – Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- We used only `application_train.csv` for simplicity and speed
- 307,000+ rows, 100+ features (post one-hot encoding)

---

## 🧠 ML Techniques Used

| Task | Tools |
|------|-------|
| Data Preprocessing | Pandas, One-Hot Encoding |
| Modeling | XGBoost |
| Class Imbalance | `scale_pos_weight` |
| Evaluation | ROC AUC, F1 Score |
| Explainability | SHAP (summary + waterfall plots) |

---

## 🖥️ Streamlit App Preview

### 🔼 Upload Input CSV  
![Upload Sample](screenshots/Streamlit.jpeg)

### 📊 Predictions + Probabilities  
![Prediction Output](screenshots/Prediction.jpeg)

### 🔍 SHAP Summary Plot  
![SHAP Summary](screenshots/sharp summary.png)

### 🧠 SHAP Waterfall (Single Prediction Explanation)  
![SHAP Waterfall](screenshots/individual.png)

---

## 📂 File Structure

📦 scris-ml-app/
├── app.py                 # Streamlit web app
├── SCRIS_Model.ipynb      # Jupyter notebook for model training
├── xgb_model.pkl          # Trained XGBoost model
├── requirements.txt       # Python dependencies
├── good_sample.csv        # Sample input file
├── README.md              # Project documentation
└── screenshots/           # Folder containing images below
    ├── upload.png
    ├── output.png
    ├── shap_summary.png
    └── shap_waterfall.png

---

## 🚀 How to Run Locally

### 🔧 Install dependencies

pip install -r requirements.txt
🖥️ Launch the app
bash
Copy
Edit
streamlit run app.py
🧪 Upload a CSV
Use good_sample.csv (or your own) to get predictions.

🧠 Model Explainability (SHAP)
We use SHAP to explain:

Which features made the model think an applicant is risky or safe

Why a prediction happened — even for individual customers

This makes the model transparent and business-usable ✅

💡 Why This Project?
✅ End-to-end: ML to UI

✅ Real-world dataset

✅ Explainable, responsible AI

✅ Internship-ready portfolio

🙋‍♀️ Author
Valli Viswa Varshini
📧 valliviswavarshini@gmail.com
🔗 GitHub Profile

📣 Like this project?
⭐️ Star this repo and connect with me on GitHub!
