# 🩺 DiabetesPredict — AI Diabetes Risk Assessment

A machine-learning web app that estimates a person's diabetes risk (No Diabetes /
Pre-Diabetes / Diabetes) from 21 health indicators, trained on **236,000+ real
records** from the CDC's BRFSS 2021 survey.

**Live app:** _add your Streamlit Cloud link here after deploying_

---

## ✨ Features
- **Random Forest model** (~80% accuracy) trained with SMOTE class balancing
- **Interactive Streamlit dashboard** — tabbed form, animated risk gauge, probability bars
- **Feature-importance chart** showing what drives the prediction
- **User authentication** (sign up / log in) via Supabase, with "Remember me"
- Clean, modern glassmorphism UI

## 🧠 Tech Stack
| Layer | Tool |
|-------|------|
| Model | scikit-learn (Random Forest), imbalanced-learn (SMOTE) |
| Data | pandas, NumPy · BRFSS 2021 dataset |
| Web app | Streamlit, Plotly |
| Auth | Supabase (cloud) |

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
> Add your Supabase keys to `.streamlit/secrets.toml` (see `secrets.toml.example`).

## 📁 Project structure
```
Diabetes_Project/
├── streamlit_app.py     # main Streamlit web app (frontend + backend)
├── app.py               # alternative Flask version
├── train_model.py       # trains & saves the model
├── test_model.py        # evaluates the model on real data
├── model_analysis.py    # EDA, cross-validation & hyperparameter tuning
├── cloud_auth.py        # Supabase login / sign-up
├── requirements.txt     # Python dependencies
├── models/              # trained model + scaler
│   ├── diabetes_model.pkl
│   └── scaler.pkl
├── data/                # dataset
│   └── diabetes_012_health_indicators_BRFSS2021.csv
├── figures/             # EDA & evaluation charts
│   ├── confusion_matrix.png
│   ├── eda_class_distribution.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_feature_importance.png
│   └── eda_pca_scatter.png
├── static/              # Flask frontend (CSS / JS)
├── templates/           # Flask frontend (HTML)
└── .streamlit/          # Streamlit theme + secrets config
```

## 📊 Model performance
- Overall accuracy: **80.8%** on 236,378 records
- Strong on *No Diabetes* (88% recall); limited on the rare *Pre-Diabetes* class
  due to dataset imbalance.

## ⚠️ Disclaimer
For educational purposes only — **not a medical diagnosis**.
