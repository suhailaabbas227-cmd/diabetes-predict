# ============================================================
#  Test / Check the trained model
#  Run:  python test_model.py
#
#  Ye script 2 kaam karta hai:
#   1) Khud ke banaye huye example patients par prediction
#   2) Asli CSV ke random rows par prediction vs actual jawab
#      (taake pata chale model kitna sahi hai)
# ============================================================

import joblib
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, classification_report

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Feature order (CSV ke columns jaisa, target ke baghair)
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income",
]
LABELS = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}

model  = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("Model loaded:", type(model).__name__)
print("=" * 60)


def predict(patient: dict):
    """Ek patient (dict) leta hai aur prediction + probabilities deta hai."""
    x  = pd.DataFrame([[patient[f] for f in FEATURES]], columns=FEATURES)
    xs = scaler.transform(x)
    pred  = int(model.predict(xs)[0])
    proba = model.predict_proba(xs)[0]
    return pred, proba


# ── PART 1 — Apne banaye huye example patients ──────────────
print("\nPART 1 — Example patients (manual input)\n")

healthy = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22, "Smoker": 0,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
    "Age": 3, "Education": 6, "Income": 8,
}

high_risk = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 40, "Smoker": 1,
    "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0,
    "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 5, "MentHlth": 15, "PhysHlth": 20, "DiffWalk": 1, "Sex": 1,
    "Age": 11, "Education": 3, "Income": 2,
}

for name, p in [("Healthy young person", healthy), ("High-risk older person", high_risk)]:
    pred, proba = predict(p)
    print(f"  {name:24s} -> {LABELS[pred]}")
    print(f"     Probabilities: No={proba[0]*100:5.1f}%  Pre={proba[1]*100:5.1f}%  Diab={proba[2]*100:5.1f}%\n")

# ── PART 2 — Asli CSV data par test (quality check) ─────────
print("=" * 60)
print("\nPART 2 — Real data se test (model kitna sahi hai)\n")

df = pd.read_csv("data/diabetes_012_health_indicators_BRFSS2021.csv")
df["Diabetes_012"] = df["Diabetes_012"].astype(int)

# 10 random asli patients: prediction vs asli jawab
sample = df.sample(10, random_state=7)
print("  Asli-Jawab  vs  Model-Prediction")
print("  " + "-" * 38)
correct = 0
for _, r in sample.iterrows():
    patient = {f: r[f] for f in FEATURES}
    pred, _ = predict(patient)
    actual = int(r["Diabetes_012"])
    ok = "OK " if pred == actual else "X  "
    if pred == actual:
        correct += 1
    print(f"  {ok} {LABELS[actual]:13s} -> {LABELS[pred]}")
print(f"\n  In 10 random patients: {correct}/10 sahi\n")

# Pure dataset par accuracy (overall quality)
print("  Calculating overall accuracy on full dataset...")
X_all  = df[FEATURES]
y_all  = df["Diabetes_012"].values
y_pred = model.predict(scaler.transform(X_all))
print(f"\n  Overall accuracy: {accuracy_score(y_all, y_pred)*100:.2f}%\n")
print(classification_report(
    y_all, y_pred, target_names=["No Diabetes", "Pre-Diabetes", "Diabetes"]))
