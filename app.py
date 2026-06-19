# ============================================================
#  Diabetes Prediction — Flask Web App (Backend API)
#  Run:  python app.py
#  Open: http://127.0.0.1:5000
# ============================================================

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import sys

# Make unicode prints safe on the Windows (cp1252) console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__)

# ── Load trained model + scaler ──────────────────────────────
model  = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature order — MUST match the training data
FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
    'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex',
    'Age', 'Education', 'Income'
]

# Human-friendly names for charts / UI
FEATURE_LABELS = {
    'HighBP': 'High Blood Pressure', 'HighChol': 'High Cholesterol',
    'CholCheck': 'Cholesterol Check', 'BMI': 'Body Mass Index',
    'Smoker': 'Smoker', 'Stroke': 'Stroke History',
    'HeartDiseaseorAttack': 'Heart Disease', 'PhysActivity': 'Physical Activity',
    'Fruits': 'Eats Fruits', 'Veggies': 'Eats Vegetables',
    'HvyAlcoholConsump': 'Heavy Alcohol Use', 'AnyHealthcare': 'Healthcare Coverage',
    'NoDocbcCost': 'Skipped Doctor (cost)', 'GenHlth': 'General Health',
    'MentHlth': 'Poor Mental-Health Days', 'PhysHlth': 'Poor Physical-Health Days',
    'DiffWalk': 'Difficulty Walking', 'Sex': 'Sex',
    'Age': 'Age Group', 'Education': 'Education', 'Income': 'Income'
}

CLASS_LABELS = {0: 'No Diabetes', 1: 'Pre-Diabetes', 2: 'Diabetes'}
CLASS_COLORS = {0: 'green', 1: 'orange', 2: 'red'}
CLASS_ADVICE = {
    0: "Your indicators suggest a low diabetes risk. Keep it up with regular exercise, "
       "a balanced diet, and routine check-ups.",
    1: "Your indicators suggest possible pre-diabetes. Consider consulting a doctor — "
       "early lifestyle changes can prevent progression to Type 2 diabetes.",
    2: "Your indicators suggest an elevated diabetes risk. Please consult a healthcare "
       "professional for a proper diagnosis and guidance."
}

# Model metadata for the dashboard (from train_model.py results)
MODEL_META = {
    'algorithm': 'Random Forest',
    'accuracy': 80.0,
    'train_samples': 236378,
    'n_features': len(FEATURES),
    'classes': 3,
    'balancing': 'SMOTE'
}

# Pre-compute top feature importances once at startup
def _top_features(k=8):
    if not hasattr(model, 'feature_importances_'):
        return []
    pairs = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda p: p[1], reverse=True
    )[:k]
    return [
        {'name': FEATURE_LABELS.get(f, f), 'importance': round(float(v) * 100, 1)}
        for f, v in pairs
    ]

TOP_FEATURES = _top_features()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/model-info')
def model_info():
    """Metadata the dashboard renders (stats + feature-importance chart)."""
    return jsonify({**MODEL_META, 'top_features': TOP_FEATURES})


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        features = [float(data[f]) for f in FEATURES]
        features_scaled = scaler.transform(pd.DataFrame([features], columns=FEATURES))

        prediction    = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]

        # Single 0-100 "risk score": pre-diabetes counts half, diabetes full
        risk_score = round((probabilities[1] * 0.5 + probabilities[2]) * 100, 1)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'label': CLASS_LABELS[prediction],
            'color': CLASS_COLORS[prediction],
            'advice': CLASS_ADVICE[prediction],
            'risk_score': risk_score,
            'probabilities': {
                'No Diabetes':  round(float(probabilities[0]) * 100, 1),
                'Pre-Diabetes': round(float(probabilities[1]) * 100, 1),
                'Diabetes':     round(float(probabilities[2]) * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("Diabetes Prediction App running at http://127.0.0.1:5000")
    app.run(debug=True)
