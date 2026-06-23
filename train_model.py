# ============================================================
#  Diabetes Prediction - Model Training
#  Dataset : BRFSS 2021 (236,378 samples, 21 features)
#  Target  : Diabetes_012  (0=No Diabetes, 1=Pre-Diabetes, 2=Diabetes)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

# Make emoji / unicode prints safe on the Windows (cp1252) console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ── 1. Load Dataset ──────────────────────────────────────────
print("=" * 55)
print("  DIABETES PREDICTION — MODEL TRAINING")
print("=" * 55)

df = pd.read_csv('data/diabetes_012_health_indicators_BRFSS2021.csv')
df['Diabetes_012'] = df['Diabetes_012'].astype(int)

print(f"\n✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\n📊 Target Class Distribution:")
dist = df['Diabetes_012'].value_counts().sort_index()
labels = {0: 'No Diabetes', 1: 'Pre-Diabetes', 2: 'Diabetes'}
for cls, cnt in dist.items():
    pct = cnt / len(df) * 100
    print(f"   Class {cls} ({labels[cls]}): {cnt:,}  ({pct:.1f}%)")

print("\n🔍 Missing Values:", df.isnull().sum().sum())

# ── 2. Features & Target ──────────────────────────────────────
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# ── 3. Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train set : {X_train.shape[0]:,} samples")
print(f"   Test set  : {X_test.shape[0]:,} samples")

# ── 4. Handle Class Imbalance — SMOTE ────────────────────────
print("\n⚙️  Applying SMOTE to balance training classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE — {X_train_res.shape[0]:,} samples (balanced)")

# ── 5. Feature Scaling ────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)
print("✅ StandardScaler applied")

# ── 6. Model 1 — Random Forest ───────────────────────────────
# NOTE: depth/leaf limits keep the saved model small (a few MB vs >1 GB)
#       and reduce overfitting so probabilities are better calibrated.
print("\n🌲 Training Random Forest (150 trees, depth-limited)...")
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_sc, y_train_res)
y_pred_rf = rf.predict(X_test_sc)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"   Accuracy : {acc_rf:.4f}  ({acc_rf*100:.2f}%)")
print(classification_report(y_test, y_pred_rf,
      target_names=['No Diabetes', 'Pre-Diabetes', 'Diabetes']))

# ── 7. Model 2 — Logistic Regression ─────────────────────────
print("📈 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train_res)
y_pred_lr = lr.predict(X_test_sc)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"   Accuracy : {acc_lr:.4f}  ({acc_lr*100:.2f}%)")

# ── 8. Select & Save Best Model ──────────────────────────────
print("\n" + "─" * 40)
if acc_rf >= acc_lr:
    best_model, best_name = rf, "Random Forest"
    y_pred_best = y_pred_rf
else:
    best_model, best_name = lr, "Logistic Regression"
    y_pred_best = y_pred_lr

print(f"🏆 Best Model : {best_name}")
joblib.dump(best_model, 'models/diabetes_model.pkl', compress=3)
joblib.dump(scaler,     'models/scaler.pkl',         compress=3)
print("✅ diabetes_model.pkl saved")
print("✅ scaler.pkl saved")

# ── 9. Confusion Matrix Plot ──────────────────────────────────
import matplotlib
matplotlib.use('Agg')

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diab', 'Pre-Diab', 'Diab'],
            yticklabels=['No Diab', 'Pre-Diab', 'Diab'])
plt.title(f'Confusion Matrix — {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=120)
print("✅ confusion_matrix.png saved")

# ── 10. Feature Importance ────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.Series(best_model.feature_importances_,
                         index=X.columns).sort_values(ascending=False)
    print("\n📊 Top 10 Features:")
    print(feat_imp.head(10).to_string())

print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print("=" * 55)