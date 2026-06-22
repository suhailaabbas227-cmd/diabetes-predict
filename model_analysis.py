# ============================================================
#  Diabetes Model — EDA, Cross-Validation & Tuning Analysis
#  Run:  python model_analysis.py
#
#  Generates report-ready figures and demonstrates:
#   - Exploratory Data Analysis (class balance, correlations)
#   - Dimensionality reduction (PCA)
#   - Feature importance
#   - Stratified K-Fold cross-validation
#   - Hyperparameter tuning (GridSearchCV)
# ============================================================

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = ["No Diabetes", "Pre-Diabetes", "Diabetes"]
COLORS = ["#22c55e", "#f59e0b", "#ef4444"]

print("=" * 55)
print("  DIABETES MODEL — ANALYSIS")
print("=" * 55)

df = pd.read_csv("diabetes_012_health_indicators_BRFSS2021.csv")
df["Diabetes_012"] = df["Diabetes_012"].astype(int)
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# ── 1. Class Distribution ────────────────────────────────────
print("\n[1/6] Class distribution plot...")
counts = y.value_counts().sort_index()
plt.figure(figsize=(6, 4))
sns.barplot(x=[LABELS[i] for i in counts.index], y=counts.values, palette=COLORS)
plt.title("Class Distribution (Target Imbalance)")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=120)
plt.close()

# ── 2. Correlation Heatmap ───────────────────────────────────
print("[2/6] Correlation heatmap...")
plt.figure(figsize=(13, 11))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=120)
plt.close()

# ── 3. Feature Importance (from trained model) ───────────────
print("[3/6] Feature importance plot...")
model = joblib.load("diabetes_model.pkl")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(8, 7))
imp.plot(kind="barh", color="#6366f1")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("eda_feature_importance.png", dpi=120)
plt.close()

# ── 4. PCA 2D Projection ─────────────────────────────────────
print("[4/6] PCA 2D projection...")
samp = df.sample(3000, random_state=42)
Xs = StandardScaler().fit_transform(samp.drop("Diabetes_012", axis=1))
pcs = PCA(n_components=2).fit_transform(Xs)
plt.figure(figsize=(7, 6))
for cls, c in zip([0, 1, 2], COLORS):
    m = samp["Diabetes_012"].values == cls
    plt.scatter(pcs[m, 0], pcs[m, 1], s=10, alpha=0.5, label=LABELS[cls], color=c)
plt.legend()
plt.title("PCA 2D Projection of Patients")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("eda_pca_scatter.png", dpi=120)
plt.close()

# ── 5. Stratified K-Fold Cross-Validation ────────────────────
print("[5/6] 5-fold cross-validation (on a 30k sample for speed)...")
samp2 = df.sample(30000, random_state=1)
Xc = StandardScaler().fit_transform(samp2.drop("Diabetes_012", axis=1))
yc = samp2["Diabetes_012"]
rf = RandomForestClassifier(n_estimators=80, max_depth=14,
                            min_samples_leaf=20, n_jobs=-1, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_val_score(rf, Xc, yc, cv=cv, scoring="accuracy")
print("   Fold accuracies :", [f"{s*100:.1f}%" for s in scores])
print(f"   Mean CV accuracy: {scores.mean()*100:.2f}%  (+/- {scores.std()*100:.2f}%)")

# ── 6. Hyperparameter Tuning (GridSearchCV) ──────────────────
print("[6/6] Hyperparameter tuning (GridSearchCV)...")
grid = {"max_depth": [10, 14, 18], "min_samples_leaf": [10, 20]}
gs = GridSearchCV(
    RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=42),
    grid, cv=3, scoring="accuracy",
)
gs.fit(Xc, yc)
print(f"   Best parameters : {gs.best_params_}")
print(f"   Best CV accuracy: {gs.best_score_*100:.2f}%")

print("\n" + "=" * 55)
print("  ANALYSIS COMPLETE")
print("  Figures saved:")
print("   - eda_class_distribution.png")
print("   - eda_correlation_heatmap.png")
print("   - eda_feature_importance.png")
print("   - eda_pca_scatter.png")
print("=" * 55)
