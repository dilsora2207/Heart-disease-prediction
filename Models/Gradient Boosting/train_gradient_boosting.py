import pandas as pd
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# =====================================================
# Paths
# =====================================================

DATA_DIR  = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Data\processed"
MODEL_DIR = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Models\Gradient Boosting"

os.makedirs(MODEL_DIR, exist_ok=True)

#=====================================================
#  Load Data
#=====================================================

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print("Data loaded successfully")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

#=====================================================
#  Train Model
#=====================================================

model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Gradient Boosting training completed")

#=====================================================
#  Evaluate
#=====================================================

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print(f"\n--- Evaluation Results ---")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

#=====================================================
#  Save Model & Evaluation
#=====================================================

with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

eval_df = pd.DataFrame([{
    "Model":     "gradient_boosting",
    "Accuracy":  round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall":    round(recall, 4),
    "F1-score":  round(f1, 4),
    "ROC-AUC":   round(roc_auc, 4),
}])
eval_df.to_csv(os.path.join(MODEL_DIR, "evaluation.csv"), index=False)

print(f"\nSaved: {MODEL_DIR}\\model.pkl")
print(f"Saved: {MODEL_DIR}\\evaluation.csv")