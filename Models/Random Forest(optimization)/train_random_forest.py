import pandas as pd
import os
import pickle
import numpy as np  # ← NEW

from sklearn.ensemble import RandomForestClassifier
#  NEW IMPORTS
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
#  NEW IMPORTS
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# =====================================================
#  Paths
# =====================================================

DATA_DIR  = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Data\processed"
MODEL_DIR = r"C:\Users\ofniz\Desktop\Heart Disease Prediction\Models\Random Forest(optimization)"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
#  Load Data  (UNCHANGED)
# =====================================================

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

print("Data loaded successfully")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")

# =====================================================
#  Use a sample to speed up search
# =====================================================

from sklearn.utils import resample

X_sample, y_sample = resample(
    X_train, y_train,
    n_samples=30000,
    random_state=42,
    stratify=y_train
)
print(f"Sample for search: {X_sample.shape}")

# =====================================================
#  Train Model
# =====================================================


# NEW: Hyperparameter search space

param_dist = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [5, 8, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [2, 4, 6],
    "max_features":      ["sqrt"],
    "class_weight":      [None, "balanced"],
}

# NEW: StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# NEW: RandomizedSearchCV replaces the single
# model.fit() — tries 50 combinations, picks best
# BEFORE: model = RandomForestClassifier(n_estimators=200, random_state=42)
#         model.fit(X_train, y_train)

print("\nStarting hyperparameter search (this may take a few minutes)...")

search = RandomizedSearchCV(
    estimator           = RandomForestClassifier(random_state=42),
    param_distributions = param_dist,
    n_iter              = 15,
    scoring             = "recall",
    cv                  = cv,
    n_jobs              = -1,
    verbose             = 1,
    random_state        = 42,
    return_train_score=True
)

search.fit(X_sample, y_sample)

print(f"\nBest parameters found:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

# =====================================================
# Overfitting Check  ← NEW
# =====================================================

results = search.cv_results_

best_index = search.best_index_

mean_train_score = results['mean_train_score'][best_index]
mean_test_score = results['mean_test_score'][best_index]
diff = mean_train_score - mean_test_score

print(f"\n--- Overfitting Check ---")
print(f"Mean Train Score : {mean_train_score:.4f}")
print(f"Mean CV Score    : {mean_test_score:.4f}")
print(f"Difference       : {diff:.4f}")

if diff <= 0.05:
    print("Result: Model is stable — no overfitting detected")
elif diff > 0.05 and mean_train_score >= 0.90:
    print("Result: Overfitting detected — model memorized training data")
else:
    print("Result: Model is not stable — consider adjusting parameters")



model = RandomForestClassifier(**search.best_params_, random_state=42)
model.fit(X_train, y_train)
print("Done.")


print("Random Forest optimized training completed")

# =====================================================
#  Evaluate  (UNCHANGED)
# =====================================================

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

# =====================================================
#  Save Model & Evaluation
# =====================================================

with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

eval_df = pd.DataFrame([{
    "Model":     "random_forest_optimized",
    "Accuracy":  round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall":    round(recall, 4),
    "F1-score":  round(f1, 4),
    "ROC-AUC":   round(roc_auc, 4),
}])
eval_df.to_csv(os.path.join(MODEL_DIR, "evaluation.csv"), index=False)

#  NEW: Save best params to CSV for reference
params_df = pd.DataFrame([search.best_params_])
params_df.to_csv(os.path.join(MODEL_DIR, "best_params.csv"), index=False)
print(f"Saved: {MODEL_DIR}\\best_params.csv")
#  NEW

print(f"Saved: {MODEL_DIR}\\model.pkl")
print(f"Saved: {MODEL_DIR}\\evaluation.csv")
print(f"Saved: {MODEL_DIR}\\best_params.csv")


