import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

SEED = 42
N_ITER = 8      # same budget for XGBoost and LightGBM later
CV_FOLDS = 3

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# int8 uses 8x less memory than int64 for 0/1 columns
X_train = train.drop(columns=["diseases", "label_id"]).astype("int8")
y_train = train["label_id"]
X_test = test.drop(columns=["diseases", "label_id"]).astype("int8")
y_test = test["label_id"]

param_space = {
    "n_estimators": [100, 150, 200],
    "max_depth": [20, 30, 40],          # no unlimited depth (memory)
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2"],
    "class_weight": [None, "balanced_subsample"],
}

search = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=SEED),
    param_distributions=param_space,
    n_iter=N_ITER,
    scoring="f1_macro",
    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
    random_state=SEED,
    refit=True,
    verbose=2,
)

# Tuning uses train data only (cross-validation)
t0 = time.time()
search.fit(X_train, y_train)
total_time = time.time() - t0
print(f"\nSearch done in {total_time / 60:.1f} min")
print("Best params:", search.best_params_)
print(f"Best CV macro F1: {search.best_score_:.4f}")

# Test set is used once, for the final number
best = search.best_estimator_
t0 = time.time()
pred = best.predict(X_test)
predict_time = time.time() - t0

results = {
    "model": "RandomForest (tuned)",
    "accuracy": accuracy_score(y_test, pred),
    "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
    "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    "cv_f1_macro": search.best_score_,
    "search_time_min": round(total_time / 60, 1),
    "predict_time_sec": round(predict_time, 1),
    "best_params": str(search.best_params_),
}

os.makedirs("results", exist_ok=True)
pd.DataFrame([results]).to_csv("results/rf_tuned_results.csv", index=False)
pd.DataFrame(search.cv_results_).to_csv("results/rf_search_log.csv", index=False)

print()
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")