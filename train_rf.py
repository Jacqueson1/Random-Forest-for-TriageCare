import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

SEED = 42
N_TREES = 100

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Features = symptom columns only. Drop both label columns.
X_train = train.drop(columns=["diseases", "label_id"])
y_train = train["label_id"]
X_test = test.drop(columns=["diseases", "label_id"])
y_test = test["label_id"]

print("Features:", X_train.shape[1], "| Train:", len(X_train), "| Test:", len(X_test))

model = RandomForestClassifier(
    n_estimators=N_TREES, n_jobs=-1, random_state=SEED
)

# Training time
t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0
print(f"Training done in {train_time:.1f}s")

# Prediction time
t0 = time.time()
pred = model.predict(X_test)
predict_time = time.time() - t0

results = {
    "model": "RandomForest (default)",
    "accuracy": accuracy_score(y_test, pred),
    "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
    "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    "train_time_sec": round(train_time, 1),
    "predict_time_sec": round(predict_time, 1),
}

os.makedirs("results", exist_ok=True)
pd.DataFrame([results]).to_csv("results/rf_results.csv", index=False)

print()
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")