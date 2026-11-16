import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MIN_ROWS = 10
SEED = 42
label = "diseases"

df = pd.read_csv("data/cleaned.csv")
print("Start:", df.shape)

# 1. Drop diseases with fewer than MIN_ROWS rows
counts = df[label].value_counts()
keep = counts[counts >= MIN_ROWS].index
df = df[df[label].isin(keep)]
print("After cutoff:", df.shape, "| diseases:", df[label].nunique())

# 2. Encode disease names as numbers
le = LabelEncoder()
df["label_id"] = le.fit_transform(df[label])

# 3. Stratified 80/20 split
train, test = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
)
print("Train:", train.shape, "| Test:", test.shape)

# 4. Save the split and the label mapping
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
pd.DataFrame({"label_id": range(len(le.classes_)), "disease": le.classes_}) \
  .to_csv("data/label_map.csv", index=False)
print("Saved train.csv, test.csv, label_map.csv in data/")