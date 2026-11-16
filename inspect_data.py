import pandas as pd

df = pd.read_csv("data/cleaned.csv")
label = df.columns[0]          # assumes disease is the left column
X = df.drop(columns=[label])

print("1. Shape (rows, cols):", df.shape)
print("2. Unique diseases:", df[label].nunique())

counts = df[label].value_counts()
print("3. Smallest class:", counts.min(), "| Largest class:", counts.max())

print("4. Duplicate rows:", df.duplicated().sum())

values = pd.unique(X.values.ravel())
print("5. Unique values in symptom columns:", sorted(values)[:10])

print("Missing values:", df.isna().sum().sum())
print("Label column name:", label)