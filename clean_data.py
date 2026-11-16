import pandas as pd

df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
print("Before:", df.shape)

df = df.drop_duplicates()
print("After:", df.shape)

df.to_csv("data/cleaned.csv", index=False)
print("Saved data/cleaned.csv")