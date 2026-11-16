import pandas as pd

df = pd.read_csv("data/cleaned.csv")
label = "diseases"

counts = df[label].value_counts()
total_rows = len(df)
total_classes = len(counts)

print(f"Total: {total_classes} diseases, {total_rows} rows\n")
print(f"{'Min rows':>8} | {'Diseases kept':>13} | {'Diseases dropped':>16} | {'Rows lost':>9} | {'Rows lost %':>11}")
print("-" * 70)

for cutoff in [2, 5, 10, 20, 50]:
    kept = counts[counts >= cutoff]
    dropped_classes = total_classes - len(kept)
    rows_lost = total_rows - kept.sum()
    pct = rows_lost / total_rows * 100
    print(f"{cutoff:>8} | {len(kept):>13} | {dropped_classes:>16} | {rows_lost:>9} | {pct:>10.2f}%")