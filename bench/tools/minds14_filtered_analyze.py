import pandas as pd

df = pd.read_csv("data/minds14_banking_raw/metadata.csv")

short_df = df[df["duration"] <= 6.0]

total_files = len(df)
selected_files = len(short_df)
selected_pct = (selected_files / total_files) * 100

print(f"Total files: {total_files}")
print(f"Selected files (<=6s): {selected_files}")
print(f"Not selected (>6s): {total_files - selected_files}")
print(f"Selected share: {selected_pct:.2f}%")

counts = pd.crosstab(
    short_df["language_variety"],
    short_df["intent"]
)

print("\nLanguage × intent counts for <=6s:")
print(counts)

print("\nMinimum samples in any language × intent cell:", counts.min().min())
print("Zero-count cells:", (counts == 0).sum().sum())
print("Cells with <10 samples:", (counts < 10).sum().sum())
print("Cells with <15 samples:", (counts < 15).sum().sum())
print("Cells with <20 samples:", (counts < 20).sum().sum())