import pandas as pd

print("=== annotation_task_automated.xlsx ===")
df = pd.read_excel('data_raw/annotation_task_automated.xlsx')
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()
for c in df.columns:
    print(f"  {c}: dtype={df[c].dtype}, nulls={df[c].isna().sum()}")
print()
print("First 3 rows:")
for i in range(3):
    print(f"\n--- Row {i} ---")
    for c in df.columns:
        print(f"  {c}: {df[c].iloc[i]}")

print()
if 'PDL1_label' in df.columns:
    print(f"PDL1_label distribution: {df['PDL1_label'].value_counts().to_dict()}")
if 'PD1_label' in df.columns:
    print(f"PD1_label distribution: {df['PD1_label'].value_counts().to_dict()}")
if 'PDL1_Prediction_Score' in df.columns:
    scores = df['PDL1_Prediction_Score'].dropna()
    print(f"PDL1_Prediction_Score: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

print("\n\n=== annotation_task_generated.xlsx ===")
df2 = pd.read_excel('data_raw/annotation_task_generated.xlsx')
print(f"Shape: {df2.shape}")
print(f"Columns: {list(df2.columns)}")
print()
for c in df2.columns:
    print(f"  {c}: dtype={df2[c].dtype}, nulls={df2[c].isna().sum()}")
print()
print("First 3 rows:")
for i in range(3):
    print(f"\n--- Row {i} ---")
    for c in df2.columns:
        print(f"  {c}: {df2[c].iloc[i]}")
