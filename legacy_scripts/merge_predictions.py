import pandas as pd
import os

results_file = "d:/TNBC/results.txt"
excel_file = "d:/TNBC/annotation_task_generated.xlsx"
output_file = "d:/TNBC/annotation_task_automated.xlsx"

# 1. Read Predictions
predictions = {}
print(f"Reading {results_file}...")
with open(results_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if ": " in line:
            path, score = line.split(": ")
            # Standardize path slashes and extract just the filename to be safe
            filename = os.path.basename(path)
            predictions[filename] = float(score)

print(f"Loaded {len(predictions)} predictions.")

# 2. Read Excel
print(f"Reading {excel_file}...")
df = pd.read_excel(excel_file)

# 3. Merge
# We need to match rows. The 'HE_path' in excel looks like "02-008_HE.../file.jpeg"
# We'll rely on the basename match.

pdl1_scores = []
pdl1_labels = []

for idx, row in df.iterrows():
    he_path_excel = row['HE_path']
    filename = os.path.basename(he_path_excel)
    
    if filename in predictions:
        score = predictions[filename]
        pdl1_scores.append(score)
        # Assuming cut-off 0.5 for Positive(1) / Negative(0)
        label = 1 if score > 0.5 else 0
        pdl1_labels.append(label)
    else:
        pdl1_scores.append(None)
        pdl1_labels.append(None)
        print(f"Warning: No prediction found for {filename}")

# Add columns
df['PDL1_Prediction_Score'] = pdl1_scores
df['PDL1_label'] = pdl1_labels # Overwrite the label column with AI prediction

# 4. Save
df.to_excel(output_file, index=False)
print(f"Saved merged data to {output_file}")
