import pandas as pd
import os

results_file = "d:/TNBC/results.txt"
# We read the ALREADY automated file to add to it, or strictly overwrite? 
# The user wants "the same", implying the final file should have both.
# Let's read the one we just made.
input_file = "d:/TNBC/annotation_task_automated.xlsx"
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
            filename = os.path.basename(path)
            predictions[filename] = float(score)

print(f"Loaded {len(predictions)} predictions.")

# 2. Read Excel
print(f"Reading {input_file}...")
df = pd.read_excel(input_file)

# 3. Merge for PD1
pd1_scores = []
pd1_labels = []

for idx, row in df.iterrows():
    he_path_excel = row['HE_path']
    filename = os.path.basename(he_path_excel)
    
    if filename in predictions:
        score = predictions[filename]
        pd1_scores.append(score)
        # Assuming SAME cut-off 0.5 
        label = 1 if score > 0.5 else 0
        pd1_labels.append(label)
    else:
        pd1_scores.append(None)
        pd1_labels.append(None)

# Add columns for PD1
df['PD1_Prediction_Score'] = pd1_scores
df['PD1_label'] = pd1_labels 

# 4. Save
df.to_excel(output_file, index=False)
print(f"Updated {output_file} with PD1 labels.")
