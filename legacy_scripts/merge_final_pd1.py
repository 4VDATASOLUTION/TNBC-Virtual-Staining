import pandas as pd
import os

results_file = "d:/TNBC/results_pd1.txt"
input_file = "d:/TNBC/annotation_task_automated.xlsx"
output_file = "d:/TNBC/annotation_task_automated.xlsx"

# 1. Read New Predictions
predictions = {}
with open(results_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if ": " in line:
            path, score = line.split(": ")
            filename = os.path.basename(path)
            predictions[filename] = float(score)

# 2. Read Excel
df = pd.read_excel(input_file)

# 3. Merge for PD1
pd1_scores = []
pd1_labels = []

# Update the columns
for idx, row in df.iterrows():
    he_path_excel = row['HE_path']
    filename = os.path.basename(he_path_excel)
    
    if filename in predictions:
        score = predictions[filename]
        pd1_scores.append(score)
        label = 1 if score > 0.5 else 0
        pd1_labels.append(label)
    else:
        # Keep existing or set to None
        pd1_scores.append(None)
        pd1_labels.append(None)

df['PD1_Prediction_Score'] = pd1_scores
df['PD1_label'] = pd1_labels 

# 4. Save
df.to_excel(output_file, index=False)
print(f"Updated {output_file} with REAL PD1 model predictions.")
