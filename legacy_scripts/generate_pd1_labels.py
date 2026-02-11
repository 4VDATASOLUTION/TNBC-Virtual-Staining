import cv2
import numpy as np
import os
import pandas as pd

# Paths
pd1_dir = "d:/TNBC/02-008_PD1(NAT105)-CellMarque_A12_v3_b3"
output_file = "d:/TNBC/pd1_ground_truth.csv"

def quantify_brown_stain(image_path):
    img = cv2.imread(image_path)
    if img is None: return 0
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Updated ranges based on debug analysis
    # Range 1: Red-Yellow/Brown
    lower1 = np.array([0, 20, 50])
    upper1 = np.array([25, 255, 200])
    
    # Range 2: High end red/purple
    lower2 = np.array([160, 20, 50])
    upper2 = np.array([180, 255, 200])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate percentage of pixels that are brown
    brown_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    
    return brown_pixels / total_pixels

print("Generating Ground Truth labels from PD-1 images...")
data = []

files = os.listdir(pd1_dir)
for f in files:
    if not f.endswith('.jpeg') and not f.endswith('.jpg'): continue
    
    path = os.path.join(pd1_dir, f)
    score = quantify_brown_stain(path)
    
    # Threshold: if more than 1% of the tissue is brown, call it positive
    # This is a heuristic; user might need to adjust
    label = 1 if score > 0.001 else 0 
    
    data.append({'filename': f, 'stain_score': score, 'label': label})

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"Saved generated labels to {output_file}")
print(df['label'].value_counts())
