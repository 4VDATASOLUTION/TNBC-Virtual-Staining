import cv2
import numpy as np
import os

pd1_dir = "d:/TNBC/02-008_PD1(NAT105)-CellMarque_A12_v3_b3"
# Pick a random image or the first one
files = [f for f in os.listdir(pd1_dir) if f.endswith('.jpeg')]
img_path = os.path.join(pd1_dir, files[0])

img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(f"Analyzing {files[0]}")
print("Mean HSV:", np.mean(hsv, axis=(0,1)))
print("Max HSV:", np.max(hsv, axis=(0,1)))
print("Min HSV:", np.min(hsv, axis=(0,1)))

# Check specific center pixel or something
center_pixel = hsv[hsv.shape[0]//2, hsv.shape[1]//2]
print("Center Pixel HSV:", center_pixel)

# Let's try to find ANY brown
# Standard OpenCv HSV ranges: H: 0-179, S: 0-255, V: 0-255
# Brown is usually around H=10-20.
