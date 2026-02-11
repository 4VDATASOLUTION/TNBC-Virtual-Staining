import pandas as pd
import os
import re

root_dir = "d:/TNBC/"
he_dir = "02-008_HE_A12_v2_s13"
pdl1_dir = "02-008_PDL1(SP142)-Springbio_A12_v3_b3"
pd1_dir = "02-008_PD1(NAT105)-CellMarque_A12_v3_b3"

# specific logic to extract the unique id (suffix) from the filename
# filenames look like: ..._001_r1c1.jpg.jpeg
def get_suffix(filename):
    match = re.search(r'_(\d{3}_r\d+c\d+)\.jpg\.jpeg$', filename)
    if match:
        return match.group(1)
    return None

data = {
    'HE_path': [],
    'PDL1_path': [],
    'PDL1_label': [],
    'PD1_path': [],
    'PD1_label': []
}

# Map suffixes to filenames for each directory
he_files = {}
for f in os.listdir(os.path.join(root_dir, he_dir)):
    suffix = get_suffix(f)
    if suffix:
        he_files[suffix] = f

pdl1_files = {}
for f in os.listdir(os.path.join(root_dir, pdl1_dir)):
    suffix = get_suffix(f)
    if suffix:
        pdl1_files[suffix] = f

pd1_files = {}
for f in os.listdir(os.path.join(root_dir, pd1_dir)):
    suffix = get_suffix(f)
    if suffix:
        pd1_files[suffix] = f

# Create the rows based on HE files (assuming HE is the anchor)
sorted_suffixes = sorted(he_files.keys())

for suffix in sorted_suffixes:
    he_file = he_files[suffix]
    pdl1_file = pdl1_files.get(suffix)
    pd1_file = pd1_files.get(suffix)

    if he_file and pdl1_file and pd1_file:
        data['HE_path'].append(f"{he_dir}/{he_file}")
        data['PDL1_path'].append(f"{pdl1_dir}/{pdl1_file}")
        data['PD1_path'].append(f"{pd1_dir}/{pd1_file}")
        data['PDL1_label'].append(None)
        data['PD1_label'].append(None)

df = pd.DataFrame(data)
output_path = os.path.join(root_dir, "annotation_task_generated.xlsx")
df.to_excel(output_path, index=False)
print(f"Generated {output_path} with {len(df)} rows.")
