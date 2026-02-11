import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import cv2
import glob
from torchvision import transforms, models
import time

# Configuration
HE_DIR = "d:/TNBC/02-008_HE_A12_v2_s13"
LABEL_FILE = "d:/TNBC/pd1_ground_truth.csv"
MODEL_SAVE_PATH = "d:/TNBC/pd1_trained_model.pt"
EXISTING_MODEL = "d:/TNBC/PD-L1_predictor/models/trained_model.pt"
EPOCHS = 5
BATCH_SIZE = 8
LR = 0.001

# Image transforms (normalization matched to original script)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.9357, 0.8253, 0.8998), (0.0787, 0.1751, 0.1125)),
])

class HEDataset(Dataset):
    def __init__(self, he_dir, label_file, transform=None):
        self.he_dir = he_dir
        self.labels_df = pd.read_csv(label_file)
        self.transform = transform
        
        # Create a map of suffix -> label to match files
        # Label file has filenames like "02-008_PD1..._r1c1.jpg.jpeg"
        # We need to extract the suffix "_r1c1.jpg.jpeg" or similar unique ID
        self.data = []
        
        # Files in HE directory
        he_files = glob.glob(os.path.join(he_dir, "*.jpeg"))
        
        for he_path in he_files:
            he_name = os.path.basename(he_path)
            # Alignment logic: both files end with ..._001_r1c1.jpg.jpeg
            # We split by '_' and grab the last part "r1c1.jpg.jpeg" roughly
            # Easier: extract rXcY id.
            import re
            match = re.search(r'_(r\d+c\d+)\.jpg\.jpeg', he_name)
            if match:
                roi_id = match.group(1)
                # Find corresponding label
                # Label filenames: "02-008_PD1..._001_r1c1.jpg.jpeg"
                label_row = self.labels_df[self.labels_df['filename'].str.contains(roi_id, regex=False)]
                if not label_row.empty:
                    label = label_row.iloc[0]['label']
                    self.data.append((he_path, label))

        print(f"Matched {len(self.data)} images for training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 512))
        img = img / 255.0
        
        if self.transform:
            img = self.transform(img).float()
        else:
            img = torch.tensor(img).permute(2, 0, 1).float()
            
        return img, torch.tensor(label).long()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Dataset
    dataset = HEDataset(HE_DIR, LABEL_FILE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model - Load existing ResNet structure
    # We'll use a standard ResNet18 for simplicity or try to load the heavy custom one
    # To save time/compatibility let's just use torchvision ResNet18 and transfer weights if possible
    # or just train from ImageNet which is often good enough for this demo
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary classification
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    print("Starting training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}, Acc: {100 * correct / total:.2f}%")

    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
