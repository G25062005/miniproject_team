# train_mil.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random

# ---------------- Dataset ----------------
class MILDataset(Dataset):
    def __init__(self, root_dir, bag_size=20, transform=None):
        self.root_dir = root_dir
        self.bag_size = bag_size
        self.transform = transform

        self.all_imgs = glob.glob(os.path.join(root_dir, "ALL", "*.png"))
        self.normal_imgs = glob.glob(os.path.join(root_dir, "NORMAL", "*.png"))

    def __len__(self):
        return 400  # number of bags

    def __getitem__(self, index):
        label = random.choice([0, 1])
        folder = "ALL" if label == 1 else "NORMAL"
        img_paths = glob.glob(os.path.join(self.root_dir, folder, "*.png"))

        # repeat images if fewer than bag_size
        if len(img_paths) < self.bag_size:
            repeats = (self.bag_size // len(img_paths)) + 1
            img_paths = (img_paths * repeats)[:self.bag_size]
        else:
            img_paths = random.sample(img_paths, self.bag_size)

        bag = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            bag.append(img)

        bag = torch.stack(bag)  # shape: (N, C, H, W)
        return bag, torch.tensor(label, dtype=torch.float32)

# ---------------- MIL Model ----------------
class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=128):
        super(AttentionMIL, self).__init__()

        # CNN backbone for features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Classifier
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # merge batch & bag

        # Extract features
        features = self.feature_extractor(x)  # shape: (B*N, feature_dim)
        features = features.view(B, N, -1)   # reshape to (B, N, feature_dim)

        # Attention
        attn_scores = self.attention(features)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Bag representation
        bag_feature = torch.sum(attn_weights * features, dim=1)
        output = self.classifier(bag_feature)

        return torch.sigmoid(output), attn_weights.squeeze(-1)

# ---------------- Train ----------------
def train():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = MILDataset("data", bag_size=20, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionMIL().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0.0
        correct = 0
        total = 0

        for bags, labels in loader:
            bags = bags.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, attn = model(bags)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        acc = (correct / total) * 100
        print(f"Epoch {epoch+1} | Loss = {total_loss:.4f} | Accuracy = {acc:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "mil_all_classifier.pth")
    print("Model saved to mil_all_classifier.pth")


if __name__ == "__main__":
    train()
