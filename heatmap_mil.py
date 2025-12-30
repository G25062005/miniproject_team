import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# -------------------------
# SAME MODEL AS TRAINING
# -------------------------
class AttentionMIL(nn.Module):
    def __init__(self):
        super(AttentionMIL, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*64*64, 256),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, N, -1)

        attn_scores = self.attention(features)
        attn_weights = torch.softmax(attn_scores, dim=1)

        bag_feature = torch.sum(attn_weights * features, dim=1)
        output = self.classifier(bag_feature)

        return torch.sigmoid(output), attn_weights.squeeze(-1)

# -------------------------
# LOAD MODEL
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionMIL().to(device)

model.load_state_dict(torch.load("mil_all_classifier.pth", map_location=device))
model.eval()

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# -------------------------
# PICK ANY IMAGE FROM TEST SET
# -------------------------
image_paths = glob.glob("data/ALL/*.png")[:1]  # take first ALL image
img_path = image_paths[0]

# -------------------------
# CREATE PATCH BAG
# -------------------------
def make_bag(img_path, bag_size=20):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((512, 512))   # big image
    patches = []

    step = 128
    for i in range(0, 512, step):
        for j in range(0, 512, step):
            crop = img.crop((j, i, j+128, i+128))
            crop = crop.resize((64, 64))
            patches.append(transform(crop))

    patches = torch.stack(patches)
    patches = patches.unsqueeze(0)
    return patches, img

bag, original_img = make_bag(img_path)
bag = bag.to(device)

# -------------------------
# INFERENCE
# -------------------------
with torch.no_grad():
    pred, attn = model(bag)

pred_label = "ALL" if pred.item() > 0.5 else "NORMAL"
print("Prediction:", pred_label)
print("Confidence:", float(pred.item()))
print("Attention shape:", attn.shape)

# -------------------------
# CREATE HEATMAP
# -------------------------
heatmap = attn.view(4, 4).cpu().numpy()
heatmap = heatmap / heatmap.max()

plt.figure(figsize=(6,6))
plt.imshow(original_img)
plt.imshow(heatmap, cmap="jet", alpha=0.4)
plt.title("Attention Heatmap")
plt.axis("off")
plt.show()
