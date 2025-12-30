# test_mil.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_mil import MILDataset, AttentionMIL  # import from your training script

# ----------------- Settings -----------------
BAG_SIZE = 16
DATA_DIR = "data"

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Dataset & DataLoader
test_dataset = MILDataset(DATA_DIR, bag_size=BAG_SIZE, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = AttentionMIL().to(device)
model.load_state_dict(torch.load("mil_all_classifier.pth", map_location=device))
model.eval()

# ----------------- Testing -----------------
with torch.no_grad():
    for i, (bags, labels) in enumerate(test_loader):
        bags = bags.to(device)
        outputs, attn = model(bags)
        preds = (outputs.squeeze() > 0.5).float()
        label_str = "ALL" if preds.item() == 1 else "NORMAL"
        print(f"Bag {i+1} → Prediction: {label_str} ({outputs.item():.4f})")
py