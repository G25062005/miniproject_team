# app.py
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from train_mil import AttentionMIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Settings ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mil_all_classifier.pth"

# Load model
model = AttentionMIL().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="ALL Detection", layout="wide")
st.title("🧬 Acute Lymphoblastic Leukemia (ALL) Detection")

# Sidebar
st.sidebar.title("Instructions")
st.sidebar.info(
    "Upload one or more PNG/JPG images of blood smear slides.\n"
    "The model will predict whether the bag contains ALL cells or Normal cells "
    "and display attention heatmaps per image."
)

uploaded_files = st.file_uploader(
    "Upload images (PNG/JPG) for analysis",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

if uploaded_files:
    bag_size = len(uploaded_files)
    st.write(f"Number of images in bag: {bag_size}")

    imgs = []
    original_imgs = []
    img_names = []

    for img_file in uploaded_files:
        img = Image.open(img_file).convert("RGB")
        original_imgs.append(img)
        imgs.append(transform(img))
        img_names.append(img_file.name)

    bag_tensor = torch.stack(imgs).unsqueeze(0).to(DEVICE)  # (1, N, C, H, W)

    with torch.no_grad():
        output, attn = model(bag_tensor)
        pred = (output.squeeze() > 0.5).float()
        bag_label = "ALL" if pred.item() == 1 else "NORMAL"

    st.markdown(f"### ✅ Bag Prediction: {bag_label} ({output.item():.4f})")

    # --- Process attention safely ---
    attn_scores = attn.cpu().detach().numpy()

    # Force attn_scores to be 1D array
    attn_scores = np.atleast_1d(attn_scores).flatten()

    # Normalize
    attn_scores = attn_scores / attn_scores.max()

    # --- Table summary ---
    summary_data = []
    for idx, score in enumerate(attn_scores):
        img_pred = "ALL" if score > 0.5 else "NORMAL"
        summary_data.append({
            "Image": img_names[idx],
            "Attention Score": round(float(score), 3),
            "Prediction": img_pred
        })

    st.markdown("### 📊 Image-wise Summary")
    st.table(pd.DataFrame(summary_data))

    # --- Progress bars ---
    st.markdown("### 🔄 Attention Progress per Image")
    for idx, score in enumerate(attn_scores):
        st.write(f"**{img_names[idx]}**")
        st.progress(float(score))

    # --- Attention heatmaps ---
    st.markdown("### 🔍 Attention Heatmaps")
    for idx, img in enumerate(original_imgs):
        fig, ax = plt.subplots()
        ax.imshow(img)

        # Heatmap overlay: red intensity proportional to attention
        heatmap = np.zeros((img.size[1], img.size[0], 3))
        heat_intensity = float(attn_scores[idx])
        heatmap[..., 0] = heat_intensity  # red channel
        ax.imshow(heatmap, alpha=0.5)
        ax.axis('off')
        st.pyplot(fig)
