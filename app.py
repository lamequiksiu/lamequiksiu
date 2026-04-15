import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import gdown
import os

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Lamequi",
    page_icon="🐾",
    layout="wide"
)

# ------------------ STYLE ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3, p {
    color: white;
    text-align: center;
}
.stButton>button {
    background-color: #4A90E2;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stFileUploader {
    background-color: #262730;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ DOWNLOAD MODEL ------------------
model_path = "model.pth"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?export=download&id=1Bz9pbfGQzf0ANGAY3Cy46FIwkxfqXIe9"
    gdown.download(url, model_path, quiet=False)

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------ HEADER ------------------
st.markdown("<h1>🐾 Lamequi</h1>", unsafe_allow_html=True)
st.markdown("<p>AI Veterinary X-ray Analyzer</p>", unsafe_allow_html=True)

st.write("---")

# ------------------ LAYOUT ------------------
col1, col2 = st.columns(2)

# -------- LEFT --------
with col1:
    st.markdown("### 📤 Upload X-ray")
    file = st.file_uploader("Upload image", type=["jpg", "png"])

# -------- RIGHT --------
with col2:
    st.markdown("### 📊 Result")

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_t = transform(img).unsqueeze(0)

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing..."):
                with torch.no_grad():
                    output = model(img_t)
                    _, pred = torch.max(output, 1)

                classes = ["🦴 Fracture", "✅ Normal"]
                result = classes[pred.item()]

                st.success(f"Result: {result}")
    else:
        st.info("Upload an image to see result")