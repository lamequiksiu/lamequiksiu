import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import gdown
import os
import time

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Lamequi AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Navbar */
.navbar {
    display:flex;
    justify-content:space-between;
    padding:20px;
    color:white;
    font-weight:bold;
    font-size:20px;
}

/* Title */
.title {
    text-align:center;
    font-size:65px;
    font-weight:bold;
    color:white;
}
.subtitle {
    text-align:center;
    color:#ccc;
    font-size:22px;
    margin-bottom:30px;
}

/* Glass box */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding:30px;
    border-radius:25px;
    border:1px solid rgba(255,255,255,0.1);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color:white;
    border-radius:12px;
    height:3em;
    font-size:18px;
    border:none;
    transition:0.3s;
}
.stButton>button:hover {
    transform:scale(1.07);
}

/* Result card */
.result-card {
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    color:white;
    animation:fadeIn 0.6s ease-in-out;
}

/* Animation */
@keyframes fadeIn {
    from {opacity:0; transform:translateY(20px);}
    to {opacity:1; transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
<div>🐾 Lamequi</div>
<div>AI • X-ray • Vet</div>
</div>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">Lamequi AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Veterinary Diagnosis Platform</div>', unsafe_allow_html=True)

# ---------------- MODEL DOWNLOAD ----------------
model_path = "model.pth"

if not os.path.exists(model_path):
    with st.spinner("Downloading AI model..."):
        url = "https://drive.google.com/uc?export=download&id=1Bz9pbfGQzf0ANGAY3Cy46FIwkxfqXIe9"
        gdown.download(url, model_path, quiet=False)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- MAIN UI ----------------
st.markdown('<div class="glass">', unsafe_allow_html=True)

file = st.file_uploader("📤 Drag & Drop X-ray Image", type=["jpg", "png"])

st.write("")

if file:
    col1, col2 = st.columns([1,1])

    with col1:
        img = Image.open(file).convert("RGB")
        st.image(img, use_column_width=True)

    with col2:
        if st.button("🚀 Analyze Now"):
            
            with st.spinner("🧠 AI is analyzing the X-ray..."):
                time.sleep(1)

                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)

                img_t = transform(img).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_t)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred = torch.max(probs, 1)

                classes = ["🦴 Fracture", "✅ Normal"]
                result = classes[pred.item()]
                conf = confidence.item() * 100

                color = "#ff4b4b" if pred.item() == 0 else "#00c853"

                st.markdown(f"""
                <div class="result-card" style="background:{color}">
                    {result}<br><br>
                    Confidence: {conf:.2f}%
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.progress(conf / 100)
                st.metric("Confidence Score", f"{conf:.2f}%")

                # 🔥 Diagnosis Details
                st.markdown("### 🔍 Diagnosis Details")

                if pred.item() == 0:
                    st.error("Possible fracture detected. Immediate veterinary consultation recommended.")
                else:
                    st.success("No fracture detected. The bone structure appears normal.")

else:
    st.info("Upload an image to start AI analysis")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr style="border:0.5px solid #444;">
<p style='text-align:center; color:gray; font-size:14px;'>
© 2026 Lamequi AI • Advanced Veterinary Intelligence
</p>
""", unsafe_allow_html=True)
