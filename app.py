import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🐶 Veterinary X-ray AI")

file = st.file_uploader("Upload image", type=["jpg","png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        _, pred = torch.max(output, 1)

    classes = ["fracture", "normal"]
    st.write("Prediction:", classes[pred.item()])