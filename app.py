
import os
import urllib.request
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

MODEL_PATH = "resnet_cifar10.pth"
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1rC0SB8e9an_na10fltMCJJXKmeg7nABe"  # Replace this ID

# -- Download model from GDrive if not present --
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
        urllib.request.urlretrieve(GDRIVE_URL, MODEL_PATH)
        st.success("Model downloaded!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('resnet_cifar10.pth', map_location=device))
model.eval().to(device)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier 🧠📸")
uploaded_file = st.file_uploader("Upload a 32x32 CIFAR-like image (jpg/png)...", type=['jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    st.markdown(f"### 🧠 Prediction: **{predicted_class.upper()}**")
