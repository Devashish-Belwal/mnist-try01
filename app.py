import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import numpy as np

class ScaledTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh((2.0/3.0) * x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 32x32 -> 28x28x6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14x6 -> 10x10x16

        # Fully connected layers
        self.fc1 = nn.Linear(16*4*4, 120)  # after pool2
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 10)      # output layer (digits 0–9)

        # Activation
        self.act = ScaledTanh()

    def forward(self, x):
        # Conv1 + Pool1
        x = self.act(self.conv1(x))        # [batch, 6, 28, 28]
        x = F.max_pool2d(x, 2)             # [batch, 6, 14, 14]

        # Conv2 + Pool2
        x = self.act(self.conv2(x))        # [batch, 16, 10, 10]
        x = F.max_pool2d(x, 2)             # [batch, 16, 5, 5]

        # Flatten
        x = x.view(-1, 16*4*4)             # [batch, 400]

        # Fully connected layers
        x = self.act(self.fc1(x))          # [batch, 120]
        x = self.act(self.fc2(x))          # [batch, 100]
        x = self.fc3(x)                    # [batch, 10]

        return x

@st.cache_resource
def load_cnn_model():
    cnn_model = CNN()
    cnn_model.load_state_dict(torch.load("final/cnn/cnn_model_state.pth", map_location="cpu"))
    cnn_model.eval()
    return cnn_model

cnn_model = load_cnn_model()

class FullyConnectedModel(nn.Module):
    def __init__(self):
        super(FullyConnectedModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 300)

        # '''TODO: Define the activation function for the first fully connected layer'''
        self.relu = nn.ReLU()

        # '''TODO: Define the second Linear layer to output the classification probabilities'''
        self.fc2 = nn.Linear(300, 100)
        # self.fc2 = # TODO

        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)

        # '''TODO: Implement the rest of forward pass of the model using the layers you have defined above'''
        x = self.relu(x)
        x = self.fc2(x)
        # '''TODO'''

        x = self.relu(x)
        x = self.fc3(x)
        return x

@st.cache_resource
def load_fc_model():
    fc_model = FullyConnectedModel()
    fc_model.load_state_dict(torch.load("final/fc/fc_model_state.pth", map_location="cpu"))
    fc_model.eval()
    return fc_model

fc_model = load_fc_model()

# --- Title ---
st.title("✏️ MNIST Digit Classifier")
st.write("Draw a digit below and I'll predict it!")

# --- Drawing Canvas ---
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Original canvas as NumPy array
    img_array = canvas_result.image_data[:, :, 0]  # just take one channel for display

    # Convert to PIL, invert, resize to 28x28
    img_pil = Image.fromarray((255 - img_array).astype(np.uint8))
    img_resized = img_pil.resize((28, 28))

    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array, caption="Your Drawing", width=150)
    with col2:
        st.image(img_resized, caption="28x28 Image Seen by Model", width=150)

    # Preprocess for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img_resized).unsqueeze(0)

# --- Fully Connected Model ---
with torch.no_grad():
    output = fc_model(img_tensor)
    probs = F.softmax(output, dim=1)
    top_probs, top_classes = probs.topk(3, dim=1)

# --- CNN Model ---
with torch.no_grad():
    output = cnn_model(img_tensor)
    probs = F.softmax(output, dim=1)
    top_probs, top_classes = probs.topk(3, dim=1)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Fully Connected Model Predictions")
    for i in range(3):
        st.write(f"{i+1}. Digit {top_classes[0][i].item()} — {top_probs[0][i].item()*100:.2f}%")
with col2:
    st.subheader("CNN Model Predictions")
    for i in range(3):
        st.write(f"{i+1}. Digit {top_classes[0][i].item()} — {top_probs[0][i].item()*100:.2f}%")
