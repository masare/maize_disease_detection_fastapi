import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

from PIL import Image

import torch
import torchvision
from fastapi import FastAPI, File, UploadFile, status

# Define artifact paths
ARTIFACTS_DIR = Path(__file__).parent / "deployment_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "resnet_model_weights.pth"
LABELS_PATH = ARTIFACTS_DIR / "class_names.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

LABELS = {}
model_loading_time = 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    global LABELS, model_loading_time
    
    # Load the labels first to get the number of classes
    with open(LABELS_PATH, "rb") as f:
        LABELS = pickle.load(f)
    
    num_classes = len(LABELS)
    
    # Load the model
    model_loading_time = time.time()
    app.state.model = torchvision.models.resnet18(weights=None)
    # Modify the final fully connected layer to match the number of classes
    app.state.model.fc = torch.nn.Linear(app.state.model.fc.in_features, num_classes)
    
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    app.state.model.load_state_dict(state_dict)
    app.state.model.eval()  # Set the model to evaluation mode

    # Load the preprocessor
    with open(PREPROCESSOR_PATH, "rb") as f:
        app.state.preprocessor = pickle.load(f)

    model_loading_time = time.time() - model_loading_time
    yield

    # clean up resources on shutdown if necessary
    del app.state.model
    del app.state.preprocessor

app = FastAPI(lifespan=lifespan, title="Maize Disease Detection API", description="API for detecting diseases in maize leaves using a pre-trained ResNet model.", version="1.0.0")

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return "Maize Disease Detection"


@app.get("/model", status_code=status.HTTP_200_OK)
def model_imfo():
    return f"Model loaded in {model_loading_time}"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Read and preprocess the image
    image = Image.open(file.file).convert('RGB')
    input_data = app.state.preprocessor(image)
    input_data = input_data.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = app.state.model(input_data)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class = torch.max(probabilities, 0)
    label = LABELS[top_class.item()]
    score = top_prob.item()

    if top_prob < 0.5:
        label = "Unknown"

    inference_time = time.time() - start_time

    return {
        'class': label,
        'score': score,
        'inference_time (s)': inference_time,
        'model_loading_time (s)': model_loading_time
    }