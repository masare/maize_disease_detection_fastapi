import pickle
import time
from pathlib import Path

import torch
import torchvision
from fastapi import FastAPI

# Define artifact paths
ARTIFACTS_DIR = Path(__file__).parent.parent / "deployment_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "resnet_model_weights.pth"
LABELS_PATH = ARTIFACTS_DIR / "class_names.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


def load_artifacts(app: FastAPI):
    
    # Load the labels first to get the number of classes
    with open(LABELS_PATH, "rb") as f:
        app.state.labels = pickle.load(f)
    
    num_classes = len(app.state.labels)
    
    # Load the model
    start_time = time.time()

    model = torchvision.models.resnet18(weights=None)
    # Modify the final fully connected layer to match the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    app.state.model = model

    # Load the preprocessor
    with open(PREPROCESSOR_PATH, "rb") as f:
        app.state.preprocessor = pickle.load(f)

    app.state.model_loading_time = time.time() - start_time
    yield

def cleanup(app):
    # clean up resources on shutdown if necessary
    del app.state.model
    del app.state.preprocessor
