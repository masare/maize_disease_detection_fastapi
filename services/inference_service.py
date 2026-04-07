# import time
from PIL import Image

import torch
from fastapi import FastAPI, File, UploadFile


def run_inference(app: FastAPI, file: UploadFile = File(...)):
    # start_time = time.time()
    
    # Read and preprocess the image
    image = Image.open(file.file).convert('RGB')
    input_data = app.state.preprocessor(image).unsqueeze(0)

    # Run inference
    with torch.inference_mode():
        outputs = app.state.model(input_data)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class = torch.max(probabilities, 0)
    
    label = app.state.labels[top_class.item()]
    score = top_prob.item()

    if round(score,1) < 0.5:
        label = "Unknown"

    # inference_time = time.time() - start_time

    return {
        'class': label,
        'score': score,
        # 'inference_time (s)': inference_time,
        # 'model_loading_time (s)': app.state.model_loading_time
    }