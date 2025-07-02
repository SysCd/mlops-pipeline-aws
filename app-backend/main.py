# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import logging
from fastapi import Request

# Define the same model architecture as in train.py
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()  # put model in inference mode

# Input data format using Pydantic
class InputData(BaseModel):
    feature1: float
    feature2: float

# Init FastAPI app
app = FastAPI()

# Inference endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input to tensor
    x = torch.tensor([[data.feature1, data.feature2]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = int(prob >= 0.5)
    return {"prediction": pred, "probability": float(prob)}

@app.get("/")
def root():
    return {"message": "ML model API is running"}
    
    
    


logging.basicConfig(filename='mlapi.log', level=logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logging.info(f"{request.method} {request.url} - {response.status_code}")
    return response
