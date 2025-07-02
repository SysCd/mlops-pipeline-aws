# src/train.py

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import os
import mlflow
import mlflow.pytorch

# Load data
df = pd.read_csv("data/sample.csv")

# Extract features and target
X = df[["feature1", "feature2"]].values
y = df["target"].values.reshape(-1, 1)  # reshape for BCEWithLogitsLoss

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Wrap in DataLoader
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)

# Define model
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

model = SimpleNN()

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # expects raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(20):
    for xb, yb in train_dl:
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
print("Model saved to models/model.pth")




# SET TRACKING URI FIRST!
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Default")


with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 2)
    mlflow.log_param("epochs", 20)
    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.sigmoid(logits).round()
        acc = (preds.numpy() == y_test).mean()
    mlflow.log_metric("accuracy", acc)
    
    # Save and log the model as an artifact (LOCAL ONLY)
    mlflow.pytorch.save_model(model, "models/pytorch-model")   # Save model locally
    mlflow.log_artifacts("models/pytorch-model")               # Log whole directory as artifacts

