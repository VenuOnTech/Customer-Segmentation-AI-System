import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_torch_churn(rfm, epochs=10):
    X = rfm.select_dtypes(include=["number"]).drop(columns=["Cluster"], errors="ignore")

    if "Churn" not in rfm.columns:
        rfm["Churn"] = (rfm["Frequency"] < 2).astype(int)

    y = rfm["Churn"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = ChurnNet(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("🔥 PyTorch model trained")

    return model, {"loss": float(loss.item())}