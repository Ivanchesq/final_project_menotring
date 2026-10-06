import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_dl_model(X, y, config):
    X_num = X.copy()
    for col in X_num.select_dtypes(include=["category", "object"]).columns:
        X_num[col] = X_num[col].astype("float32")
    X_num = X_num.astype("float32")

    X_tensor = torch.tensor(X_num.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TitanicMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(25):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        preds = (torch.sigmoid(logits) >= 0.5).int()
        accuracy = (preds == y_tensor.int()).float().mean().item()

    print(f"DL Model Train Accuracy: {accuracy:.4f}")
    return model