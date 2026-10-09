import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from torch.utils.data import DataLoader, TensorDataset

from encode_house_price_linear import FREQ_COLS, NUM_COLS, OHE_COLS, ORDINAL_COLS, ORDINAL_MAPS

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


class House(nn.Module):
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
            nn.Linear(32, 1),
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
    for epoch in range(config.EPOCH):
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

def train_house_model(X, y, config):
    ordinal_categories = [list(ORDINAL_MAPS[col]) for col in ORDINAL_COLS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), OHE_COLS),
            ("neighborhood", TargetEncoder(target_type="continuous"), FREQ_COLS),
            ("ord", OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ORDINAL_COLS),
        ],
        remainder="passthrough",
    )
    X_num = preprocessor.fit_transform(X, y)

    X_tensor = torch.tensor(X_num, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = House(input_dim=X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(config.EPOCH):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        rmse = torch.sqrt(criterion(preds, y_tensor)).item()

    print(f"DL Model Train RMSE: {rmse:.4f}")
    model.preprocessor = preprocessor
    return model