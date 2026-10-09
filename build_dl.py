import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from torch.utils.data import DataLoader, TensorDataset

from encode_house_price_linear import FREQ_COLS, NUM_COLS, OHE_COLS, ORDINAL_COLS, ORDINAL_MAPS

class MLP(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(params["n_layers"]):
            layers.extend([
                nn.Linear(in_dim, params["hidden"]),
                nn.BatchNorm1d(params["hidden"]),
                _activation(params["activation"]),
                nn.Dropout(params["dropout"]),
            ])
            in_dim = params["hidden"]
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _activation(name):
    if name == "relu":
        return nn.ReLU()
    return nn.GELU()


def house_preprocessor():
    ordinal_categories = [list(ORDINAL_MAPS[col]) for col in ORDINAL_COLS]
    return ColumnTransformer(
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


def titanic_numeric(X):
    data = X.copy()
    text_cols = data.select_dtypes(include=["category", "object", "string", "bool"]).columns
    for col in text_cols:
        data[col] = data[col].astype("float32")
    return data.astype("float32")


def _tensors(X_num, y):
    values = X_num.to_numpy() if hasattr(X_num, "to_numpy") else X_num
    targets = y.to_numpy() if hasattr(y, "to_numpy") else y
    X_tensor = torch.tensor(values, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor


def split_monitor(X, y, config):
    options = {
        "test_size": config.DL_MONITOR_FRACTION,
        "random_state": config.RANDOM_STATE,
    }
    if config.DATASET != "house_prices":
        options["stratify"] = y
    return train_test_split(X, y, **options)


def fit_mlp(X_tensor, y_tensor, X_monitor, y_monitor, params, config):
    torch.manual_seed(config.RANDOM_STATE)
    model = MLP(X_tensor.shape[1], params)
    if params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=config.DL_SGD_MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.DL_LR_FACTOR,
        patience=config.DL_LR_PATIENCE,
        threshold=config.DL_MIN_DELTA,
    )

    house = config.DATASET == "house_prices"
    criterion = nn.MSELoss() if house else nn.BCEWithLogitsLoss()
    batch_size = min(params["batch_size"], max(2, len(X_tensor) // 2))
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(X_tensor) > batch_size,
    )

    best_loss = float("inf")
    best_state = None
    wait = 0
    for _ in range(params["epochs"]):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            monitor_loss = criterion(model(X_monitor), y_monitor).item()
        scheduler.step(monitor_loss)
        if monitor_loss + config.DL_MIN_DELTA < best_loss:
            best_loss = monitor_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.DL_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def score_mlp(model, X_tensor, y_tensor, config):
    house = config.DATASET == "house_prices"
    model.eval()
    criterion = nn.MSELoss() if house else nn.BCEWithLogitsLoss()
    with torch.no_grad():
        preds = model(X_tensor)
        if house:
            return torch.sqrt(criterion(preds, y_tensor)).item()
        labels = (torch.sigmoid(preds) >= config.DL_THRESHOLD).int()
        return (labels == y_tensor.int()).float().mean().item()

def train_dl_model(X, y, config):
    X_num = titanic_numeric(X)
    X_tensor = torch.tensor(X_num.to_numpy(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    params = dict(config.DL_DEFAULT)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model = MLP(input_dim=X.shape[1], params=params)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(params["epochs"]):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        preds = (torch.sigmoid(logits) >= config.DL_THRESHOLD).int()
        accuracy = (preds == y_tensor.int()).float().mean().item()

    print(f"DL Model Train Accuracy: {accuracy:.4f}")
    return model

def train_house_model(X, y, config, params=None):
    if params is None:
        params = dict(config.DL_DEFAULT)
    X_fit, X_monitor, y_fit, y_monitor = split_monitor(X, y, config)
    preprocessor = house_preprocessor()
    X_fit_num = preprocessor.fit_transform(X_fit, y_fit)
    X_monitor_num = preprocessor.transform(X_monitor)
    X_tensor, y_tensor = _tensors(X_fit_num, y_fit)
    X_monitor_tensor, y_monitor_tensor = _tensors(X_monitor_num, y_monitor)
    model = fit_mlp(X_tensor, y_tensor, X_monitor_tensor, y_monitor_tensor, params, config)
    rmse = score_mlp(model, X_tensor, y_tensor, config)
    print(f"DL Model Train RMSE: {rmse:.4f}")
    model.preprocessor = preprocessor
    return model