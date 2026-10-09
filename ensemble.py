import numpy as np
import joblib
import torch

from config import Config
from build_baseline import _build_baseline_houses, _build_baseline_titanic, _house_regressor
from optimize import fit_boost_model, _encode_frame
from build_dl import titanic_numeric
from optimize_dl import fit_dl_model


class AverageEnsemble:
    def __init__(self, members, dataset):
        self.members = members
        self.dataset = dataset

    def predict(self, X):
        house = self.dataset == "house_prices"
        parts = []
        for member in self.members:
            parts.append(_predict_member(member, X, house))
        return np.mean(parts, axis=0)


def _predict_member(member, X, house):
    model = member["model"]
    kind = member["kind"]
    if kind == "mlp":
        model.eval()
        if house:
            values = model.preprocessor.transform(X)
        else:
            values = titanic_numeric(X).to_numpy()
        with torch.no_grad():
            raw = model(torch.tensor(values, dtype=torch.float32)).numpy().reshape(-1)
        if house:
            return raw
        return torch.sigmoid(torch.tensor(raw)).numpy()

    if kind == "encoded":
        ready = _encode_frame(X, Config)
    else:
        ready = X

    if house:
        return np.asarray(model.predict(ready)).reshape(-1)
    return np.asarray(model.predict_proba(ready)[:, 1]).reshape(-1)


def _best_of_each(rows, config):
    lower_is_better = config.DATASET == "house_prices"
    chosen = {}
    for row in rows:
        name = row["model"]
        if name not in chosen:
            chosen[name] = row
            continue
        better = row["metric"] < chosen[name]["metric"]
        if not lower_is_better:
            better = row["metric"] > chosen[name]["metric"]
        if better:
            chosen[name] = row
    return list(chosen.values())


def _fit_member(row, X, y, config):
    name = row["model"]
    params = row["params"]
    if name in ("lasso", "ridge", "elasticnet"):
        regressor = _house_regressor(params["penalty"], params["alpha"], params.get("l1_ratio"), config)
        model = _build_baseline_houses(regressor, config)
        model.fit(X, y)
        return {"name": name, "model": model, "kind": "pipeline"}

    if name == "logreg" and "logreg_C" not in params:
        model = _build_baseline_titanic(config)
        model.fit(X, y)
        return {"name": name, "model": model, "kind": "pipeline"}

    if name == "mlp":
        model = fit_dl_model(params, X, y, config)
        return {"name": name, "model": model, "kind": "mlp"}

    model = fit_boost_model(params, X, y, config)
    house_gb = name == "gradient_boosting" and config.DATASET == "house_prices"
    if name == "logreg" or house_gb:
        kind = "pipeline"
    else:
        kind = "encoded"
    return {"name": name, "model": model, "kind": kind}


def fit_ensemble(rows, X, y, config, filename=None):
    if filename is None:
        filename = config.MODEL_PATH
    members = []
    for row in _best_of_each(rows, config):
        members.append(_fit_member(row, X, y, config))
    ensemble = AverageEnsemble(members, config.DATASET)
    joblib.dump(ensemble, filename)
    names = [member["name"] for member in members]
    print("Ensemble:", ", ".join(names))
    return ensemble
