import optuna
import joblib
from sklearn.model_selection import KFold, StratifiedKFold

from build_dl import fit_mlp, house_preprocessor, score_mlp, split_monitor, titanic_numeric, _tensors


def suggest_dl_params(trial, config):
    space = config.DL_SEARCH_SPACE
    return {
        "n_layers": trial.suggest_int("n_layers", *space["n_layers"]),
        "hidden": trial.suggest_int("hidden", *space["hidden"], step=config.DL_HIDDEN_STEP),
        "dropout": trial.suggest_float("dropout", *space["dropout"]),
        "lr": trial.suggest_float("lr", *space["lr"], log=config.DL_LR_LOG),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "epochs": trial.suggest_int("epochs", *space["epochs"]),
        "activation": trial.suggest_categorical("activation", space["activation"]),
        "optimizer": trial.suggest_categorical("optimizer", space["optimizer"]),
    }


def _fold_score(X_train, y_train, X_val, y_val, params, config):
    house = config.DATASET == "house_prices"
    X_fit, X_monitor, y_fit, y_monitor = split_monitor(X_train, y_train, config)
    if house:
        preprocessor = house_preprocessor()
        X_fit_num = preprocessor.fit_transform(X_fit, y_fit)
        X_monitor_num = preprocessor.transform(X_monitor)
        X_val_num = preprocessor.transform(X_val)
    else:
        X_fit_num = titanic_numeric(X_fit)
        X_monitor_num = titanic_numeric(X_monitor)
        X_val_num = titanic_numeric(X_val)

    X_fit_tensor, y_fit_tensor = _tensors(X_fit_num, y_fit)
    X_monitor_tensor, y_monitor_tensor = _tensors(X_monitor_num, y_monitor)
    X_val_tensor, y_val_tensor = _tensors(X_val_num, y_val)
    model = fit_mlp(X_fit_tensor, y_fit_tensor, X_monitor_tensor, y_monitor_tensor, params, config)
    return score_mlp(model, X_val_tensor, y_val_tensor, config)


def objective(trial, X, y, config):
    params = suggest_dl_params(trial, config)
    house = config.DATASET == "house_prices"
    if house:
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        splits = cv.split(X)
    else:
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        splits = cv.split(X, y)

    scores = []
    for train_idx, val_idx in splits:
        score = _fold_score(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[val_idx],
            y.iloc[val_idx],
            params,
            config,
        )
        scores.append(score)
    return float(sum(scores) / len(scores))


def run_dl_optimization(X, y, config):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if config.DATASET == "house_prices" else "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(
        lambda trial: objective(trial, X, y, config),
        n_trials=config.DL_N_TRIALS,
        show_progress_bar=True,
    )
    return study


def fit_dl_model(params, X, y, config):
    X_fit, X_monitor, y_fit, y_monitor = split_monitor(X, y, config)
    if config.DATASET == "house_prices":
        preprocessor = house_preprocessor()
        X_fit_num = preprocessor.fit_transform(X_fit, y_fit)
        X_monitor_num = preprocessor.transform(X_monitor)
        X_tensor, y_tensor = _tensors(X_fit_num, y_fit)
        X_monitor_tensor, y_monitor_tensor = _tensors(X_monitor_num, y_monitor)
        model = fit_mlp(X_tensor, y_tensor, X_monitor_tensor, y_monitor_tensor, params, config)
        model.preprocessor = preprocessor
    else:
        X_tensor, y_tensor = _tensors(titanic_numeric(X_fit), y_fit)
        X_monitor_tensor, y_monitor_tensor = _tensors(titanic_numeric(X_monitor), y_monitor)
        model = fit_mlp(X_tensor, y_tensor, X_monitor_tensor, y_monitor_tensor, params, config)

    return model


def save_best_dl_model(study, X, y, config, filename=None):
    if filename is None:
        filename = config.MODEL_PATH
    model = fit_dl_model(study.best_params, X, y, config)
    joblib.dump(model, filename)
    return model
