import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from encode_house_price_linear import NUM_COLS, OHE_COLS, BINARY_COLS, ORDINAL_MAPS, ORDINAL_COLS,FREQ_COLS
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import  OrdinalEncoder,TargetEncoder
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
import optuna


def _house_regressor(penalty, alpha, l1_ratio, config):
    if penalty == "l1":
        return Lasso(alpha=alpha, max_iter=10000, random_state=config.RANDOM_STATE)
    if penalty == "l2":
        return Ridge(alpha=alpha, random_state=config.RANDOM_STATE)
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=10000,
        random_state=config.RANDOM_STATE,
    )



def _build_baseline_titanic(config):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS_TITANIC)
        ],
        remainder="passthrough"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000))
    ])



def _build_baseline_houses(regressor,config):
    ordinal_categories = [list(ORDINAL_MAPS[col]) for col in ORDINAL_COLS]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False), OHE_COLS),
            ("neighborhood", TargetEncoder(target_type="continuous"), FREQ_COLS),
            ("ord", OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ), ORDINAL_COLS)
        ],
        remainder = "passthrough"
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

def run_baseline(X: pd.DataFrame, y: pd.Series, config) -> float:
    if config.DATASET == "titanic":
        model = _build_baseline_titanic(config)
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return float(scores.mean()), None
    if config.DATASET == "house_prices":
        space = config.REGULARIZER_SEARCH_SPACE
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

        def objective(trial):
            penalty = trial.suggest_categorical("penalty", space["penalty"])
            alpha = trial.suggest_float("alpha", *space["alpha"], log=True)
            l1_ratio = None
            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float("l1_ratio", *space["l1_ratio"])
            regressor = _house_regressor(penalty, alpha, l1_ratio, config)
            model = _build_baseline_houses(regressor, config)
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
            return float(-scores.mean())

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.N_TRIALS, show_progress_bar=True)
        print(f"Best linear params: {study.best_params}")
        return study.best_value, study
