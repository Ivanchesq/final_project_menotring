import optuna
import joblib
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from encode_house_price_linear import FREQ_COLS, OHE_COLS, ORDINAL_MAPS

NOMINAL_COLS = OHE_COLS + FREQ_COLS


class _HouseCatBoost(BaseEstimator, RegressorMixin):
    def __init__(self, iterations=100, depth=6, learning_rate=0.1, verbose=0, random_seed=0):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.random_seed = random_seed

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            random_seed=self.random_seed,
            cat_features=NOMINAL_COLS,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class _OrdinalMapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        for col, mapping in ORDINAL_MAPS.items():
            data[col] = data[col].map(mapping)
        return data


def _encode_frame(X, config, model_name=None):
    frame = X.copy()
    if config.DATASET != "house_prices":
        for col in frame.select_dtypes(include=["category", "object", "string"]).columns:
            frame[col] = frame[col].astype("int64")
        return frame

    for col, mapping in ORDINAL_MAPS.items():
        frame[col] = frame[col].map(mapping)
    for col in NOMINAL_COLS:
        frame[col] = frame[col].astype("category")
    if model_name == "gradient_boosting":
        for col in NOMINAL_COLS:
            frame[col] = frame[col].cat.codes
    return frame


def _house_gb_pipeline(params):
    return Pipeline([
        ("ordinal", _OrdinalMapper()),
        ("nominal", ColumnTransformer(
            transformers=[(
                "codes",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                NOMINAL_COLS,
            )],
            remainder="passthrough",
        )),
        ("regressor", GradientBoostingRegressor(**params)),
    ])


def objective(trial, X, y, config):
    house = config.DATASET == "house_prices"
    choices = (
        ["lgbm", "catboost", "gradient_boosting"]
        if house
        else ["lgbm", "catboost", "logreg", "gradient_boosting"]
    )
    classifier_name = trial.suggest_categorical("classifier", choices)
    X_trial = X if house and classifier_name == "gradient_boosting" else _encode_frame(X, config, classifier_name)

    if classifier_name == "lgbm":
        space = config.LGBM_SEARCH_SPACE
        params = {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("lgbm_max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", *space["learning_rate"], log=True),
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        }
        model = (LGBMRegressor if house else LGBMClassifier)(**params)
    elif classifier_name == "gradient_boosting":
        space = config.GRADIENT_BOOST_SEARCH_SPACE
        params = {
            "n_estimators": trial.suggest_int("gb_n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("gb_max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("gb_learning_rate", *space["learning_rate"], log=True),
            "subsample": trial.suggest_float("gb_subsample", *space["subsample"]),
            "random_state": config.RANDOM_STATE,
        }
        model = _house_gb_pipeline(params) if house else GradientBoostingClassifier(**params)
    elif classifier_name == "catboost":
        space = config.CAT_SEARCH_SPACE
        params = {
            "iterations": trial.suggest_int("cat_iterations", *space["iterations"]),
            "depth": trial.suggest_int("cat_depth", *space["depth"]),
            "learning_rate": trial.suggest_float("cat_learning_rate", *space["learning_rate"], log=True),
            "verbose": 0,
            "random_seed": config.RANDOM_STATE,
        }
        model = _HouseCatBoost(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            verbose=params["verbose"],
            random_seed=params["random_seed"],
        ) if house else CatBoostClassifier(**params)
    else:
        space = config.LOGREG_SEARCH_SPACE
        params = {
            "C": trial.suggest_float("logreg_C", *space["C"], log=True),
            "max_iter": 1000,
            "random_state": config.RANDOM_STATE,
        }
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), config.NUMERIC_COLS_TITANIC)],
            remainder="passthrough",
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**params)),
        ])
        X_trial = X

    if house:
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        scores = cross_val_score(model, X_trial, y, cv=cv, scoring="neg_root_mean_squared_error")
        return float(-scores.mean())

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X_trial, y, cv=cv, scoring="accuracy")
    return float(scores.mean())


def run_optimization(X, y, config):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    direction = "minimize" if config.DATASET == "house_prices" else "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(
        lambda trial: objective(trial, X, y, config),
        n_trials=config.N_TRIALS,
        show_progress_bar=True,
    )
    return study


def fit_boost_model(best_params, X, y, config):
    house = config.DATASET == "house_prices"
    classifier_name = best_params["classifier"]
    X_train = X if house and classifier_name == "gradient_boosting" else _encode_frame(X, config, classifier_name)

    if classifier_name == "lgbm":
        params = {
            "n_estimators": best_params["lgbm_n_estimators"],
            "max_depth": best_params["lgbm_max_depth"],
            "learning_rate": best_params["lgbm_learning_rate"],
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        }
        model = (LGBMRegressor if house else LGBMClassifier)(**params)
    elif classifier_name == "catboost":
        params = {
            "iterations": best_params["cat_iterations"],
            "depth": best_params["cat_depth"],
            "learning_rate": best_params["cat_learning_rate"],
            "verbose": 0,
            "random_seed": config.RANDOM_STATE,
        }
        model = _HouseCatBoost(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            verbose=params["verbose"],
            random_seed=params["random_seed"],
        ) if house else CatBoostClassifier(**params)
    elif classifier_name == "gradient_boosting":
        params = {
            "n_estimators": best_params["gb_n_estimators"],
            "max_depth": best_params["gb_max_depth"],
            "learning_rate": best_params["gb_learning_rate"],
            "subsample": best_params["gb_subsample"],
            "random_state": config.RANDOM_STATE,
        }
        model = _house_gb_pipeline(params) if house else GradientBoostingClassifier(**params)
    else:
        params = {
            "C": best_params["logreg_C"],
            "max_iter": 1000,
            "random_state": config.RANDOM_STATE,
        }
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), config.NUMERIC_COLS_TITANIC)],
            remainder="passthrough",
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**params)),
        ])
        X_train = X

    model.fit(X_train, y)
    return model


def save_best_model(study, X, y, config, filename=None):
    if filename is None:
        filename = config.MODEL_PATH
    model = fit_boost_model(study.best_params, X, y, config)
    joblib.dump(model, filename)
    return model