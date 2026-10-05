import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def objective(trial, X, y, config):
    X_trial = X.copy()
    classifier_name = trial.suggest_categorical("classifier", ["lgbm", "catboost", "logreg", "gradient_boosting"])

    for col in X_trial.select_dtypes(include=["category", "object"]).columns:
        X_trial[col] = X_trial[col].astype("int64")

    if classifier_name == "lgbm":
        space = config.LGBM_SEARCH_SPACE
        params = {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("lgbm_max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", *space["learning_rate"], log=True),
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        }
        model = LGBMClassifier(**params)
    elif classifier_name == "gradient_boosting":
        space = config.GRADIENT_BOOST_SEARCH_SPACE
        params = {
            "n_estimators": trial.suggest_int("gb_n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("gb_max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("gb_learning_rate", *space["learning_rate"], log=True),
            "subsample": trial.suggest_float("gb_subsample", *space["subsample"]),
            "random_state": config.RANDOM_STATE,
        }
        model = GradientBoostingClassifier(**params)
    elif classifier_name == "catboost":
        space = config.CAT_SEARCH_SPACE
        params = {
            "iterations": trial.suggest_int("cat_iterations", *space["iterations"]),
            "depth": trial.suggest_int("cat_depth", *space["depth"]),
            "learning_rate": trial.suggest_float("cat_learning_rate", *space["learning_rate"], log=True),
            "verbose": 0,
            "random_seed": config.RANDOM_STATE,
        }
        model = CatBoostClassifier(**params)

    else:
        space = config.LOGREG_SEARCH_SPACE
        params = {
            "C": trial.suggest_float("logreg_C", *space["C"], log=True),
            "max_iter": 1000,
            "random_state": config.RANDOM_STATE,
        }
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), config.NUMERIC_COLS)],
            remainder="passthrough"
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**params))
        ])
        X_trial = X

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X_trial, y, cv=cv, scoring="accuracy")
    return scores.mean()


def run_optimization(X, y, config):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, config),
        n_trials=config.N_TRIALS,
        show_progress_bar=True,
    )
    return study


def save_best_model(study, X, y, config, filename="best_model.joblib"):
    best_params = study.best_params
    classifier_name = best_params["classifier"]

    X_train = X.copy()
    if classifier_name in ["lgbm", "catboost", "gradient_boosting"]:
        for col in X_train.select_dtypes(include=["category", "object"]).columns:
            X_train[col] = X_train[col].astype("int64")

    if classifier_name == "lgbm":
        params = {
            "n_estimators": best_params["lgbm_n_estimators"],
            "max_depth": best_params["lgbm_max_depth"],
            "learning_rate": best_params["lgbm_learning_rate"],
            "verbose": -1,
            "random_state": config.RANDOM_STATE,
        }
        model = LGBMClassifier(**params)

    elif classifier_name == "catboost":
        params = {
            "iterations": best_params["cat_iterations"],
            "depth": best_params["cat_depth"],
            "learning_rate": best_params["cat_learning_rate"],
            "verbose": 0,
            "random_seed": config.RANDOM_STATE,
        }
        model = CatBoostClassifier(**params)

    elif classifier_name == "gradient_boosting":
        params = {
            "n_estimators": best_params["gb_n_estimators"],
            "max_depth": best_params["gb_max_depth"],
            "learning_rate": best_params["gb_learning_rate"],
            "subsample": best_params["gb_subsample"],
            "random_state": config.RANDOM_STATE,
        }
        model = GradientBoostingClassifier(**params)
    else:
        params = {
            "C": best_params["logreg_C"],
            "max_iter": 1000,
            "random_state": config.RANDOM_STATE,
        }
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), config.NUMERIC_COLS)],
            remainder="passthrough"
        )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**params))
        ])
        X_train = X

    model.fit(X_train, y)
    joblib.dump(model, filename)
    return model
