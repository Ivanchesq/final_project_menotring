from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    RANDOM_STATE = 42
    USE_DL = False
    EPOCH = 100
    DATA_DIR = BASE_DIR / "data"
    TITANIC_DIR = DATA_DIR / "titanic"
    PROCESSED_DIR_TITANIC = TITANIC_DIR / "processed"
    TRAIN_DIR_TITANIC = TITANIC_DIR / "train.csv"
    TEST_DIR_TITANIC = TITANIC_DIR / "test.csv"
    SUBMISSION_DIR_TITANIC = TITANIC_DIR / "submission.csv"
    TARGET_COL_TITANIC = "Survived"
    NUMERIC_COLS_TITANIC = ["Age", "Fare", "FamilySize"]
    HOUSES_PRICES_DIR = DATA_DIR / "house_prices"
    PROCESSED_DIR_HOUSES_PRICES = HOUSES_PRICES_DIR / "processed"
    TRAIN_DIR_HOUSES_PRICES = HOUSES_PRICES_DIR / "train.csv"
    TEST_DIR_HOUSES_PRICES = HOUSES_PRICES_DIR / "test.csv"
    SUBMISSION_DIR_HOUSES_PRICES = HOUSES_PRICES_DIR / "submission.csv"
    TARGET_COL_HOUSES_PRICES = "SalePrice"
    DATASET =  "house_prices"#"titanic"#
    N_TRIALS = 50
    CV_FOLDS = 5
    MODEL_PATH = "best_model.joblib"
    RESULTS_DIR = BASE_DIR / "results"
    TOP_N = 5
    DL_N_TRIALS = 10
    DL_PATIENCE = 5
    DL_LR_PATIENCE = 2
    DL_LR_FACTOR = 0.5
    DL_MIN_DELTA = 1e-4
    DL_MONITOR_FRACTION = 0.1
    DL_HIDDEN_STEP = 32
    DL_LR_LOG = True
    DL_SGD_MOMENTUM = 0.9
    DL_THRESHOLD = 0.5
    LGBM_SEARCH_SPACE = {
        "n_estimators": (50, 300),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
    }

    CAT_SEARCH_SPACE = {
        "iterations": (50, 300),
        "depth": (3, 10),
        "learning_rate": (0.01, 0.3),
    }

    LOGREG_SEARCH_SPACE = {
        "C": (1e-3, 1e2),
    }
    GRADIENT_BOOST_SEARCH_SPACE = {
        "n_estimators": (50, 300),
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0),
    }
    REGULARIZER_SEARCH_SPACE = {
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": (1e-3, 1e2),
        "l1_ratio": (0.1, 0.9),
    }
    DL_DEFAULT = {
        "n_layers": 2,
        "hidden": 64,
        "dropout": 0.3,
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": EPOCH,
        "activation": "relu",
        "optimizer": "adam",
    }
    DL_SEARCH_SPACE = {
        "n_layers": (1, 3),
        "hidden": (32, 128),
        "dropout": (0.0, 0.5),
        "lr": (1e-4, 1e-2),
        "batch_size": [16, 32, 64],
        "epochs": (10, 500),
        "activation": ["relu", "gelu"],
        "optimizer": ["adam", "adamw", "sgd"],
    }