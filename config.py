from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    RANDOM_STATE = 42
    USE_DL = False
    DATA_DIR = BASE_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    TRAIN_DIR = DATA_DIR / "train.csv"
    TEST_DIR = DATA_DIR / "test.csv"

    TARGET_COL = "Survived"
    NUMERIC_COLS = ["Age", "Fare", "FamilySize"]

    N_TRIALS = 50
    CV_FOLDS = 5

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