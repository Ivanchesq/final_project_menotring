from config import Config
from process_data_titanic import process_data
from build_baseline import run_baseline
from optimize import run_optimization, save_best_model
from build_dl import train_dl_model
from predict import generate_submission
from process_data_house_price import process_data as process_house_data
import joblib


def main():
    USE_DL = Config.USE_DL
    Config.PROCESSED_DIR_HOUSES_PRICES.mkdir(parents=True, exist_ok=True)
    Config.PROCESSED_DIR_TITANIC.mkdir(parents=True, exist_ok=True)
    dataset = Config.DATASET
    print(f"Dataset: {dataset}")
    if dataset == "titanic":
        train_df, age_medians, fare_median, embarked_mode, age_median = process_data(
            Config.TRAIN_DIR_TITANIC, is_train=True
        )
        test_df = process_data(
            Config.TEST_DIR_TITANIC,
            is_train=False,
            age_medians=age_medians,
            fare_median=fare_median,
            embarked_mode=embarked_mode,
            age_median=age_median
        )

        train_df.to_csv(Config.PROCESSED_DIR_TITANIC / "train_processed.csv", index=False)
        test_df.to_csv(Config.PROCESSED_DIR_TITANIC / "test_processed.csv", index=False)

        X = train_df.drop(Config.TARGET_COL_TITANIC, axis=1)
        y = train_df[Config.TARGET_COL_TITANIC]

        baseline_score = run_baseline(X, y, Config)
        print(f"Baseline Accuracy: {baseline_score:.4f}")

        if USE_DL:
            dl_model = train_dl_model(X, y, Config)
            joblib.dump(dl_model, "best_model.joblib")
        else:
            study = run_optimization(X, y, Config)
            print(f"Best Optimized Accuracy: {study.best_value:.4f} Parameters: {study.best_params}")
            save_best_model(study, X, y, Config)
        # я не видел смысла брать и дл и бустинги, поэтому такая логика
        generate_submission()
    elif dataset == "house_prices":
        train_df, stats = process_house_data(
            Config.TRAIN_DIR_HOUSES_PRICES, is_train=True
        )
        test_df = process_house_data(
            Config.TEST_DIR_HOUSES_PRICES,
            is_train=False,
            stats=stats,
        )

        train_df.to_csv(Config.PROCESSED_DIR_HOUSES_PRICES / "train_processed.csv", index=False)
        test_df.to_csv(Config.PROCESSED_DIR_HOUSES_PRICES / "test_processed.csv", index=False)



if __name__ == "__main__":
    main()