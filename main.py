from config import Config
from process_data_titanic import process_data
from build_baseline import run_baseline
from optimize import run_optimization
from optimize_dl import run_dl_optimization
from ensemble import fit_ensemble
from predict import generate_submission
from results import report_results, rows_from_study
from process_data_house_price import process_data as process_house_data


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

        baseline_score, _ = run_baseline(X, y, Config)
        print(f"Baseline Accuracy: {baseline_score:.4f}")
        rows = [{"model": "logreg", "metric": baseline_score, "params": {}}]

        if USE_DL:
            study = run_dl_optimization(X, y, Config)
            print(f"Best DL Accuracy: {study.best_value:.4f} Parameters: {study.best_params}")
            rows += rows_from_study(study, "dl")
        else:
            study = run_optimization(X, y, Config)
            print(f"Best Optimized Accuracy: {study.best_value:.4f} Parameters: {study.best_params}")
            rows += rows_from_study(study, "boost")
        report_results(rows, Config)
        fit_ensemble(rows, X, y, Config)
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
        X = train_df.drop(Config.TARGET_COL_HOUSES_PRICES, axis=1)
        y = train_df[Config.TARGET_COL_HOUSES_PRICES]
        baseline_score, baseline_study = run_baseline(X, y, Config)
        print(f"Baseline RMSE: {baseline_score:.4f}")
        rows = rows_from_study(baseline_study, "linear")

        if USE_DL:
            study = run_dl_optimization(X, y, Config)
            print(f"Best DL RMSE: {study.best_value:.4f} Parameters: {study.best_params}")
            rows += rows_from_study(study, "dl")
        else:
            study = run_optimization(X, y, Config)
            print(f"Best Optimized RMSE: {study.best_value:.4f} Parameters: {study.best_params}")
            rows += rows_from_study(study, "boost")
        report_results(rows, Config)
        fit_ensemble(rows, X, y, Config)
        generate_submission()




if __name__ == "__main__":
    main()