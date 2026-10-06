from config import Config
from process_data_titanic import process_data
from build_baseline import run_baseline
from optimize import run_optimization, save_best_model
from predict import generate_submission

def main():
    Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, age_medians, fare_median, embarked_mode, age_median = process_data(
        Config.TRAIN_DIR, is_train=True
    )
    test_df = process_data(
        Config.TEST_DIR,
        is_train=False,
        age_medians=age_medians,
        fare_median=fare_median,
        embarked_mode=embarked_mode,
        age_median=age_median
    )

    train_df.to_csv(Config.PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(Config.PROCESSED_DIR / "test_processed.csv", index=False)

    X = train_df.drop(Config.TARGET_COL, axis=1)
    y = train_df[Config.TARGET_COL]

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

if __name__ == "__main__":
    main()