import numpy as np
import pandas as pd
import joblib
import torch
from config import Config
from encode_house_price_linear import FREQ_COLS, OHE_COLS, ORDINAL_COLS


def _restore_house_strings(X_test):
    data = X_test.copy()
    for col in OHE_COLS + ORDINAL_COLS + FREQ_COLS:
        data[col] = data[col].astype(str)
    return data


def _predict_house(model, X_test):
    X_test = _restore_house_strings(X_test)
    if isinstance(model, torch.nn.Module):
        model.eval()
        X_num = model.preprocessor.transform(X_test)
        with torch.no_grad():
            preds = model(torch.tensor(X_num, dtype=torch.float32)).numpy().flatten()
    elif hasattr(model, "named_steps"):
        preds = model.predict(X_test)
    else:
        from optimize import _encode_frame
        preds = model.predict(_encode_frame(X_test, Config))
    return np.maximum(np.expm1(preds), 0)


def generate_submission(model_path="best_model.joblib", output_path="submission.csv"):
    if Config.DATASET == "house_prices":
        import build_dl
        test_df = pd.read_csv(Config.PROCESSED_DIR_HOUSES_PRICES / "test_processed.csv")
        ids = pd.read_csv(Config.TEST_DIR_HOUSES_PRICES)["Id"]
        model = joblib.load(model_path)
        prices = _predict_house(model, test_df)
        submission = pd.DataFrame({"Id": ids, "SalePrice": prices})
        submission_path = Config.SUBMISSION_DIR_HOUSES_PRICES
        submission.to_csv(submission_path, index=False)
        print(f"Сабмит успешно сохранен: {submission_path}")
        return

    test_df = pd.read_csv(Config.PROCESSED_DIR_TITANIC / "test_processed.csv")

    if "PassengerId" in test_df.columns:
        passenger_ids = test_df["PassengerId"]
        X_test = test_df.drop("PassengerId", axis=1)
    else:
        raw_test = pd.read_csv(Config.TEST_DIR_TITANIC)
        passenger_ids = raw_test["PassengerId"]
        X_test = test_df

    model = joblib.load(model_path)

    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            X_inf = X_test.copy()
            for col in X_inf.select_dtypes(include=["category", "object"]).columns:
                X_inf[col] = X_inf[col].astype("float32")
            X_tensor = torch.tensor(X_inf.astype("float32").values, dtype=torch.float32)
            logits = model(X_tensor)
            preds = (torch.sigmoid(logits) >= 0.5).int().numpy().flatten()
    elif not hasattr(model, "named_steps"):
        X_test_inference = X_test.copy()
        for col in X_test_inference.select_dtypes(include=["category", "object"]).columns:
            X_test_inference[col] = X_test_inference[col].astype("int64")
        preds = model.predict(X_test_inference)
    else:
        preds = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds.astype(int)
    })

    submission_path = Config.SUBMISSION_DIR_TITANIC
    submission.to_csv(submission_path, index=False)
    print(f"Сабмит успешно сохранен: {submission_path}")


if __name__ == "__main__":
    generate_submission()