import pandas as pd
import joblib
import torch
from config import Config


def generate_submission(model_path="best_model.joblib", output_path="submission.csv"):
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