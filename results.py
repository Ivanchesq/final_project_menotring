import json

import pandas as pd


LINEAR_NAMES = {"l1": "lasso", "l2": "ridge", "elasticnet": "elasticnet"}


def _model_name(params, family):
    if family == "linear":
        return LINEAR_NAMES.get(params.get("penalty"), "linear")
    if family == "boost":
        return params.get("classifier", "boost")
    return "mlp"


def rows_from_study(study, family):
    rows = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        rows.append({
            "model": _model_name(trial.params, family),
            "metric": trial.value,
            "params": trial.params,
        })
    return rows


def report_results(rows, config):
    frame = pd.DataFrame(rows)
    metric_name = "RMSE" if config.DATASET == "house_prices" else "Accuracy"
    best_first = config.DATASET == "house_prices"
    ordered = frame.sort_values("metric", ascending=best_first)
    by_model = (
        ordered.groupby("model", as_index=False)
        .first()
        .sort_values("metric", ascending=best_first)
        .reset_index(drop=True)
    )
    top = ordered.head(config.TOP_N).reset_index(drop=True)

    by_model_out = by_model[["model", "metric"]].rename(columns={"metric": metric_name})
    top_out = top[["model", "metric"]].rename(columns={"metric": metric_name})
    top_out["params"] = top["params"].map(lambda item: json.dumps(item, sort_keys=True))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    by_model_out.to_csv(config.RESULTS_DIR / f"{config.DATASET}_models.csv", index=False)
    top_out.to_csv(config.RESULTS_DIR / f"{config.DATASET}_top{config.TOP_N}.csv", index=False)

    print(f"\n{metric_name} by model")
    print(by_model_out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nTop {config.TOP_N}")
    print(top_out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return by_model_out, top_out
