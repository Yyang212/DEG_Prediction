from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for dep in [ROOT / ".model_deps", ROOT / ".scipy_pkg", ROOT / ".xgb_pkg"]:
    if dep.exists():
        sys.path.insert(0, str(dep))

import xgboost as xgb


DATASET = ROOT / "outputs" / "deg_nn_dataset" / "DEG_NN_Dataset.xlsx"
OUT = ROOT / "outputs" / "deg_xgboost_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "target_DEG_pct"


def clean_value(value):
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [clean_value(v) for v in value.tolist()]
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return value


def records(df: pd.DataFrame):
    return [{str(k): clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def pearson_corr(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def spearman_corr(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(paired.iloc[:, 0].rank(method="average").corr(paired.iloc[:, 1].rank(method="average")))


def approx_p_value(r, n):
    if pd.isna(r) or abs(r) >= 1 or n <= 3:
        return np.nan
    z = np.arctanh(r) * math.sqrt(n - 3)
    return float(math.erfc(abs(z) / math.sqrt(2)))


def load_data():
    wide = pd.read_excel(DATASET, sheet_name="Wide_Model_Data")
    cols = pd.read_excel(DATASET, sheet_name="Input_Columns")
    feature_cols = cols.loc[cols["role"] == "feature", "column"].tolist()
    wide["sample_time"] = pd.to_datetime(wide["sample_time"], errors="coerce")
    wide["lab_time"] = pd.to_datetime(wide["lab_time"], errors="coerce")
    wide[TARGET] = pd.to_numeric(wide[TARGET], errors="coerce")
    for c in feature_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    return wide, feature_cols


def make_split_masks(data):
    split = data["split_chrono"].astype(str)
    return split.eq("train"), split.eq("validation"), split.eq("test")


def feature_analysis(data, feature_cols, train_mask):
    y_train = data.loc[train_mask, TARGET]
    rows = []
    for c in feature_cols:
        s = data[c]
        train_s = s.loc[train_mask]
        rows.append(
            {
                "feature": c,
                "train_missing": int(train_s.isna().sum()),
                "all_missing": int(s.isna().sum()),
                "train_std": float(train_s.std(ddof=0)) if train_s.notna().any() else np.nan,
                "train_mean": float(train_s.mean()) if train_s.notna().any() else np.nan,
                "pearson_r_train": pearson_corr(train_s, y_train),
                "spearman_r_train": spearman_corr(train_s, y_train),
            }
        )
    diag = pd.DataFrame(rows)
    diag["abs_pearson"] = diag["pearson_r_train"].abs()
    diag["abs_spearman"] = diag["spearman_r_train"].abs()
    diag = diag.sort_values(["abs_pearson", "abs_spearman"], ascending=False).reset_index(drop=True)
    return diag


def select_top_k(diag, k):
    usable = diag.loc[diag["train_std"].fillna(0) > 1e-12]
    return usable.head(k)["feature"].tolist()


def baseline_predictions(data, mask_train_for_fit, mask_eval):
    y = data[TARGET]
    train_mean = float(y.loc[mask_train_for_fit].mean())
    phase_means = y.loc[mask_train_for_fit].groupby(data.loc[mask_train_for_fit, "phase"]).mean().to_dict()
    pred_mean = np.full(mask_eval.sum(), train_mean)
    pred_phase = data.loc[mask_eval, "phase"].map(phase_means).fillna(train_mean).astype(float).to_numpy()
    return {
        "mean": pred_mean,
        "phase_mean": pred_phase,
    }


def fit_ridge(train_df, y_train, features, alpha):
    X = train_df[features].copy()
    med = X.median()
    mean = X.fillna(med).mean()
    std = X.fillna(med).std(ddof=0).replace(0, 1.0)
    Xs = ((X.fillna(med) - mean) / std).to_numpy(dtype=float)
    Xb = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
    y = np.asarray(y_train, dtype=float).reshape(-1, 1)
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(Xb.T @ Xb + alpha * I, Xb.T @ y).ravel()
    return {"beta": beta, "median": med, "mean": mean, "std": std, "features": list(features), "alpha": alpha}


def ridge_predict(model, df, features):
    X = df[features].copy().fillna(model["median"])
    X = ((X - model["mean"]) / model["std"]).to_numpy(dtype=float)
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return Xb @ model["beta"]


def fit_xgb(train_df, y_train, val_df, y_val, features, params, num_boost_round=500, early_stopping_rounds=30):
    dtrain = xgb.DMatrix(train_df[features], label=y_train, feature_names=features, missing=np.nan)
    dval = xgb.DMatrix(val_df[features], label=y_val, feature_names=features, missing=np.nan)
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", num_boost_round - 1))
    pred_val = booster.predict(dval, iteration_range=(0, best_iter + 1))
    return booster, best_iter, pred_val


def xgb_predict(booster, df, features, best_iter):
    dmat = xgb.DMatrix(df[features], feature_names=features, missing=np.nan)
    return booster.predict(dmat, iteration_range=(0, best_iter + 1))


def feature_importance_from_booster(booster, features):
    gain = booster.get_score(importance_type="gain")
    rows = []
    for f in features:
        rows.append({"feature": f, "importance_gain": float(gain.get(f, 0.0))})
    return pd.DataFrame(rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)


def main():
    data, feature_cols = load_data()
    train_mask, val_mask, test_mask = make_split_masks(data)

    train_df = data.loc[train_mask].copy()
    val_df = data.loc[val_mask].copy()
    test_df = data.loc[test_mask].copy()
    trainval_df = data.loc[train_mask | val_mask].copy()

    y_train = train_df[TARGET].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=float)
    y_trainval = trainval_df[TARGET].to_numpy(dtype=float)

    diag = feature_analysis(data, feature_cols, train_mask)
    top_corr = diag.head(40).copy()

    target_summary = pd.DataFrame(
        [
            {"metric": "samples_total", "value": int(len(data)), "note": ""},
            {"metric": "samples_train", "value": int(train_mask.sum()), "note": ""},
            {"metric": "samples_validation", "value": int(val_mask.sum()), "note": ""},
            {"metric": "samples_test", "value": int(test_mask.sum()), "note": ""},
            {"metric": "feature_count", "value": int(len(feature_cols)), "note": "role=feature"},
            {"metric": "target_min", "value": float(data[TARGET].min()), "note": ""},
            {"metric": "target_max", "value": float(data[TARGET].max()), "note": ""},
            {"metric": "target_mean", "value": float(data[TARGET].mean()), "note": ""},
            {"metric": "target_std", "value": float(data[TARGET].std(ddof=0)), "note": ""},
            {"metric": "train_target_mean", "value": float(train_df[TARGET].mean()), "note": ""},
            {"metric": "val_target_mean", "value": float(val_df[TARGET].mean()), "note": ""},
            {"metric": "test_target_mean", "value": float(test_df[TARGET].mean()), "note": ""},
        ]
    )

    phase_means = data.loc[train_mask, TARGET].groupby(data.loc[train_mask, "phase"]).mean().to_dict()

    candidate_ks = [8, 12, 16, 24, 32]
    xgb_param_grid = [
        {
            "max_depth": d,
            "eta": eta,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "min_child_weight": mcw,
            "gamma": 0.0,
            "lambda": 1.0,
            "alpha": 0.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": 42,
            "tree_method": "hist",
        }
        for d, eta, subsample, colsample, mcw in itertools.product(
            [2, 3],
            [0.03, 0.05],
            [0.8, 1.0],
            [0.7, 1.0],
            [1, 3],
        )
    ]
    ridge_alphas = [0.1, 1.0, 10.0]

    trials = []
    xgb_models = []
    ridge_models = []

    for k in candidate_ks:
        selected = select_top_k(diag, k)

        # Ridge benchmark
        for alpha in ridge_alphas:
            ridge_model = fit_ridge(train_df, y_train, selected, alpha)
            pred_val = ridge_predict(ridge_model, val_df, selected)
            ridge_models.append(
                {
                    "model": "ridge",
                    "k": k,
                    "alpha": alpha,
                    "selected_features": selected,
                    "val_rmse": rmse(y_val, pred_val),
                    "val_mae": mae(y_val, pred_val),
                    "val_r2": r2(y_val, pred_val),
                    "model_obj": ridge_model,
                }
            )

        # XGBoost grid
        for params in xgb_param_grid:
            booster, best_iter, pred_val = fit_xgb(train_df, y_train, val_df, y_val, selected, params)
            xgb_models.append(
                {
                    "model": "xgboost",
                    "k": k,
                    "max_depth": params["max_depth"],
                    "eta": params["eta"],
                    "subsample": params["subsample"],
                    "colsample_bytree": params["colsample_bytree"],
                    "min_child_weight": params["min_child_weight"],
                    "best_iter": best_iter,
                    "selected_features": selected,
                    "val_rmse": rmse(y_val, pred_val),
                    "val_mae": mae(y_val, pred_val),
                    "val_r2": r2(y_val, pred_val),
                    "model_obj": booster,
                }
            )

    mean_pred = baseline_predictions(data, train_mask, val_mask)
    baseline_trials = [
        {
            "model": "mean",
            "k": 0,
            "val_rmse": rmse(y_val, mean_pred["mean"]),
            "val_mae": mae(y_val, mean_pred["mean"]),
            "val_r2": r2(y_val, mean_pred["mean"]),
        },
        {
            "model": "phase_mean",
            "k": 0,
            "val_rmse": rmse(y_val, mean_pred["phase_mean"]),
            "val_mae": mae(y_val, mean_pred["phase_mean"]),
            "val_r2": r2(y_val, mean_pred["phase_mean"]),
        },
    ]

    combined = pd.DataFrame(
        [
            {k: clean_value(v) for k, v in row.items() if k != "model_obj"}
            for row in (baseline_trials + ridge_models + xgb_models)
        ]
    ).sort_values("val_rmse").reset_index(drop=True)
    combined["rank_by_val_rmse"] = np.arange(1, len(combined) + 1)

    best_xgb = min(xgb_models, key=lambda x: x["val_rmse"])
    best_ridge = min(ridge_models, key=lambda x: x["val_rmse"])

    if best_xgb["val_rmse"] <= best_ridge["val_rmse"]:
        selected_kind = "xgboost"
        selected = best_xgb["selected_features"]
        params = {k: best_xgb[k] for k in ["max_depth", "eta", "subsample", "colsample_bytree", "min_child_weight"]}
        params.update(
            {
                "gamma": 0.0,
                "lambda": 1.0,
                "alpha": 0.0,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "seed": 42,
                "tree_method": "hist",
            }
        )
        dtrainval = xgb.DMatrix(trainval_df[selected], label=y_trainval, feature_names=selected, missing=np.nan)
        dtest = xgb.DMatrix(test_df[selected], label=y_test, feature_names=selected, missing=np.nan)
        final_booster = xgb.train(
            params=params,
            dtrain=dtrainval,
            num_boost_round=best_xgb["best_iter"] + 1,
            evals=[(dtrainval, "trainval")],
            verbose_eval=False,
        )
        pred_test = final_booster.predict(dtest)
        pred_all = final_booster.predict(xgb.DMatrix(data[selected], feature_names=selected, missing=np.nan))
        feature_importance = feature_importance_from_booster(final_booster, selected)
        model_state = {
            "model": "xgboost",
            "selected_features": selected,
            "params": params,
            "best_iter": best_xgb["best_iter"],
        }
        trainval_metrics = {
            "rmse": rmse(y_trainval, final_booster.predict(dtrainval)),
            "mae": mae(y_trainval, final_booster.predict(dtrainval)),
            "r2": r2(y_trainval, final_booster.predict(dtrainval)),
            "mape_pct": mape(y_trainval, final_booster.predict(dtrainval)),
        }
        test_metrics = {
            "rmse": rmse(y_test, pred_test),
            "mae": mae(y_test, pred_test),
            "r2": r2(y_test, pred_test),
            "mape_pct": mape(y_test, pred_test),
        }
        best_model_meta = {
            "model": "xgboost",
            "selected_features": selected,
            "k": int(best_xgb["k"]),
            "max_depth": int(best_xgb["max_depth"]),
            "eta": float(best_xgb["eta"]),
            "subsample": float(best_xgb["subsample"]),
            "colsample_bytree": float(best_xgb["colsample_bytree"]),
            "min_child_weight": int(best_xgb["min_child_weight"]),
            "best_iter": int(best_xgb["best_iter"]),
        }
        final_booster.save_model(str(OUT / "best_xgb_model.json"))
    else:
        selected_kind = "ridge"
        selected = best_ridge["selected_features"]
        ridge_final = fit_ridge(trainval_df, y_trainval, selected, best_ridge["alpha"])
        pred_test = ridge_predict(ridge_final, test_df, selected)
        pred_all = ridge_predict(ridge_final, data, selected)
        feature_importance = pd.DataFrame(
            {"feature": selected, "importance_gain": np.abs(ridge_final["beta"][1:])}
        ).sort_values("importance_gain", ascending=False)
        model_state = {
            "model": "ridge",
            "selected_features": selected,
            "alpha": float(best_ridge["alpha"]),
        }
        trainval_pred = ridge_predict(ridge_final, trainval_df, selected)
        trainval_metrics = {
            "rmse": rmse(y_trainval, trainval_pred),
            "mae": mae(y_trainval, trainval_pred),
            "r2": r2(y_trainval, trainval_pred),
            "mape_pct": mape(y_trainval, trainval_pred),
        }
        test_metrics = {
            "rmse": rmse(y_test, pred_test),
            "mae": mae(y_test, pred_test),
            "r2": r2(y_test, pred_test),
            "mape_pct": mape(y_test, pred_test),
        }
        best_model_meta = {
            "model": "ridge",
            "selected_features": selected,
            "k": int(best_ridge["k"]),
            "alpha": float(best_ridge["alpha"]),
        }

    test_pred_df = test_df[["sample_id", "sample_time", "phase", TARGET]].copy()
    test_pred_df["predicted_DEG_pct"] = pred_test
    test_pred_df["error"] = test_pred_df["predicted_DEG_pct"] - test_pred_df[TARGET]
    test_pred_df["abs_error"] = test_pred_df["error"].abs()

    all_pred_df = data[["sample_id", "sample_time", "phase", TARGET]].copy()
    all_pred_df["predicted_DEG_pct"] = pred_all
    all_pred_df["error"] = all_pred_df["predicted_DEG_pct"] - all_pred_df[TARGET]

    selected_diag = diag[diag["feature"].isin(selected)].copy().reset_index(drop=True)
    selected_diag["selected_rank"] = np.arange(1, len(selected_diag) + 1)

    model_trials = combined.head(80).copy()
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)

    stage_summary = (
        diag.assign(
            base_variable=diag["feature"].str.replace(r"^(now_|lag_\d+h_|win_\d+h_)", "", regex=True)
        )
        .assign(stage=lambda d: d["base_variable"].str.extract(r"^(feed_flow|molar_ratio|ester1|ester2|prepoly1|prepoly2|final)")[0])
        .groupby("stage", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_pearson=("abs_pearson", "max"),
            mean_abs_pearson=("abs_pearson", "mean"),
            max_abs_spearman=("abs_spearman", "max"),
            mean_abs_spearman=("abs_spearman", "mean"),
        )
        .reset_index()
        .sort_values("max_abs_pearson", ascending=False)
    )

    readme = pd.DataFrame(
        [
            {"item": "目的", "detail": "先做数据诊断，再用官方 XGBoost 在小样本上试预测 DEG。"},
            {"item": "样本", "detail": "总计 64 条样本，训练 44、验证 10、测试 10。"},
            {"item": "特征", "detail": "默认使用 336 个过程派生特征中的训练集相关性前 k 个做输入，k 通过验证集选择。"},
            {"item": "XGBoost", "detail": "使用官方 xgboost 包，固定验证集选超参数，再用 train+validation 重训并在测试集上评估。"},
            {"item": "提醒", "detail": "这是小样本探索，不是最终生产模型；如果测试误差不稳，继续补样本比堆模型更重要。"},
        ]
    )

    summary = {
        "selected_model": best_model_meta,
        "trainval_metrics": trainval_metrics,
        "test_metrics": test_metrics,
        "validation_best_rmse": float(min(best_xgb["val_rmse"], best_ridge["val_rmse"])),
        "xgb_best_val_rmse": float(best_xgb["val_rmse"]),
        "ridge_best_val_rmse": float(best_ridge["val_rmse"]),
        "baseline_val_rmse": {r["model"]: float(r["val_rmse"]) for r in baseline_trials},
        "n_samples": int(len(data)),
        "n_features": int(len(feature_cols)),
        "selected_feature_count": int(len(selected)),
    }

    payload = {
        "sheets": {
            "README": records(readme),
            "Target_Summary": records(target_summary),
            "Feature_Analysis": records(diag),
            "Top_Correlations": records(top_corr),
            "Model_Trials": records(model_trials),
            "Best_Model": records(pd.DataFrame([best_model_meta | trainval_metrics | test_metrics])),
            "Selected_Features": records(selected_diag),
            "Feature_Importance": records(feature_importance),
            "Stage_Summary": records(stage_summary),
            "Test_Predictions": records(test_pred_df),
            "All_Predictions": records(all_pred_df),
        },
        "columns": {
            "README": list(readme.columns),
            "Target_Summary": list(target_summary.columns),
            "Feature_Analysis": list(diag.columns),
            "Top_Correlations": list(top_corr.columns),
            "Model_Trials": list(model_trials.columns),
            "Best_Model": list(pd.DataFrame([best_model_meta | trainval_metrics | test_metrics]).columns),
            "Selected_Features": list(selected_diag.columns),
            "Feature_Importance": list(feature_importance.columns),
            "Stage_Summary": list(stage_summary.columns),
            "Test_Predictions": list(test_pred_df.columns),
            "All_Predictions": list(all_pred_df.columns),
        },
        "summary": summary,
        "model_state": model_state,
    }

    with (OUT / "deg_xgboost_payload.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with (OUT / "deg_xgboost_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = "42"
    main()
