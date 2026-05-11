from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is available in the bundled runtime, but keep graceful fallback.
    stats = None


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs" / "deg_nn_dataset" / "DEG_NN_Dataset.xlsx"
OUT = ROOT / "outputs" / "deg_correlation_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "target_DEG_pct"

VARIABLE_LABELS = {
    "feed_flow": "进料流量",
    "molar_ratio": "摩尔比",
    "ester1_temp": "第一酯化_温度",
    "ester1_pressure": "第一酯化_压力",
    "ester1_level": "第一酯化_液位",
    "ester2_temp": "第二酯化_温度",
    "ester2_pressure": "第二酯化_压力",
    "ester2_level": "第二酯化_液位",
    "prepoly1_temp": "第一预缩聚_温度",
    "prepoly1_pressure": "第一预缩聚_压力",
    "prepoly1_level": "第一预缩聚_液位",
    "prepoly2_temp": "第二预缩聚_温度",
    "prepoly2_pressure": "第二预缩聚_压力",
    "prepoly2_level": "第二预缩聚_液位",
    "final_pressure": "终缩聚反应器_压力",
    "final_level": "终缩聚反应器_液位",
}

STAGE_LABELS = {
    "feed": "浆料配比",
    "molar": "浆料配比",
    "ester1": "第一酯化",
    "ester2": "第二酯化",
    "prepoly1": "第一预缩聚",
    "prepoly2": "第二预缩聚",
    "final": "终缩聚反应器",
}


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return value


def df_records(df: pd.DataFrame):
    return [
        {str(k): clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def parse_feature_name(feature: str):
    if feature.startswith("now_"):
        base = feature.removeprefix("now_")
        return "current", "now", "", base

    m = re.match(r"^lag_(\d+)h_(.+)$", feature)
    if m:
        return "lag", f"lag_{m.group(1)}h", "", m.group(2)

    m = re.match(r"^win_(\d+)h_(.+)_(mean|std|min|max)$", feature)
    if m:
        return "window", f"win_{m.group(1)}h", m.group(3), m.group(2)

    return "other", "", "", feature


def stage_for_base(base: str):
    for prefix, label in STAGE_LABELS.items():
        if base.startswith(prefix):
            return label
    return ""


def corr_pair(x: pd.Series, y: pd.Series, method: str):
    paired = pd.concat([pd.to_numeric(x, errors="coerce"), y], axis=1).dropna()
    n = len(paired)
    if n < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, n
    if stats is not None:
        if method == "pearson":
            res = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        else:
            res = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
        return float(res.statistic), float(res.pvalue), n
    if method == "spearman":
        left = paired.iloc[:, 0].rank(method="average")
        right = paired.iloc[:, 1].rank(method="average")
    else:
        left = paired.iloc[:, 0]
        right = paired.iloc[:, 1]
    corr = left.corr(right, method="pearson")
    if pd.isna(corr) or abs(corr) >= 1 or n <= 3:
        p_value = np.nan
    else:
        z = np.arctanh(corr) * math.sqrt(n - 3)
        p_value = math.erfc(abs(z) / math.sqrt(2))
    return float(corr), float(p_value) if not pd.isna(p_value) else np.nan, n


def significance(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def direction(corr):
    if pd.isna(corr):
        return ""
    return "positive" if corr >= 0 else "negative"


def main():
    wide = pd.read_excel(DATASET, sheet_name="Wide_Model_Data")
    input_cols = pd.read_excel(DATASET, sheet_name="Input_Columns")
    features = input_cols.loc[input_cols["role"] == "feature", "column"].tolist()
    y = pd.to_numeric(wide[TARGET], errors="coerce")

    rows = []
    for feature in features:
        pearson_r, pearson_p, pearson_n = corr_pair(wide[feature], y, "pearson")
        spearman_r, spearman_p, spearman_n = corr_pair(wide[feature], y, "spearman")
        feature_type, time_basis, window_stat, base_var = parse_feature_name(feature)
        rows.append(
            {
                "feature": feature,
                "base_variable": base_var,
                "base_variable_cn": VARIABLE_LABELS.get(base_var, base_var),
                "stage": stage_for_base(base_var),
                "feature_type": feature_type,
                "time_basis": time_basis,
                "window_stat": window_stat,
                "n": pearson_n,
                "pearson_r": pearson_r,
                "pearson_abs": abs(pearson_r) if not pd.isna(pearson_r) else np.nan,
                "pearson_p_value": pearson_p,
                "pearson_sig": significance(pearson_p),
                "pearson_direction": direction(pearson_r),
                "spearman_r": spearman_r,
                "spearman_abs": abs(spearman_r) if not pd.isna(spearman_r) else np.nan,
                "spearman_p_value": spearman_p,
                "spearman_sig": significance(spearman_p),
            }
        )

    corr = pd.DataFrame(rows).sort_values(["pearson_abs", "spearman_abs"], ascending=False).reset_index(drop=True)
    corr.insert(0, "rank_by_abs_pearson", np.arange(1, len(corr) + 1))

    top_abs = corr.head(40).copy()
    top_positive = corr.sort_values("pearson_r", ascending=False).head(25).reset_index(drop=True)
    top_negative = corr.sort_values("pearson_r", ascending=True).head(25).reset_index(drop=True)

    idx = corr.groupby("base_variable")["pearson_abs"].idxmax()
    best_by_base = corr.loc[idx].sort_values("pearson_abs", ascending=False).reset_index(drop=True)
    best_by_base.insert(0, "rank_base_by_abs_pearson", np.arange(1, len(best_by_base) + 1))

    stage_summary = (
        corr.groupby("stage", dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_pearson=("pearson_abs", "max"),
            mean_abs_pearson=("pearson_abs", "mean"),
            max_abs_spearman=("spearman_abs", "max"),
            mean_abs_spearman=("spearman_abs", "mean"),
        )
        .reset_index()
        .sort_values("max_abs_pearson", ascending=False)
    )

    type_summary = (
        corr.groupby(["feature_type", "time_basis", "window_stat"], dropna=False)
        .agg(
            feature_count=("feature", "count"),
            max_abs_pearson=("pearson_abs", "max"),
            mean_abs_pearson=("pearson_abs", "mean"),
            max_abs_spearman=("spearman_abs", "max"),
            mean_abs_spearman=("spearman_abs", "mean"),
        )
        .reset_index()
        .sort_values("max_abs_pearson", ascending=False)
    )

    target_summary = pd.DataFrame(
        [
            {"metric": "sample_count", "value": int(y.notna().sum()), "note": "有效 DEG 样本数"},
            {"metric": "target_min", "value": y.min(), "note": "二甘醇(%) 最小值"},
            {"metric": "target_max", "value": y.max(), "note": "二甘醇(%) 最大值"},
            {"metric": "target_mean", "value": y.mean(), "note": "二甘醇(%) 均值"},
            {"metric": "target_std", "value": y.std(ddof=0), "note": "二甘醇(%) 标准差"},
            {"metric": "feature_count", "value": len(features), "note": "默认过程输入特征数，不含 aux_* 化验项"},
            {"metric": "pearson_p_lt_0_05", "value": int((corr["pearson_p_value"] < 0.05).sum()), "note": "未做多重比较校正，仅作探索"},
            {"metric": "spearman_p_lt_0_05", "value": int((corr["spearman_p_value"] < 0.05).sum()), "note": "未做多重比较校正，仅作探索"},
        ]
    )

    readme = pd.DataFrame(
        [
            {"item": "分析对象", "detail": "Wide_Model_Data 中 role=feature 的 336 个过程特征与 target_DEG_pct 的单变量相关性。"},
            {"item": "Pearson", "detail": "衡量近似线性相关；pearson_r 为正表示特征升高时 DEG 倾向升高，负值相反。"},
            {"item": "Spearman", "detail": "基于排序的单调相关，对非线性但单调关系更稳健。"},
            {"item": "显著性", "detail": "* p<0.05，** p<0.01，*** p<0.001；p 值为 Fisher z 近似，当前样本只有 64 条且派生特征很多，仅作探索参考。"},
            {"item": "解读提醒", "detail": "强相关不等于因果；同一变量的当前值、滞后值和窗口统计高度相关，建模前建议做特征筛选或正则化。"},
            {"item": "建议", "detail": "优先关注 Best_By_Base_Variable 中每个基础变量的最佳派生特征，再结合工艺机理和模型验证集表现筛选。"},
        ]
    )

    payload = {
        "sheets": {
            "README": df_records(readme),
            "Target_Summary": df_records(target_summary),
            "Top_Abs_Correlations": df_records(top_abs),
            "Top_Positive": df_records(top_positive),
            "Top_Negative": df_records(top_negative),
            "Best_By_Base_Variable": df_records(best_by_base),
            "Stage_Summary": df_records(stage_summary),
            "Feature_Type_Summary": df_records(type_summary),
            "All_Correlations": df_records(corr),
        },
        "columns": {
            "README": list(readme.columns),
            "Target_Summary": list(target_summary.columns),
            "Top_Abs_Correlations": list(top_abs.columns),
            "Top_Positive": list(top_positive.columns),
            "Top_Negative": list(top_negative.columns),
            "Best_By_Base_Variable": list(best_by_base.columns),
            "Stage_Summary": list(stage_summary.columns),
            "Feature_Type_Summary": list(type_summary.columns),
            "All_Correlations": list(corr.columns),
        },
    }

    with (OUT / "deg_correlation_payload.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    compact = {
        "sample_count": int(y.notna().sum()),
        "feature_count": len(features),
        "top_abs": df_records(top_abs.head(10)),
        "best_by_base": df_records(best_by_base.head(10)),
        "payload": str(OUT / "deg_correlation_payload.json"),
    }
    with (OUT / "deg_correlation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
