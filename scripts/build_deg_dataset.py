from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "deg_nn_dataset"
OUT.mkdir(parents=True, exist_ok=True)

PROCESS_FILE = ROOT / "ZCP14.xlsx"
HISTORY_PROCESS_FILE = ROOT / "ZCP14历史数据.xlsx"
LAB_FILE = ROOT / "DEGResult.xlsx"

PROCESS_COLUMNS = [
    "process_time",
    "feed_flow",
    "molar_ratio",
    "ester1_temp",
    "ester1_pressure",
    "ester1_level",
    "ester2_temp",
    "ester2_pressure",
    "ester2_level",
    "prepoly1_temp",
    "prepoly1_pressure",
    "prepoly1_level",
    "prepoly2_temp",
    "prepoly2_pressure",
    "prepoly2_level",
    "final_pressure",
    "final_level",
]

VARIABLE_LABELS = {
    "feed_flow": "浆料配比_进料流量计",
    "molar_ratio": "浆料配比_摩尔比",
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

LAG_HOURS = [1, 2, 4, 8]
WINDOW_HOURS = [1, 2, 4, 8]
WINDOW_STATS = ["mean", "std", "min", "max"]
SEQ_HOURS = 8
FREQ_MINUTES = 5
SEQ_STEPS = SEQ_HOURS * 60 // FREQ_MINUTES + 1


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
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return value


def df_records(df: pd.DataFrame):
    return [
        {str(k): clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def classify_period_phase(t: pd.Timestamp) -> tuple[str, str]:
    if pd.Timestamp("2026-01-29 00:00:00") <= t <= pd.Timestamp("2026-02-28 23:55:00"):
        return "1.29-2.28 稳定", "stable"
    if pd.Timestamp("2026-03-21 00:00:00") <= t <= pd.Timestamp("2026-04-20 23:55:00"):
        return "3.21-4.20 减产", "reduced_rate"
    return "ZCP14历史数据", "historical"


def load_process_data() -> pd.DataFrame:
    frames = []
    source_files = [HISTORY_PROCESS_FILE] if HISTORY_PROCESS_FILE.exists() else [PROCESS_FILE]
    for source_file in source_files:
        xls = pd.ExcelFile(source_file)
        for sheet in xls.sheet_names:
            df = pd.read_excel(source_file, sheet_name=sheet, header=[0, 1])
            if df.shape[1] < len(PROCESS_COLUMNS):
                raise ValueError(f"{source_file.name}/{sheet} has {df.shape[1]} columns; expected at least {len(PROCESS_COLUMNS)}")
            # Keep the common 17 process columns. ZCP14历史数据.xlsx has an extra VI18020 column,
            # which is intentionally excluded so the feature definition stays consistent.
            df = df.iloc[:, : len(PROCESS_COLUMNS)].copy()
            df.columns = PROCESS_COLUMNS
            df["process_time"] = pd.to_datetime(df["process_time"], errors="coerce")
            for col in PROCESS_COLUMNS[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            classified = df["process_time"].apply(lambda t: classify_period_phase(t) if pd.notna(t) else (None, None))
            df["process_period"] = [item[0] for item in classified]
            df["phase"] = [item[1] for item in classified]
            frames.append(df.dropna(subset=["process_time"]))

    process = pd.concat(frames, ignore_index=True).sort_values("process_time")
    process = process.drop_duplicates(subset=["process_time"], keep="last")
    return process.reset_index(drop=True)


def load_lab_data() -> pd.DataFrame:
    lab = pd.read_excel(LAB_FILE, sheet_name="Sheet0", header=1)
    lab["sample_time"] = pd.to_datetime(lab["取样时间"], errors="coerce")
    lab["lab_time"] = pd.to_datetime(lab["化验时间"], errors="coerce")
    lab["target_DEG_pct"] = pd.to_numeric(lab["二甘醇(%)"], errors="coerce")
    lab["device"] = lab["装置号"].astype(str).str.strip()
    lab = lab[
        (lab["device"].str.upper() == "ZCP14")
        & lab["sample_time"].notna()
        & lab["target_DEG_pct"].notna()
    ].copy()
    lab = lab.rename(
        columns={
            "取样点": "sample_point",
            "测试类型": "test_type",
            "特性粘度(dl/g)": "aux_intrinsic_viscosity_dl_g",
            "端羧基含量(mol/t)": "aux_carboxyl_mol_t",
            "色值L": "aux_color_L",
            "色值B": "aux_color_B",
            "二氧化钛(%)": "aux_tio2_pct",
        }
    )
    keep = [
        "device",
        "sample_time",
        "lab_time",
        "sample_point",
        "test_type",
        "target_DEG_pct",
        "aux_intrinsic_viscosity_dl_g",
        "aux_carboxyl_mol_t",
        "aux_color_L",
        "aux_color_B",
        "aux_tio2_pct",
    ]
    for col in keep:
        if col not in lab.columns:
            lab[col] = np.nan
    return lab[keep].sort_values("sample_time").reset_index(drop=True)


def process_period_for_time(t: pd.Timestamp, ranges):
    for row in ranges:
        if row["start"] <= t <= row["end"]:
            return row["period"], row["phase"]
    return None, None


def nearest_row(process: pd.DataFrame, t: pd.Timestamp, tolerance_min=3):
    idx = pd.Index(process["process_time"]).get_indexer([t], method="nearest")[0]
    if idx < 0:
        return None, None
    row = process.iloc[idx]
    gap = abs((row["process_time"] - t).total_seconds()) / 60
    if gap > tolerance_min:
        return None, gap
    return row, gap


def build_wide_and_sequence(process: pd.DataFrame, lab: pd.DataFrame):
    variables = PROCESS_COLUMNS[1:]
    ranges = []
    for period, group in process.groupby("process_period", sort=False):
        ranges.append(
            {
                "period": period,
                "phase": group["phase"].iloc[0],
                "start": group["process_time"].min(),
                "end": group["process_time"].max(),
                "rows": len(group),
            }
        )

    process_indexed = process.set_index("process_time").sort_index()
    rows = []
    sequence_rows = []
    excluded = []

    for source_idx, lab_row in lab.iterrows():
        sample_time = lab_row["sample_time"]
        current_row, nearest_gap = nearest_row(process, sample_time)
        if current_row is None:
            continue
        period = current_row["process_period"]
        phase = current_row["phase"]

        sample_id = f"ZCP14_{sample_time.strftime('%Y%m%d_%H%M')}"
        base = {
            "sample_id": sample_id,
            "device": lab_row["device"],
            "sample_time": sample_time,
            "lab_time": lab_row["lab_time"],
            "process_period": period,
            "phase": phase,
            "sample_point": lab_row["sample_point"],
            "test_type": lab_row["test_type"],
            "target_DEG_pct": lab_row["target_DEG_pct"],
            "aux_intrinsic_viscosity_dl_g": lab_row["aux_intrinsic_viscosity_dl_g"],
            "aux_carboxyl_mol_t": lab_row["aux_carboxyl_mol_t"],
            "aux_color_L": lab_row["aux_color_L"],
            "aux_color_B": lab_row["aux_color_B"],
            "aux_tio2_pct": lab_row["aux_tio2_pct"],
        }

        base["nearest_process_gap_min"] = nearest_gap
        for var in variables:
            base[f"now_{var}"] = current_row[var]

        for lag_h in LAG_HOURS:
            target_time = sample_time - pd.Timedelta(hours=lag_h)
            lag_vals, lag_gap = nearest_row(process, target_time, tolerance_min=3)
            base[f"lag_{lag_h}h_gap_min"] = lag_gap
            for var in variables:
                base[f"lag_{lag_h}h_{var}"] = lag_vals[var] if lag_vals is not None else np.nan

        for win_h in WINDOW_HOURS:
            start_time = sample_time - pd.Timedelta(hours=win_h)
            window = process_indexed.loc[
                (process_indexed.index > start_time) & (process_indexed.index <= sample_time),
                variables,
            ]
            base[f"win_{win_h}h_count"] = int(len(window))
            for var in variables:
                series = window[var]
                base[f"win_{win_h}h_{var}_mean"] = series.mean()
                base[f"win_{win_h}h_{var}_std"] = series.std(ddof=0)
                base[f"win_{win_h}h_{var}_min"] = series.min()
                base[f"win_{win_h}h_{var}_max"] = series.max()

        rows.append(base)

        expected_times = pd.date_range(
            sample_time - pd.Timedelta(hours=SEQ_HOURS),
            sample_time,
            freq=f"{FREQ_MINUTES}min",
        )
        seq = process_indexed.reindex(expected_times)
        full_sequence = len(seq) == SEQ_STEPS and not seq[variables].isna().any().any()
        base["sequence_8h_full"] = full_sequence
        if full_sequence:
            for step_index, (ts, seq_row) in enumerate(seq.iterrows()):
                sequence_rows.append(
                    {
                        "sample_id": sample_id,
                        "step_index": step_index,
                        "minutes_before_sample": -SEQ_HOURS * 60 + step_index * FREQ_MINUTES,
                        "process_time": ts,
                        "sample_time": sample_time,
                        "target_DEG_pct": lab_row["target_DEG_pct"],
                        "phase": phase,
                        **{var: seq_row[var] for var in variables},
                    }
                )
        else:
            excluded.append(
                {
                    "sample_id": sample_id,
                    "sample_time": sample_time,
                    "reason": "insufficient 8h history inside provided process range",
                    "available_rows": int(seq[variables].dropna(how="any").shape[0]),
                    "required_rows": SEQ_STEPS,
                }
            )

    wide = pd.DataFrame(rows).sort_values("sample_time").reset_index(drop=True)
    if not wide.empty:
        n = len(wide)
        train_end = int(np.floor(n * 0.70))
        val_end = int(np.floor(n * 0.85))
        split = np.array(["test"] * n, dtype=object)
        split[:train_end] = "train"
        split[train_end:val_end] = "validation"
        wide.insert(1, "split_chrono", split)

    sequence = pd.DataFrame(sequence_rows)
    excluded_df = pd.DataFrame(excluded)
    return wide, sequence, excluded_df, ranges


def build_metadata(wide: pd.DataFrame, sequence: pd.DataFrame, excluded: pd.DataFrame, ranges):
    metadata_cols = {
        "sample_id",
        "split_chrono",
        "device",
        "sample_time",
        "lab_time",
        "process_period",
        "phase",
        "sample_point",
        "test_type",
        "target_DEG_pct",
        "aux_intrinsic_viscosity_dl_g",
        "aux_carboxyl_mol_t",
        "aux_color_L",
        "aux_color_B",
        "aux_tio2_pct",
        "nearest_process_gap_min",
        "sequence_8h_full",
    }
    coverage_cols = {c for c in wide.columns if c.endswith("_gap_min") or (c.startswith("win_") and c.endswith("_count"))}
    feature_cols = [c for c in wide.columns if c not in metadata_cols and c not in coverage_cols]

    input_rows = []
    for col in wide.columns:
        if col == "target_DEG_pct":
            role = "target"
        elif col in feature_cols:
            role = "feature"
        elif col.startswith("aux_"):
            role = "aux_lab_not_default_feature"
        elif col in coverage_cols:
            role = "coverage_qc"
        else:
            role = "metadata"
        source = ""
        if col.startswith("now_"):
            source = "nearest 5-min process value at sample_time"
        elif col.startswith("lag_"):
            source = "nearest 5-min process value at sample_time minus lag"
        elif col.startswith("win_"):
            source = "historical rolling window before sample_time"
        elif col == "target_DEG_pct":
            source = "DEGResult.xlsx 二甘醇(%)"
        input_rows.append({"column": col, "role": role, "source_or_note": source})

    train = wide[wide["split_chrono"] == "train"] if "split_chrono" in wide else wide
    norm_rows = []
    for col in feature_cols:
        series_all = pd.to_numeric(wide[col], errors="coerce")
        series_train = pd.to_numeric(train[col], errors="coerce")
        std = series_train.std(ddof=0)
        norm_rows.append(
            {
                "feature": col,
                "train_mean": series_train.mean(),
                "train_std": std,
                "std_for_scaling": 1.0 if pd.isna(std) or std == 0 else std,
                "all_min": series_all.min(),
                "all_max": series_all.max(),
                "missing_count_all": int(series_all.isna().sum()),
            }
        )

    coverage = pd.DataFrame(
        [
            {"metric": "matched_lab_samples", "value": len(wide), "note": "Rows with DEG target and sample_time inside supplied process data"},
            {"metric": "wide_feature_columns", "value": len(feature_cols), "note": "Use role=feature columns as default neural-network inputs"},
            {"metric": "sequence_full_samples", "value": wide["sequence_8h_full"].sum(), "note": f"Each full sequence has {SEQ_STEPS} rows at {FREQ_MINUTES}-min intervals"},
            {"metric": "sequence_rows", "value": len(sequence), "note": "Long-format sequence rows"},
            {"metric": "sequence_excluded_samples", "value": len(excluded), "note": "Usually samples too close to start of a supplied process period"},
            {"metric": "target_min_DEG_pct", "value": wide["target_DEG_pct"].min(), "note": ""},
            {"metric": "target_max_DEG_pct", "value": wide["target_DEG_pct"].max(), "note": ""},
            {"metric": "target_mean_DEG_pct", "value": wide["target_DEG_pct"].mean(), "note": ""},
        ]
    )
    period_rows = pd.DataFrame(
        [
            {
                "process_period": r["period"],
                "phase": r["phase"],
                "start": r["start"],
                "end": r["end"],
                "process_rows": r["rows"],
                "matched_samples": int((wide["process_period"] == r["period"]).sum()),
            }
            for r in ranges
        ]
    )

    readme = pd.DataFrame(
        [
            {"item": "用途", "detail": "用于 ZCP14 五釜聚酯工艺中二甘醇含量预测的数据集。"},
            {"item": "目标变量", "detail": "target_DEG_pct，对应 DEGResult.xlsx 中的 二甘醇(%)。"},
            {"item": "默认输入", "detail": "Input_Columns 表中 role=feature 的列；aux_* 化验项默认不要作为输入，避免目标泄漏。"},
            {"item": "宽表特征", "detail": "包含取样时刻最近值 now_*，1/2/4/8h 滞后值 lag_*，以及 1/2/4/8h 历史窗口均值、标准差、最小值、最大值。"},
            {"item": "序列特征", "detail": "Sequence_8h_Full 是每个样本取样前 8 小时、5 分钟间隔的长表，适合 LSTM/GRU/TCN；按 sample_id 分组。"},
            {"item": "划分建议", "detail": "split_chrono 是按时间顺序 70%/15%/15% 的训练/验证/测试提示，可按实际建模方案调整。"},
            {"item": "标准化", "detail": "Normalization_Params 使用训练集统计量计算，建模时建议按这些均值和标准差缩放输入特征。"},
            {"item": "样本提醒", "detail": f"当前过程数据覆盖 {len(wide)} 条有效 DEG 样本；样本量较之前增加，但训练神经网络仍建议继续积累更多历史批次。"},
        ]
    )

    return (
        readme,
        pd.DataFrame(input_rows),
        pd.DataFrame(norm_rows),
        coverage,
        period_rows,
    )


def main():
    process = load_process_data()
    lab = load_lab_data()
    wide, sequence, excluded, ranges = build_wide_and_sequence(process, lab)
    readme, input_cols, norm, coverage, period_rows = build_metadata(wide, sequence, excluded, ranges)

    payload = {
        "sheets": {
            "README": df_records(readme),
            "Coverage_Report": df_records(pd.concat([coverage, pd.DataFrame([{}]), period_rows], ignore_index=True)),
            "Wide_Model_Data": df_records(wide),
            "Sequence_8h_Full": df_records(sequence),
            "Input_Columns": df_records(input_cols),
            "Normalization_Params": df_records(norm),
            "Sequence_Excluded": df_records(excluded),
        },
        "columns": {
            "README": list(readme.columns),
            "Coverage_Report": list(pd.concat([coverage, pd.DataFrame([{}]), period_rows], ignore_index=True).columns),
            "Wide_Model_Data": list(wide.columns),
            "Sequence_8h_Full": list(sequence.columns),
            "Input_Columns": list(input_cols.columns),
            "Normalization_Params": list(norm.columns),
            "Sequence_Excluded": list(excluded.columns),
        },
    }
    with (OUT / "deg_dataset_payload.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    summary = {
        "wide_rows": len(wide),
        "wide_cols": wide.shape[1],
        "sequence_rows": len(sequence),
        "sequence_samples": int(sequence["sample_id"].nunique()) if not sequence.empty else 0,
        "excluded_sequences": len(excluded),
        "output_json": str(OUT / "deg_dataset_payload.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
