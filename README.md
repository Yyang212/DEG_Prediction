# DEG Prediction

This project builds a modeling dataset for predicting diethylene glycol (DEG) content in a five-reactor polyester process.

The workflow includes:

- Cleaning and aligning process data with laboratory DEG measurements
- Building wide tabular features and 8-hour sequence features
- Running feature correlation analysis
- Training and evaluating a small-sample XGBoost model
- Plotting test-set prediction results

## Data Files

Raw data files are intentionally excluded from Git by default:

- `ZCP14.xlsx`
- `DEGResult.xlsx`
- `Polyester Melt Characteristic Viscosity Prediction Method Under Incomplete Data.pdf`

Place those files in the project root before running the scripts.

## Main Scripts

- `scripts/build_deg_dataset.py`  
  Builds the aligned DEG modeling dataset from process data and lab results.

- `scripts/build_deg_correlation_analysis.py`  
  Computes Pearson and Spearman correlations between process features and DEG.

- `scripts/run_deg_xgboost_analysis.py`  
  Runs small-sample XGBoost modeling and compares it with simple baselines.

- `scripts/draw_prediction_results.py`  
  Draws the test-set actual-vs-predicted result chart.

Some workbook export scripts use the Codex spreadsheet artifact runtime:

- `scripts/build_deg_workbook.mjs`
- `scripts/build_deg_correlation_workbook.mjs`
- `scripts/build_deg_xgboost_workbook.mjs`

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Then run the scripts from the repository root.

```bash
python scripts/build_deg_dataset.py
python scripts/build_deg_correlation_analysis.py
python scripts/run_deg_xgboost_analysis.py
python scripts/draw_prediction_results.py
```

## Current Modeling Note

The current aligned dataset contains only 64 DEG samples. This is useful for exploratory modeling and feature screening, but not enough for a robust production model. More historical process data and DEG lab results are needed, especially across stable, reduced-rate, transition, and abnormal operating conditions.

