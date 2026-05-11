import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(".");
const outDir = path.join(root, "outputs", "deg_xgboost_analysis");
const payloadPath = path.join(outDir, "deg_xgboost_payload.json");
const outputPath = path.join(outDir, "DEG_XGBoost_Small_Sample_Analysis.xlsx");

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();

function colLetter(n) {
  let s = "";
  let x = n + 1;
  while (x > 0) {
    const m = (x - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    x = Math.floor((x - 1) / 26);
  }
  return s;
}

function normalizeCell(value) {
  if (value === null || value === undefined) return null;
  if (Array.isArray(value)) return value.map(normalizeCell);
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return value;
}

function matrixFor(sheetName) {
  const columns = payload.columns[sheetName] || [];
  const rows = payload.sheets[sheetName] || [];
  return [columns, ...rows.map((row) => columns.map((col) => normalizeCell(row[col] ?? null)))];
}

function addTable(sheet, sheetName, rowCount, colCount) {
  if (rowCount <= 1 || colCount <= 1) return;
  try {
    const lastCell = `${colLetter(colCount - 1)}${rowCount}`;
    const tableName = `${sheetName.replace(/[^A-Za-z0-9]/g, "")}Table`.slice(0, 240);
    const table = sheet.tables.add(`A1:${lastCell}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showFilterButton = true;
  } catch {
    // Readability does not depend on table metadata.
  }
}

function styleCommon(sheet, columns) {
  const used = sheet.getUsedRange();
  if (!used) return;
  used.format = {
    font: { name: "Aptos", size: 10, color: "#1F2937" },
    verticalAlignment: "top",
  };
  const colCount = columns.length;
  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  header.format = {
    fill: "#1F4E79",
    font: { bold: true, color: "#FFFFFF", name: "Aptos", size: 10 },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "middle",
  };
  header.format.rowHeightPx = 36;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.columnWidthPx = 120;
}

const sheetNames = [
  "README",
  "Target_Summary",
  "Feature_Analysis",
  "Top_Correlations",
  "Model_Trials",
  "Best_Model",
  "Selected_Features",
  "Feature_Importance",
  "Stage_Summary",
  "Test_Predictions",
  "All_Predictions",
];

for (const sheetName of sheetNames) {
  const sheet = workbook.worksheets.add(sheetName);
  const matrix = matrixFor(sheetName);
  const columns = payload.columns[sheetName] || [];
  const rowCount = Math.max(matrix.length, 1);
  const colCount = Math.max(matrix[0]?.length ?? 1, 1);
  sheet.getRangeByIndexes(0, 0, rowCount, colCount).values = matrix;
  styleCommon(sheet, columns);
  addTable(sheet, sheetName, rowCount, colCount);

  const setWidth = (colName, px) => {
    const idx = columns.indexOf(colName);
    if (idx >= 0) sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.columnWidthPx = px;
  };

  if (sheetName === "README") {
    setWidth("item", 140);
    setWidth("detail", 900);
    sheet.getRange("B:B").format.wrapText = true;
  }
  if (sheetName === "Target_Summary") {
    setWidth("metric", 180);
    setWidth("note", 360);
  }
  if (sheetName === "Feature_Analysis" || sheetName === "Top_Correlations" || sheetName === "Selected_Features") {
    setWidth("feature", 250);
    setWidth("stage", 120);
    setWidth("base_variable", 140);
    setWidth("base_variable_cn", 150);
  }
  if (sheetName === "Model_Trials") {
    setWidth("model", 100);
    setWidth("selected_features", 460);
    setWidth("val_rmse", 100);
  }
  if (sheetName === "Best_Model") {
    setWidth("model", 100);
    setWidth("selected_features", 460);
  }
  if (sheetName === "Feature_Importance") {
    setWidth("feature", 280);
    setWidth("importance_gain", 120);
  }
  if (sheetName === "Test_Predictions" || sheetName === "All_Predictions") {
    setWidth("sample_id", 160);
    setWidth("sample_time", 170);
    setWidth("phase", 120);
    setWidth("predicted_DEG_pct", 130);
    setWidth("abs_error", 120);
  }
  if (sheetName === "Stage_Summary") {
    setWidth("stage", 130);
  }

  if (["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2", "mape_pct", "target_std", "train_target_mean", "val_target_mean", "test_target_mean", "pearson_r_train", "spearman_r_train", "abs_pearson", "abs_spearman", "importance_gain", "predicted_DEG_pct", "error", "abs_error"].some((c) => columns.includes(c))) {
    for (const col of ["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2", "mape_pct", "target_std", "train_target_mean", "val_target_mean", "test_target_mean", "pearson_r_train", "spearman_r_train", "abs_pearson", "abs_spearman", "importance_gain", "predicted_DEG_pct", "error", "abs_error"]) {
      const idx = columns.indexOf(col);
      if (idx >= 0) {
        const letter = colLetter(idx);
        sheet.getRange(`${letter}2:${letter}${rowCount}`).format.numberFormat = "0.0000";
      }
    }
  }
}

const featureSheet = workbook.worksheets.getItem("Feature_Importance");
const fiRows = Math.min((payload.sheets.Feature_Importance || []).length + 1, 16);
if (fiRows > 1) {
  const chart = featureSheet.charts.add("bar", featureSheet.getRange(`A1:B${fiRows}`));
  chart.title = "Top Feature Importance";
  chart.hasLegend = false;
  chart.setPosition("D2", "L18");
}

const predSheet = workbook.worksheets.getItem("Test_Predictions");
const predRows = (payload.sheets.Test_Predictions || []).length + 1;
if (predRows > 1) {
  const helper = predSheet.getRange(`F1:H${predRows}`);
  helper.values = [
    ["sample_time", "actual", "predicted"],
    ...payload.sheets.Test_Predictions.map((row) => [row.sample_time, row.target_DEG_pct, row.predicted_DEG_pct]),
  ];
  const chart = predSheet.charts.add("line", predSheet.getRange(`F1:H${predRows}`));
  chart.title = "Test Set Actual vs Predicted";
  chart.hasLegend = true;
  chart.xAxis = { axisType: "textAxis" };
  chart.yAxis = { numberFormatCode: "0.000" };
  chart.setPosition("J2", "U20");
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

for (const sheetName of sheetNames) {
  const columns = payload.columns[sheetName] || [];
  const rows = payload.sheets[sheetName] || [];
  const lastCol = colLetter(Math.min(Math.max(columns.length, 1), 10) - 1);
  const lastRow = Math.min(Math.max(rows.length + 1, 2), 24);
  const preview = await workbook.render({
    sheetName,
    range: `A1:${lastCol}${lastRow}`,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(outDir, `${sheetName}_preview.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath }, null, 2));
