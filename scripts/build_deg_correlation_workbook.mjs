import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(".");
const outDir = path.join(root, "outputs", "deg_correlation_analysis");
const payloadPath = path.join(outDir, "deg_correlation_payload.json");
const outputPath = path.join(outDir, "DEG_Correlation_Analysis.xlsx");

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

function matrixFor(sheetName) {
  const columns = payload.columns[sheetName] || [];
  const rows = payload.sheets[sheetName] || [];
  return [columns, ...rows.map((row) => columns.map((col) => row[col] ?? null))];
}

function styleNumericColumns(sheet, columns, rowCount) {
  const percentish = new Set(["pearson_r", "pearson_abs", "spearman_r", "spearman_abs"]);
  for (let i = 0; i < columns.length; i += 1) {
    const col = columns[i];
    const letter = colLetter(i);
    if (["pearson_p_value", "spearman_p_value"].includes(col)) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.numberFormat = "0.0000";
    } else if (percentish.has(col) || col.startsWith("max_abs_") || col.startsWith("mean_abs_")) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.numberFormat = "0.000";
    } else if (["value"].includes(col)) {
      sheet.getRange(`${letter}2:${letter}${rowCount}`).format.numberFormat = "0.000";
    }
  }
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
    // Some large sheets are still fully readable without table metadata.
  }
}

const sheetNames = [
  "README",
  "Target_Summary",
  "Top_Abs_Correlations",
  "Top_Positive",
  "Top_Negative",
  "Best_By_Base_Variable",
  "Stage_Summary",
  "Feature_Type_Summary",
  "All_Correlations",
];

for (const sheetName of sheetNames) {
  const sheet = workbook.worksheets.add(sheetName);
  const matrix = matrixFor(sheetName);
  const columns = payload.columns[sheetName] || [];
  const rowCount = Math.max(matrix.length, 1);
  const colCount = Math.max(matrix[0]?.length ?? 1, 1);
  sheet.getRangeByIndexes(0, 0, rowCount, colCount).values = matrix;
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;

  const used = sheet.getRangeByIndexes(0, 0, rowCount, colCount);
  used.format = {
    font: { name: "Aptos", size: 10, color: "#1F2937" },
    verticalAlignment: "top",
  };

  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  header.format = {
    fill: "#1F4E79",
    font: { bold: true, color: "#FFFFFF", name: "Aptos", size: 10 },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "middle",
  };
  header.format.rowHeightPx = 36;
  used.format.columnWidthPx = 120;

  if (columns.includes("feature")) {
    const idx = columns.indexOf("feature");
    sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.columnWidthPx = 260;
  }
  if (columns.includes("base_variable_cn")) {
    const idx = columns.indexOf("base_variable_cn");
    sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.columnWidthPx = 150;
  }
  if (columns.includes("detail")) {
    const idx = columns.indexOf("detail");
    sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.columnWidthPx = 760;
    sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.wrapText = true;
  }
  if (columns.includes("note")) {
    const idx = columns.indexOf("note");
    sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.columnWidthPx = 360;
    sheet.getRange(`${colLetter(idx)}:${colLetter(idx)}`).format.wrapText = true;
  }

  styleNumericColumns(sheet, columns, rowCount);
  addTable(sheet, sheetName, rowCount, colCount);
}

const topSheet = workbook.worksheets.getItem("Top_Abs_Correlations");
topSheet.getRange("A1:R1").format.fill = "#0F766E";

const stageSheet = workbook.worksheets.getItem("Stage_Summary");
const stageRows = (payload.sheets.Stage_Summary || []).length + 1;
if (stageRows > 1) {
  const chart = stageSheet.charts.add("bar", stageSheet.getRange(`A1:C${stageRows}`));
  chart.title = "Stage-Level Maximum Absolute Pearson Correlation";
  chart.hasLegend = true;
  chart.setPosition("H2", "N18");
}

const typeSheet = workbook.worksheets.getItem("Feature_Type_Summary");
const typeRows = Math.min((payload.sheets.Feature_Type_Summary || []).length + 1, 18);
if (typeRows > 1) {
  const chart = typeSheet.charts.add("bar", typeSheet.getRange(`B1:D${typeRows}`));
  chart.title = "Feature Type Correlation Strength";
  chart.hasLegend = true;
  chart.setPosition("I2", "P18");
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
  const lastCol = colLetter(Math.min(Math.max(columns.length, 1), 9) - 1);
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
