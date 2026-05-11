import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(".");
const outDir = path.join(root, "outputs", "deg_nn_dataset");
const payloadPath = path.join(outDir, "deg_dataset_payload.json");
const outputPath = path.join(outDir, "DEG_NN_Dataset.xlsx");

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

function tableName(sheetName) {
  return `${sheetName.replace(/[^A-Za-z0-9]/g, "")}Table`.slice(0, 240);
}

function matrixFor(sheetName) {
  const columns = payload.columns[sheetName] || [];
  const rows = payload.sheets[sheetName] || [];
  return [columns, ...rows.map((row) => columns.map((col) => row[col] ?? null))];
}

const sheetNames = [
  "README",
  "Coverage_Report",
  "Wide_Model_Data",
  "Sequence_8h_Full",
  "Input_Columns",
  "Normalization_Params",
  "Sequence_Excluded",
];

for (const sheetName of sheetNames) {
  const sheet = workbook.worksheets.add(sheetName);
  const matrix = matrixFor(sheetName);
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
  header.format.rowHeightPx = 34;

  used.format.columnWidthPx = 120;
  if (sheetName === "README") {
    sheet.getRange("A:A").format.columnWidthPx = 120;
    sheet.getRange("B:B").format.columnWidthPx = 760;
    sheet.getRange("B:B").format.wrapText = true;
  } else if (sheetName === "Wide_Model_Data") {
    sheet.getRange("A:A").format.columnWidthPx = 170;
    sheet.getRange("B:B").format.columnWidthPx = 95;
    sheet.getRange("C:D").format.columnWidthPx = 150;
    sheet.getRange("E:F").format.columnWidthPx = 150;
  } else if (sheetName === "Sequence_8h_Full") {
    sheet.getRange("A:A").format.columnWidthPx = 170;
    sheet.getRange("B:C").format.columnWidthPx = 95;
    sheet.getRange("D:E").format.columnWidthPx = 150;
  } else if (sheetName === "Input_Columns") {
    sheet.getRange("A:A").format.columnWidthPx = 260;
    sheet.getRange("B:B").format.columnWidthPx = 170;
    sheet.getRange("C:C").format.columnWidthPx = 340;
    sheet.getRange("C:C").format.wrapText = true;
  } else if (sheetName === "Normalization_Params") {
    sheet.getRange("A:A").format.columnWidthPx = 260;
  } else if (sheetName === "Coverage_Report") {
    sheet.getRange("A:A").format.columnWidthPx = 180;
    sheet.getRange("B:B").format.columnWidthPx = 120;
    sheet.getRange("C:C").format.columnWidthPx = 420;
    sheet.getRange("C:C").format.wrapText = true;
  }

  if (rowCount > 1 && colCount > 1) {
    try {
      const lastCell = `${colLetter(colCount - 1)}${rowCount}`;
      const table = sheet.tables.add(`A1:${lastCell}`, true, tableName(sheetName));
      table.style = "TableStyleMedium2";
      table.showFilterButton = true;
    } catch {
      // Wide modeling sheets remain usable without Excel table metadata.
    }
  }
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
  const lastCol = colLetter(Math.min(Math.max(columns.length, 1), 8) - 1);
  const lastRow = Math.min(Math.max(rows.length + 1, 2), 20);
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
