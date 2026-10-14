// Converts the server's data files (~/exp_data/representation_stability/*.jsonl, downloaded with FileZilla) to CSV.
//
//   npm run data:csv -- <folder with .jsonl files> [output folder]
//
// Writes one <sessionId>.csv (or <sessionId>_INCOMPLETE.csv) per session, with exactly the
// columns of the pilot CSV, plus all_sessions.csv with every row of every session. The input
// folder is only read. The output folder defaults to <input folder>/csv.
import fs from "node:fs";
import path from "node:path";
import { CSV_COLUMNS } from "../src/services/dataStorage/csvSerializer.js";
import { sessionCsvFromJsonl } from "../src/services/dataStorage/serverRecords.js";

const [inputDir, outputArg] = process.argv.slice(2);
if (!inputDir) {
  console.error("Usage: npm run data:csv -- <folder with .jsonl files> [output folder]");
  process.exit(2);
}
const outputDir = outputArg ?? path.join(inputDir, "csv");
const files = fs.readdirSync(inputDir).filter((f) => f.endsWith(".jsonl")).sort();
if (!files.length) {
  console.error(`No .jsonl files in ${inputDir}`);
  process.exit(1);
}
fs.mkdirSync(outputDir, { recursive: true });

const header = `${CSV_COLUMNS.join(",")}\r\n`;
const combined = [header];
const counts = { complete: 0, incomplete: 0, skipped: 0 };
for (const file of files) {
  const session = sessionCsvFromJsonl(fs.readFileSync(path.join(inputDir, file), "utf8"));
  for (const warning of session.warnings) console.warn(`${file}: ${warning}`);
  if (!session.csv) {
    counts.skipped += 1;
    continue;
  }
  fs.writeFileSync(path.join(outputDir, session.name), session.csv);
  combined.push(session.csv.slice(header.length)); // rows may contain quoted line breaks, so no splitting
  counts[session.incomplete ? "incomplete" : "complete"] += 1;
  console.log(`${session.name}  ${session.trials.length} trials  ${session.meta.completionStatus}  participant ${session.meta.participantId}`);
}
fs.writeFileSync(path.join(outputDir, "all_sessions.csv"), combined.join(""));
console.log(`\n${counts.complete} complete, ${counts.incomplete} incomplete, ${counts.skipped} skipped -> ${outputDir}`);
