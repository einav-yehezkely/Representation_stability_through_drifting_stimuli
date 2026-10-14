// Reads the per-session JSON-lines files written by public/save.php back into the same
// { meta, trials } shape the other adapters keep, so the CSV comes from csvSerializer.js.

import { csvFileName, serializeSessionCsv } from "./csvSerializer.js";
import { trialKey } from "./storageContract.js";

/**
 * @param {string} text  Contents of one <sessionId>.jsonl file.
 * @returns {{ sessionId: string|null, meta: object|null, trials: object[], incomplete: boolean, warnings: string[] }}
 */
export function sessionFromJsonl(text) {
  const warnings = [];
  let meta = null;
  let completion = null;
  let interrupted = false;
  const trials = new Map();

  text.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    let record;
    try {
      record = JSON.parse(line);
    } catch {
      warnings.push(`line ${index + 1}: not valid JSON (skipped)`);
      return;
    }
    if (record.type === "start") meta = { ...record.data };
    else if (record.type === "trial") {
      const key = trialKey(record.data);
      if (!trials.has(key)) trials.set(key, record.data);
    } else if (record.type === "complete") completion = record.data;
    else if (record.type === "interrupted") interrupted = true;
    else warnings.push(`line ${index + 1}: unknown record type "${record.type}" (skipped)`);
  });

  if (!meta) {
    warnings.push("no start record");
    return { sessionId: null, meta: null, trials: [...trials.values()], incomplete: true, warnings };
  }
  if (completion) Object.assign(meta, completion);
  else {
    // No completion record: the participant left, reloaded, or lost the connection.
    meta.completionStatus = "interrupted";
    meta.terminationReason = interrupted ? "page_reloaded_or_closed_before_completion" : "no_completion_received";
  }
  const ordered = [...trials.values()].sort((a, b) => a.trialNumber - b.trialNumber);
  return { sessionId: meta.sessionId, meta, trials: ordered, incomplete: meta.completionStatus !== "complete", warnings };
}

/** CSV file name and text for one session file. */
export function sessionCsvFromJsonl(text) {
  const session = sessionFromJsonl(text);
  if (!session.meta) return { ...session, name: null, csv: null };
  return {
    ...session,
    name: csvFileName(session.sessionId, { incomplete: session.incomplete }),
    csv: serializeSessionCsv(session.meta, session.trials, { incomplete: session.incomplete }),
  };
}
