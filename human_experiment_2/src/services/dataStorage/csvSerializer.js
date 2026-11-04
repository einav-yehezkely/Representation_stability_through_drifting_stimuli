// CSV serialization. One row per answered trial; session metadata is repeated on
// every row as explicit columns; nested values are serialized as deterministic JSON.

const stimulusColumns = (prefix) => ["ImageId", "ImageIndex", "Pc1", "Pc2", "Angle", "DistanceToCenter"].map((field) => `${prefix}${field}`);

// Trial record, in fixed order.
export const TRIAL_COLUMNS = Object.freeze([
  "participantId",
  "sessionId",
  "trialNumber",
  "sampleGroup",
  "sampleSourceTrial",
  ...stimulusColumns("sample"),
  ...stimulusColumns("optionA"),
  ...stimulusColumns("optionB"),
  "leftGroup",
  "currentAngleA",
  "currentAngleB",
  "response",
  "chosenGroup",
  "chosenImageId",
  "correct",
  "responseTimeMs",
  "sampleDurationMs",
  "blankDurationMs",
  "interTrialDurationMs",
  "timestamp",
]);

// Additional per-trial reconstruction columns.
export const EXTRA_TRIAL_COLUMNS = Object.freeze([
  "selectionRule",
  "centerA_pc1",
  "centerA_pc2",
  "centerA_actualAngle",
  "centerB_pc1",
  "centerB_pc2",
  "centerB_actualAngle",
  "imageLoadFailures",
]);

// Session metadata repeated on every row.
export const METADATA_COLUMNS = Object.freeze([
  "experimentVersion",
  "schemaVersion",
  "participantIdSource",
  "sessionStartTime",
  "sessionEndTime",
  "consentGiven",
  "consentTimestamp",
  "consentBypassed",
  "developmentMode",
  "sessionSeed",
  "rngAlgorithm",
  "rngAlgorithmVersion",
  "pcaSource",
  "pcaSha256",
  "facesListSha256",
  "completedTrials",
  "completionStatus",
  "terminationReason",
  "exportedAsIncomplete",
  "initialCentersJson",
  "effectiveConfigJson",
  "developmentOverridesJson",
  "randomizationJson",
]);

export const CSV_COLUMNS = Object.freeze([...TRIAL_COLUMNS, ...EXTRA_TRIAL_COLUMNS, ...METADATA_COLUMNS]);

/** JSON with object keys sorted recursively (deterministic). */
export function stableStringify(value) {
  if (value === undefined) return "";
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((v) => (v === undefined ? "null" : stableStringify(v))).join(",")}]`;
  const keys = Object.keys(value).filter((k) => value[k] !== undefined).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(",")}}`;
}

/** RFC 4180 field encoding; booleans as true/false, missing values as empty fields. */
export function encodeField(value) {
  if (value === null || value === undefined) return "";
  let text;
  if (typeof value === "boolean") text = value ? "true" : "false";
  else if (typeof value === "number") text = Number.isFinite(value) ? String(value) : "";
  else text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function metadataValues(meta, trials, { incomplete }) {
  const status = meta.completionStatus;
  // The trial count is derived from the saved records, so it is also correct for interrupted
  // sessions whose final metadata was never written.
  return {
    experimentVersion: meta.experimentVersion,
    schemaVersion: meta.schemaVersion,
    participantIdSource: meta.participantIdSource,
    sessionStartTime: meta.sessionStartTime,
    sessionEndTime: meta.sessionEndTime,
    consentGiven: meta.consentGiven,
    consentTimestamp: meta.consentTimestamp,
    consentBypassed: meta.consentBypassed,
    developmentMode: meta.developmentMode,
    sessionSeed: meta.sessionSeed,
    rngAlgorithm: meta.rngAlgorithm,
    rngAlgorithmVersion: meta.rngAlgorithmVersion,
    pcaSource: meta.pcaSource,
    pcaSha256: meta.pcaSha256,
    facesListSha256: meta.facesListSha256,
    completedTrials: trials.length,
    completionStatus: status,
    terminationReason: meta.terminationReason,
    exportedAsIncomplete: Boolean(incomplete) || status !== "complete",
    initialCentersJson: stableStringify(meta.initialCenters ?? null),
    effectiveConfigJson: stableStringify(meta.effectiveConfig ?? null),
    developmentOverridesJson: stableStringify(meta.developmentOverrides ?? {}),
    randomizationJson: stableStringify(meta.randomization ?? null),
  };
}

/** Serialize a session (metadata + ordered trial records) to CSV text. */
export function serializeSessionCsv(meta, trials, { incomplete = false } = {}) {
  const shared = metadataValues(meta, trials, { incomplete });
  const lines = [CSV_COLUMNS.join(",")];
  const ordered = [...trials].sort((a, b) => a.trialNumber - b.trialNumber);
  for (const trial of ordered) {
    const row = { ...trial, ...shared };
    lines.push(CSV_COLUMNS.map((column) => encodeField(row[column])).join(","));
  }
  return `${lines.join("\r\n")}\r\n`;
}

export function csvFileName(sessionId, { incomplete = false } = {}) {
  return incomplete ? `${sessionId}_INCOMPLETE.csv` : `${sessionId}.csv`;
}

/** Minimal RFC 4180 parser (used by tests and the researcher review page). */
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += ch;
  }
  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }
  if (!rows.length) return [];
  const [header, ...body] = rows;
  return body.map((values) => Object.fromEntries(header.map((name, i) => [name, values[i] ?? ""])));
}
