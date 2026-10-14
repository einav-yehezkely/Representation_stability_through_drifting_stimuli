<?php
// Data endpoint for the online experiment (src/services/dataStorage/serverStorage.js).
//
// POST JSON  { action: "start",       session: {sessionId, ...metadata} }
//            { action: "trial",       trial: {sessionId, trialNumber, ...} }
//            { action: "complete",    completion: {sessionId, ...} }
//            { action: "interrupted", sessionId }
// Each session is appended to DATA_DIR/<sessionId>.jsonl, one JSON object per line:
//   {"type":"start|trial|complete|interrupted","receivedAt":"...","trialKey":"...","data":{...}}
// Trials are idempotent on "<sessionId>:<trialNumber>" (a retry is acknowledged, not re-written).
// Convert the files to CSV with `npm run data:csv -- <folder with .jsonl files>`.
//
// GET save.php?check=1 reports whether DATA_DIR is writable (deployment check).
// GET save.php?settings=1 returns the experiment settings saved on the admin page
//     ({ ok, settings: {values, updatedAt} | null }); every new session applies them.

declare(strict_types=1);

// DATA_DIR (~/exp_data/representation_stability, outside the web root) and SESSION_ID_PATTERN.
require __DIR__ . '/data_config.php';

const MAX_BODY_BYTES = 1000000;

header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');
header('X-Content-Type-Options: nosniff');

function reply(int $status, array $body): void
{
    json_reply($status, $body);
}

function fail(int $status, string $error): void
{
    json_fail($status, $error);
}

function session_file(string $sessionId): string
{
    if (!preg_match(SESSION_ID_PATTERN, $sessionId)) fail(400, 'invalid sessionId');
    return DATA_DIR . '/' . strtolower($sessionId) . '.jsonl';
}

function encode_line(array $line): string
{
    $text = json_encode($line, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION);
    if ($text === false) fail(400, 'data could not be encoded');
    return $text . "\n";
}

/**
 * Open the session file under an exclusive lock and run $write($handle, $contents).
 * $mustExist: reply 404 when the session was never started.
 */
function with_session_file(string $path, bool $mustExist, callable $write): array
{
    if (!is_dir(DATA_DIR) || !is_writable(DATA_DIR)) fail(500, 'data directory is missing or not writable');
    if ($mustExist && !is_file($path)) fail(404, 'unknown session');
    $handle = fopen($path, 'c+');
    if ($handle === false) fail(500, 'session file could not be opened');
    try {
        if (!flock($handle, LOCK_EX)) fail(500, 'session file could not be locked');
        $contents = stream_get_contents($handle);
        if ($contents === false) fail(500, 'session file could not be read');
        fseek($handle, 0, SEEK_END);
        $result = $write($handle, $contents);
        fflush($handle);
        flock($handle, LOCK_UN);
        return $result;
    } finally {
        fclose($handle);
    }
}

function append($handle, string $text): void
{
    if (fwrite($handle, $text) !== strlen($text)) fail(500, 'session file could not be written');
}

$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
if ($method === 'GET' && isset($_GET['settings'])) {
    reply(200, ['ok' => true, 'settings' => read_settings()]);
}
if ($method === 'GET' && isset($_GET['check'])) {
    reply(200, ['ok' => true, 'dataDirWritable' => is_dir(DATA_DIR) && is_writable(DATA_DIR)]);
}
if ($method !== 'POST') fail(405, 'POST only');

$raw = file_get_contents('php://input', false, null, 0, MAX_BODY_BYTES + 1);
if ($raw === false || $raw === '') fail(400, 'empty request');
if (strlen($raw) > MAX_BODY_BYTES) fail(413, 'request too large');
$body = json_decode($raw, true);
if (!is_array($body)) fail(400, 'request is not a JSON object');

$now = gmdate('Y-m-d\TH:i:s\Z');
$action = $body['action'] ?? '';

switch ($action) {
    case 'start':
        $session = $body['session'] ?? null;
        if (!is_array($session) || !is_string($session['sessionId'] ?? null)) fail(400, 'missing session');
        $path = session_file($session['sessionId']);
        $result = with_session_file($path, false, function ($handle, string $contents) use ($session, $now) {
            // A retried start (the first reply was lost) must not fail or add a second header.
            if ($contents !== '') return ['ok' => true, 'duplicate' => true];
            append($handle, encode_line(['type' => 'start', 'receivedAt' => $now, 'data' => $session]));
            return ['ok' => true];
        });
        reply(200, $result);

    case 'trial':
        $trial = $body['trial'] ?? null;
        if (!is_array($trial) || !is_string($trial['sessionId'] ?? null)) fail(400, 'missing trial');
        $number = $trial['trialNumber'] ?? null;
        if (!is_int($number) || $number < 1) fail(400, 'invalid trialNumber');
        $path = session_file($trial['sessionId']);
        $key = strtolower($trial['sessionId']) . ':' . $number;
        $result = with_session_file($path, true, function ($handle, string $contents) use ($trial, $key, $now) {
            if (strpos($contents, '"trialKey":"' . $key . '"') !== false) return ['ok' => true, 'duplicate' => true];
            append($handle, encode_line(['type' => 'trial', 'receivedAt' => $now, 'trialKey' => $key, 'data' => $trial]));
            return ['ok' => true];
        });
        reply(200, $result);

    case 'complete':
        $completion = $body['completion'] ?? null;
        if (!is_array($completion) || !is_string($completion['sessionId'] ?? null)) fail(400, 'missing completion');
        $path = session_file($completion['sessionId']);
        $result = with_session_file($path, true, function ($handle) use ($completion, $now) {
            append($handle, encode_line(['type' => 'complete', 'receivedAt' => $now, 'data' => $completion]));
            return ['ok' => true];
        });
        reply(200, $result);

    case 'interrupted':
        $sessionId = $body['sessionId'] ?? null;
        if (!is_string($sessionId)) fail(400, 'missing sessionId');
        $path = session_file($sessionId);
        $result = with_session_file($path, true, function ($handle) use ($sessionId, $now) {
            append($handle, encode_line(['type' => 'interrupted', 'receivedAt' => $now, 'data' => ['sessionId' => $sessionId]]));
            return ['ok' => true];
        });
        reply(200, $result);

    default:
        fail(400, 'unknown action');
}
