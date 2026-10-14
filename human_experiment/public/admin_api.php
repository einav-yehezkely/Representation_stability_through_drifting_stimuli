<?php
// Password-protected access for the researcher admin page (admin.html, src/admin/). Session
// files are only read, never changed or deleted. The one thing this endpoint writes is the
// experiment settings file (SETTINGS_FILE).
//
// GET  admin_api.php?action=status          { ok, authenticated, configured }
// POST { action: "login", password }        { ok } | 401 wrong password | 429 locked | 503 not set up
// POST { action: "logout" }                 { ok }
// GET  admin_api.php?action=sessions        { ok, serverTime, sessions: [summary, ...] }   (login required)
// GET  admin_api.php?action=session&id=ID   { ok, sessionId, jsonl, lastActivity }        (login required)
// GET  admin_api.php?action=settings        { ok, settings: {values, updatedAt} | null }  (login required)
// POST { action: "save_settings", values }  { ok, settings }                              (login required)
//
// The password is stored only as a password_hash() in ADMIN_PASSWORD_HASH_FILE, outside the web
// root (see README "Admin page"). Logins use a PHP session cookie (HttpOnly, SameSite=Strict,
// Secure over HTTPS). After MAX_FAILED_LOGINS failed attempts an IP address is locked out for
// LOCKOUT_SECONDS.

declare(strict_types=1);

require __DIR__ . '/data_config.php';

const MAX_FAILED_LOGINS = 5;
const LOCKOUT_SECONDS = 900;
const IDLE_TIMEOUT_SECONDS = 7200;
const ABSOLUTE_TIMEOUT_SECONDS = 43200;
// A session without a completion record counts as in progress while its file changed this recently.
const ACTIVE_WINDOW_SECONDS = 1800;

header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');
header('X-Content-Type-Options: nosniff');
header('X-Frame-Options: DENY');
header('Referrer-Policy: no-referrer');

function iso_time(int $timestamp): string
{
    return gmdate('Y-m-d\TH:i:s\Z', $timestamp);
}

// ---------------------------------------------------------------- authentication

function start_admin_session(): void
{
    $https = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off')
        || (($_SERVER['HTTP_X_FORWARDED_PROTO'] ?? '') === 'https');
    $path = rtrim(str_replace('\\', '/', dirname($_SERVER['SCRIPT_NAME'] ?? '/')), '/') . '/';
    session_name('hx_admin');
    session_set_cookie_params(['lifetime' => 0, 'path' => $path, 'secure' => $https, 'httponly' => true, 'samesite' => 'Strict']);
    ini_set('session.use_strict_mode', '1');
    ini_set('session.use_only_cookies', '1');
    session_start();
}

function is_authenticated(): bool
{
    $now = time();
    $since = $_SESSION['admin_since'] ?? null;
    $last = $_SESSION['admin_last'] ?? null;
    if (!is_int($since) || !is_int($last)) return false;
    if ($now - $last > IDLE_TIMEOUT_SECONDS || $now - $since > ABSOLUTE_TIMEOUT_SECONDS) {
        $_SESSION = [];
        return false;
    }
    $_SESSION['admin_last'] = $now;
    return true;
}

function client_key(): string
{
    return hash('sha256', $_SERVER['REMOTE_ADDR'] ?? 'unknown');
}

/** Run $update(array $attempts): array on the attempts file under an exclusive lock. */
function with_attempts(callable $update)
{
    $handle = @fopen(ADMIN_ATTEMPTS_FILE, 'c+');
    if ($handle === false) json_fail(500, 'the login attempts file could not be opened');
    try {
        flock($handle, LOCK_EX);
        $attempts = json_decode((string) stream_get_contents($handle), true);
        if (!is_array($attempts)) $attempts = [];
        $now = time();
        foreach ($attempts as $key => $entry) {
            if (($entry['lockedUntil'] ?? 0) < $now && ($entry['last'] ?? 0) < $now - LOCKOUT_SECONDS) unset($attempts[$key]);
        }
        [$attempts, $result] = $update($attempts, $now);
        ftruncate($handle, 0);
        rewind($handle);
        fwrite($handle, json_encode($attempts));
        fflush($handle);
        flock($handle, LOCK_UN);
        return $result;
    } finally {
        fclose($handle);
    }
}

function login(string $password): void
{
    if (!is_file(ADMIN_PASSWORD_HASH_FILE)) {
        json_fail(503, 'The admin password has not been set up on the server (see README, "Admin page").');
    }
    $hash = trim((string) file_get_contents(ADMIN_PASSWORD_HASH_FILE));
    $key = client_key();
    $lockedFor = with_attempts(function (array $attempts, int $now) use ($key) {
        $until = $attempts[$key]['lockedUntil'] ?? 0;
        return [$attempts, $until > $now ? $until - $now : 0];
    });
    if ($lockedFor > 0) json_fail(429, 'Too many failed attempts. Try again in ' . (int) ceil($lockedFor / 60) . ' minutes.');

    if ($hash === '' || !password_verify($password, $hash)) {
        usleep(600000); // slows down guessing
        with_attempts(function (array $attempts, int $now) use ($key) {
            $count = ($attempts[$key]['count'] ?? 0) + 1;
            $attempts[$key] = ['count' => $count, 'last' => $now, 'lockedUntil' => $count >= MAX_FAILED_LOGINS ? $now + LOCKOUT_SECONDS : 0];
            return [$attempts, null];
        });
        json_fail(401, 'Incorrect password.');
    }
    with_attempts(function (array $attempts) use ($key) {
        unset($attempts[$key]);
        return [$attempts, null];
    });
    session_regenerate_id(true);
    $_SESSION['admin_since'] = time();
    $_SESSION['admin_last'] = time();
    json_reply(200, ['ok' => true]);
}

// ---------------------------------------------------------------- data

/** Summary of one session file, for the session list. */
function summarize_session(string $path): ?array
{
    $handle = fopen($path, 'r');
    if ($handle === false) return null;
    $meta = null;
    $completion = null;
    $interrupted = false;
    $seen = [];
    $counts = ['training' => 0, 'drift' => 0];
    $correct = ['training' => 0, 'drift' => 0];
    $criterionMetAt = null;
    $lastWindowAccuracy = null;
    $rtSum = 0.0;
    $lastTrial = null;
    while (($line = fgets($handle)) !== false) {
        $record = json_decode($line, true);
        if (!is_array($record)) continue;
        $type = $record['type'] ?? '';
        $data = is_array($record['data'] ?? null) ? $record['data'] : [];
        if ($type === 'start') {
            $meta = $data;
        } elseif ($type === 'trial') {
            $key = (string) ($record['trialKey'] ?? '');
            if (isset($seen[$key])) continue;
            $seen[$key] = true;
            $phase = ($data['phase'] ?? '') === 'drift' ? 'drift' : 'training';
            $counts[$phase] += 1;
            if (($data['correct'] ?? false) === true) $correct[$phase] += 1;
            if ($phase === 'training') {
                $windowAccuracy = $data['windowAccuracy'] ?? null;
                if ($windowAccuracy !== null) $lastWindowAccuracy = $windowAccuracy;
                if ($criterionMetAt === null && ($data['criterionPassed'] ?? false) === true) $criterionMetAt = $data['trialNumber'] ?? null;
            }
            $rtSum += (float) ($data['responseTimeMs'] ?? 0);
            $lastTrial = max((int) ($data['trialNumber'] ?? 0), (int) $lastTrial);
        } elseif ($type === 'complete') {
            $completion = $data;
        } elseif ($type === 'interrupted') {
            $interrupted = true;
        }
    }
    fclose($handle);
    if ($meta === null) return null;

    $modified = (int) filemtime($path);
    if ($completion !== null) $status = (string) ($completion['completionStatus'] ?? 'complete');
    elseif (!$interrupted && time() - $modified < ACTIVE_WINDOW_SECONDS) $status = 'in_progress';
    else $status = 'interrupted';
    $answered = $counts['training'] + $counts['drift'];
    $config = is_array($meta['effectiveConfig'] ?? null) ? $meta['effectiveConfig'] : [];
    return [
        'sessionId' => $meta['sessionId'] ?? basename($path, '.jsonl'),
        'participantId' => $meta['participantId'] ?? null,
        'participantIdSource' => $meta['participantIdSource'] ?? null,
        'developmentMode' => (bool) ($meta['developmentMode'] ?? false),
        'sessionSeed' => $meta['sessionSeed'] ?? null,
        'sessionStartTime' => $meta['sessionStartTime'] ?? null,
        'sessionEndTime' => $completion['sessionEndTime'] ?? null,
        'status' => $status,
        'terminationReason' => $completion['terminationReason'] ?? null,
        'trainingTrials' => $counts['training'],
        'trainingCorrect' => $correct['training'],
        'trainingWindowAccuracy' => $lastWindowAccuracy,
        'criterionMetAtTrial' => $criterionMetAt,
        'driftTrials' => $counts['drift'],
        'driftCorrect' => $correct['drift'],
        'driftTrialsPlanned' => $config['drift']['numberOfTrials'] ?? null,
        'meanResponseTimeMs' => $answered > 0 ? $rtSum / $answered : null,
        'lastTrialNumber' => $lastTrial,
        'lastActivity' => iso_time($modified),
    ];
}

function session_path_from_query(): string
{
    $id = (string) ($_GET['id'] ?? '');
    if (!preg_match(SESSION_ID_PATTERN, $id)) json_fail(400, 'invalid session id');
    $path = DATA_DIR . '/' . strtolower($id) . '.jsonl';
    if (!is_file($path)) json_fail(404, 'unknown session');
    return $path;
}

// ---------------------------------------------------------------- settings

const MAX_DRIFT_TRIALS = 11817; // number of PCA images (no repeats within drift)

/** Validate the settings sent by the admin page (the browser validates them too). */
function validated_settings($values): array
{
    if (!is_array($values)) json_fail(400, 'missing settings');
    $number = function (string $key) use ($values): float {
        $v = $values[$key] ?? null;
        if (!(is_int($v) || is_float($v)) || !is_finite((float) $v)) json_fail(400, "$key must be a number");
        return (float) $v;
    };
    $angleA = $number('training.initialAngleA');
    $angleB = $number('training.initialAngleB');
    $degrees = $number('drift.degreesPerTrial');
    $trials = $values['drift.numberOfTrials'] ?? null;
    $separation = fmod(fmod($angleB - $angleA, 360.0) + 360.0, 360.0);
    if (abs($separation - 180.0) > 1e-9) json_fail(400, 'center B must be 180° from center A');
    if (!($degrees > 0 && $degrees <= 360)) json_fail(400, 'degrees per trial must be greater than 0 and at most 360');
    if (!is_int($trials) || $trials < 1 || $trials > MAX_DRIFT_TRIALS) json_fail(400, 'the number of drift trials must be an integer between 1 and ' . MAX_DRIFT_TRIALS);
    return [
        'training.initialAngleA' => $values['training.initialAngleA'],
        'training.initialAngleB' => $values['training.initialAngleB'],
        'drift.degreesPerTrial' => $values['drift.degreesPerTrial'],
        'drift.numberOfTrials' => $trials,
    ];
}

function save_settings(array $values): array
{
    $settings = ['values' => $values, 'updatedAt' => iso_time(time())];
    $text = json_encode($settings, JSON_UNESCAPED_SLASHES | JSON_PRESERVE_ZERO_FRACTION | JSON_PRETTY_PRINT);
    // Write a temporary file and rename it, so a reader never sees a half-written file.
    $temporary = SETTINGS_FILE . '.tmp';
    if (file_put_contents($temporary, $text, LOCK_EX) === false || !rename($temporary, SETTINGS_FILE)) {
        json_fail(500, 'the settings could not be saved');
    }
    return $settings;
}

// ---------------------------------------------------------------- routing

start_admin_session();
$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';

if ($method === 'POST') {
    // JSON body only: a cross-site form cannot send it, and the cookie is SameSite=Strict.
    if (stripos($_SERVER['CONTENT_TYPE'] ?? '', 'application/json') !== 0) json_fail(415, 'JSON body required');
    $body = json_decode((string) file_get_contents('php://input', false, null, 0, 10000), true);
    if (!is_array($body)) json_fail(400, 'request is not a JSON object');
    $action = $body['action'] ?? '';
    if ($action === 'login') {
        $password = $body['password'] ?? null;
        if (!is_string($password) || $password === '' || strlen($password) > 1024) json_fail(400, 'missing password');
        login($password);
    }
    if ($action === 'save_settings') {
        if (!is_authenticated()) json_fail(401, 'login required');
        session_write_close();
        json_reply(200, ['ok' => true, 'settings' => save_settings(validated_settings($body['values'] ?? null))]);
    }
    if ($action === 'logout') {
        $_SESSION = [];
        session_destroy();
        json_reply(200, ['ok' => true]);
    }
    json_fail(400, 'unknown action');
}
if ($method !== 'GET') json_fail(405, 'GET or POST only');

$action = (string) ($_GET['action'] ?? '');
$authenticated = is_authenticated();
session_write_close(); // reads below do not need the session lock

if ($action === 'status') {
    json_reply(200, ['ok' => true, 'authenticated' => $authenticated, 'configured' => is_file(ADMIN_PASSWORD_HASH_FILE)]);
}
if (!$authenticated) json_fail(401, 'login required');

if ($action === 'settings') {
    json_reply(200, ['ok' => true, 'settings' => read_settings()]);
}
if ($action === 'sessions') {
    $sessions = [];
    foreach (glob(DATA_DIR . '/*.jsonl') ?: [] as $path) {
        $summary = summarize_session($path);
        if ($summary !== null) $sessions[] = $summary;
    }
    usort($sessions, fn ($a, $b) => strcmp((string) $b['sessionStartTime'], (string) $a['sessionStartTime']));
    json_reply(200, ['ok' => true, 'serverTime' => iso_time(time()), 'sessions' => $sessions]);
}
if ($action === 'session') {
    $path = session_path_from_query();
    $text = file_get_contents($path);
    if ($text === false) json_fail(500, 'session file could not be read');
    json_reply(200, ['ok' => true, 'sessionId' => basename($path, '.jsonl'), 'jsonl' => $text, 'lastActivity' => iso_time((int) filemtime($path))]);
}
json_fail(400, 'unknown action');
