<?php
// Shared server configuration for save.php and admin_api.php. Produces no output when requested directly.
//
// The app lives in ~/apps/<app>/, so two levels up is the home directory. Everything below is
// outside the web root, so none of it can be downloaded from the site.

declare(strict_types=1);

const DATA_ROOT = __DIR__ . '/../../exp_data';
// One <sessionId>.jsonl file per session.
const DATA_DIR = DATA_ROOT . '/representation_stability';
// password_hash() of the admin password; see README "Admin page". No file: admin login is disabled.
const ADMIN_PASSWORD_HASH_FILE = DATA_ROOT . '/representation_stability_admin_password';
// Experiment settings saved on the admin page (src/config/experimentSettings.js). No file: defaults.
const SETTINGS_FILE = DATA_ROOT . '/representation_stability_settings.json';
// Failed admin logins per IP address (brute-force protection).
const ADMIN_ATTEMPTS_FILE = DATA_ROOT . '/representation_stability_admin_attempts.json';

const SESSION_ID_PATTERN = '/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/';

function json_reply(int $status, array $body): void
{
    http_response_code($status);
    echo json_encode($body, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION);
    exit;
}

function json_fail(int $status, string $error): void
{
    json_reply($status, ['ok' => false, 'error' => $error]);
}

/** The saved experiment settings ({values, updatedAt}), or null when none were saved. */
function read_settings(): ?array
{
    if (!is_file(SETTINGS_FILE)) return null;
    $settings = json_decode((string) file_get_contents(SETTINGS_FILE), true);
    if (!is_array($settings) || !is_array($settings['values'] ?? null)) json_fail(500, 'the settings file is unreadable');
    return $settings;
}
