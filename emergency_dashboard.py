"""
Her Routine - emergency backend + dashboard, packaged as a Flask Blueprint for the cjc app.

HOW TO CONNECT (the only changes to app.py - two lines):

    from emergency_dashboard import emergency_bp        # with the other imports
    app.register_blueprint(emergency_bp)                 # right after `app = Flask(__name__)`

Design rules, because this lives inside your live chatbot process:

* It can never take the chatbot down. Missing/weak settings or an unusable database file make
  ONLY the emergency routes answer 503; import never raises. (Same idea as require_admin_key.)
* It never touches the chatbot's Postgres / veronica_messages. It uses its own SQLite file.
* Its settings are all prefixed EMERGENCY_ so they cannot collide with the chatbot's.
* Security headers (CSP, no-store, X-Frame-Options ...) are attached ONLY to its own routes, so
  your widget.js / static pages / /predict responses are not changed.
* It does not need app.secret_key: the dashboard login is its own signed cookie.
* Location arrives ONLY after Emergency Mode was deliberately triggered on the phone. There is no
  endpoint that accepts routine tracking pings.

Routes it adds
    POST /api/emergency/{trigger,location,event,end,selftest}   phone -> server (HMAC-signed)
    GET  /emergency  /emergency/login  /emergency/dashboard     you (password login)
    POST /emergency/logout
    GET  /api/emergency/sessions, /api/emergency/sessions/<id>, /api/emergency/audit   (login needed)

Settings (environment variables)
    EMERGENCY_SECRET_KEY          required, >= 32 chars   signs the dashboard login cookie
    EMERGENCY_DASHBOARD_PASSWORD  required, >= 12 chars   what you type on the dashboard
    EMERGENCY_DEVICE_ID           required                a name for her phone, e.g. sanchari-phone
    EMERGENCY_DEVICE_SECRET       required, >= 32 chars   shared with her phone (typed into the app once)
    EMERGENCY_DB_PATH             optional                default /var/data/emergency.db if /var/data is
                                                          writable (persistent disk), else next to this file
    EMERGENCY_PROXY_HOPS          optional, default 0     number of trusted reverse proxies in front of the
                                                          app (1 on Render) so audit/throttle use real client IPs
    EMERGENCY_RETENTION_DAYS (30), EMERGENCY_AUDIT_RETENTION_DAYS (90)
    EMERGENCY_ALLOW_HTTP=1        local http testing only: drops the cookie's Secure flag

Signature scheme (identical to the Android app's RequestSigner.kt; both are tested against the same vectors):
    canonical   = METHOD \\n PATH \\n TIMESTAMP(epoch s) \\n NONCE \\n SHA256_HEX(body)
    X-Signature = HMAC_SHA256_HEX(DEVICE_SECRET, canonical)
    headers     : X-Device-Id, X-Timestamp, X-Nonce, X-Signature
"""
import hashlib
import hmac
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import wraps

from flask import Blueprint, g, jsonify, redirect, render_template, request, url_for
from itsdangerous import BadSignature, TimestampSigner, URLSafeTimedSerializer

__all__ = ["emergency_bp"]

log = logging.getLogger("emergency")

# --------------------------------------------------------------------------- configuration

_REQUIRED = {                      # name -> minimum length
    "EMERGENCY_SECRET_KEY": 32,
    "EMERGENCY_DASHBOARD_PASSWORD": 12,
    "EMERGENCY_DEVICE_ID": 1,
    "EMERGENCY_DEVICE_SECRET": 32,
}
_problems = []
for _name, _min in _REQUIRED.items():
    _value = os.environ.get(_name, "")
    if not _value:
        _problems.append(f"{_name} is not set")
    elif len(_value) < _min:
        _problems.append(f"{_name} is too short (needs at least {_min} characters)")

CONFIGURED = not _problems
if not CONFIGURED:
    log.error(
        "Emergency dashboard DISABLED (the chatbot is unaffected): %s. Generate secrets with: "
        "python -c \"import secrets; print(secrets.token_hex(32))\"", "; ".join(_problems))

SECRET_KEY = os.environ.get("EMERGENCY_SECRET_KEY", "")
DASHBOARD_PASSWORD = os.environ.get("EMERGENCY_DASHBOARD_PASSWORD", "")
DEVICE_ID = os.environ.get("EMERGENCY_DEVICE_ID", "")
DEVICE_SECRET = os.environ.get("EMERGENCY_DEVICE_SECRET", "")


def _int_env(name, default, low, high):
    try:
        return min(max(int(os.environ.get(name, default)), low), high)
    except (TypeError, ValueError):
        return default


def _default_db_path():
    persistent = "/var/data"                       # the disk start.sh already uses for the model
    if os.path.isdir(persistent) and os.access(persistent, os.W_OK):
        return os.path.join(persistent, "emergency.db")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergency.db")


DB_PATH = os.environ.get("EMERGENCY_DB_PATH") or _default_db_path()
RETENTION_DAYS = _int_env("EMERGENCY_RETENTION_DAYS", 30, 1, 3650)
AUDIT_RETENTION_DAYS = _int_env("EMERGENCY_AUDIT_RETENTION_DAYS", 90, 1, 3650)
PROXY_HOPS = _int_env("EMERGENCY_PROXY_HOPS", 0, 0, 5)
COOKIE_SECURE = os.environ.get("EMERGENCY_ALLOW_HTTP") != "1"

if CONFIGURED and os.path.dirname(os.path.abspath(DB_PATH)) == os.path.dirname(os.path.abspath(__file__)):
    log.warning("Emergency database is inside the app folder (%s); on most hosts that is wiped on every "
                "redeploy. Point EMERGENCY_DB_PATH at a persistent disk.", DB_PATH)

COOKIE_NAME = "emergency_auth"
SESSION_SECONDS = 12 * 3600

MAX_BODY_BYTES = 256 * 1024
MAX_CLOCK_SKEW_SECONDS = 300            # signed-request timestamp window (send time, so retries re-sign)
NONCE_TTL_SECONDS = 2 * MAX_CLOCK_SKEW_SECONDS + 60
LOCATION_MAX_FUTURE_SECONDS = 300
LOCATION_MAX_AGE_SECONDS = 30 * 86400   # older than this is not a plausible emergency position
STALE_AFTER_SECONDS = 120               # a position older than this when received is flagged "older"
MAX_BATCH_EVENTS = 100

LOGIN_MAX_FAILURES = 8
LOGIN_LOCKOUT_SECONDS = 300

TRIGGER_SOURCES = {"IN_APP_BUTTON", "QUICK_SETTINGS_TILE", "POWER_x4_SCREEN_HEURISTIC"}
LOCATION_SOURCES = {"CURRENT", "LAST_KNOWN_LOCATION", "UNAVAILABLE"}
CALL_STATUSES = {"NOT_ATTEMPTED", "INITIATED", "FAILED", "PERMISSION_MISSING", "DIALER_OPENED_MANUALLY", "DISABLED", "NO_NUMBER"}
SMS_STATUSES = {"NOT_ATTEMPTED", "QUEUED", "FAILED", "PERMISSION_MISSING", "DISABLED", "NO_TRUSTED_CONTACTS"}
END_STATUSES = {"ENDED", "EXPIRED"}

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def now() -> float:
    """Wall clock, in one place so tests can control time."""
    return time.time()


def iso(epoch_seconds):
    if epoch_seconds is None:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value):
    """ISO-8601 string or epoch number -> epoch seconds (float), or None if unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str) and value:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _same(a: str, b: str) -> bool:
    """Constant-time equality that cannot raise on non-ASCII input (compare_digest on str can)."""
    return hmac.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


emergency_bp = Blueprint("emergency", __name__, template_folder="templates")

# --------------------------------------------------------------------------- rate limiting

class SlidingWindowLimiter:
    """In-memory, per-process. Adequate for one phone + one viewer (your gunicorn runs one worker);
    with several worker processes each keeps its own counters (the replay/nonce table IS shared)."""

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key, limit, window_seconds):
        t = now()
        with self._lock:
            q = self._hits[key]
            while q and t - q[0] > window_seconds:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(t)
            return True

    def count(self, key, window_seconds):
        t = now()
        with self._lock:
            q = self._hits[key]
            while q and t - q[0] > window_seconds:
                q.popleft()
            return len(q)

    def add(self, key):
        with self._lock:
            self._hits[key].append(now())

    def clear(self, key):
        with self._lock:
            self._hits.pop(key, None)

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()

DEVICE_LIMITS = {          # requests per minute, per device
    "trigger": 6,
    "location": 30,
    "event": 20,
    "end": 10,
    "selftest": 10,
}
DASHBOARD_READS_PER_MINUTE = 120
AUTH_FAILURES_PER_MINUTE = 20

# --------------------------------------------------------------------------- database

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    server_session_id TEXT PRIMARY KEY,
    client_session_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    status TEXT NOT NULL,
    trigger_source TEXT,
    tracking_duration_minutes INTEGER,
    app_version TEXT,
    started_at REAL NOT NULL,
    received_at REAL NOT NULL,
    ended_at REAL,
    call_status TEXT NOT NULL DEFAULT 'NOT_ATTEMPTED',
    call_detail TEXT,
    sms_status TEXT NOT NULL DEFAULT 'NOT_ATTEMPTED',
    sms_detail TEXT,
    last_event_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_session_id TEXT NOT NULL,
    client_event_id TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    accuracy_m REAL,
    measured_at REAL NOT NULL,
    received_at REAL NOT NULL,
    provider TEXT,
    location_source TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    UNIQUE (server_session_id, client_event_id)
);
CREATE INDEX IF NOT EXISTS idx_locations_session ON locations (server_session_id, measured_at);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT,
    event_at REAL NOT NULL,
    received_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events (server_session_id, event_at);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    server_session_id TEXT,
    ip TEXT,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS seen_nonces (
    nonce TEXT PRIMARY KEY,
    at REAL NOT NULL
);
"""


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    try:                                   # emergency data may include positions: keep the file private
        os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


def db():
    if "emergency_db" not in g:
        g.emergency_db = _connect()
    return g.emergency_db


@emergency_bp.teardown_request
def _close_db(exception=None):
    conn = g.pop("emergency_db", None)
    if conn is not None:
        conn.close()


_last_purge = 0.0
_purge_lock = threading.Lock()


def purge_old(force=False):
    """Retention: emergency data is not kept forever. A still-ACTIVE emergency is never purged."""
    global _last_purge
    t = now()
    with _purge_lock:
        if not force and t - _last_purge < 3600:
            return
        _last_purge = t
    cutoff = t - RETENTION_DAYS * 86400
    conn = _connect()
    try:
        old = [r["server_session_id"] for r in conn.execute(
            "SELECT server_session_id FROM sessions WHERE started_at < ? AND status != 'ACTIVE'", (cutoff,))]
        for sid in old:
            conn.execute("DELETE FROM locations WHERE server_session_id = ?", (sid,))
            conn.execute("DELETE FROM events WHERE server_session_id = ?", (sid,))
            conn.execute("DELETE FROM sessions WHERE server_session_id = ?", (sid,))
        conn.execute("DELETE FROM audit WHERE at < ?", (t - AUDIT_RETENTION_DAYS * 86400,))
        conn.execute("DELETE FROM seen_nonces WHERE at < ?", (t - NONCE_TTL_SECONDS,))
        conn.commit()
    finally:
        conn.close()


_db_ready = False
_db_lock = threading.Lock()


def _ensure_db() -> bool:
    """Create the database on first use. A failure is logged and retried on the next request; it
    never propagates into the chatbot."""
    global _db_ready
    if _db_ready:
        return True
    with _db_lock:
        if _db_ready:
            return True
        try:
            init_db()
            purge_old(force=True)
            _db_ready = True
        except Exception:
            log.exception("Emergency database unavailable at %s", DB_PATH)
    return _db_ready


if CONFIGURED:
    _ensure_db()

# --------------------------------------------------------------------------- helpers

def client_ip():
    """The socket peer, or - if EMERGENCY_PROXY_HOPS trusted proxies sit in front - the address the
    outermost trusted proxy saw. (Counting from the right means a client cannot forge it.)"""
    if PROXY_HOPS:
        parts = [p.strip() for p in request.headers.get("X-Forwarded-For", "").split(",") if p.strip()]
        if len(parts) >= PROXY_HOPS:
            return parts[-PROXY_HOPS][:64]
    return (request.remote_addr or "unknown")[:64]


def audit(actor, action, server_session_id=None, detail=None):
    db().execute(
        "INSERT INTO audit (at, actor, action, server_session_id, ip, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (now(), actor, action, server_session_id, client_ip(), (detail or "")[:500] or None),
    )
    db().commit()


def error(status, code, **extra):
    body = {"error": code}
    body.update(extra)
    return jsonify(body), status


def int_arg(name, default, low, high):
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, low), high)


def _render(template, **context):
    return render_template(template, csp_nonce=g.csp_nonce, **context)


@emergency_bp.before_request
def _gate():
    g.csp_nonce = secrets.token_urlsafe(16)
    if not CONFIGURED:
        return error(503, "emergency_not_configured")
    if not _ensure_db():
        return error(503, "emergency_storage_unavailable")
    if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
        return error(413, "payload_too_large")
    try:
        purge_old()
    except Exception:
        log.exception("Emergency retention sweep failed")


@emergency_bp.after_request
def _security_headers(resp):
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    # "origin" sends only the site address (never the path) to other sites. OpenStreetMap's tile
    # servers block map tiles that arrive with no Referer at all, so "no-referrer" broke the map.
    resp.headers["Referrer-Policy"] = "origin"
    nonce = getattr(g, "csp_nonce", "")
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}' https://cdnjs.cloudflare.com; "
        f"style-src 'self' 'nonce-{nonce}' https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org https://cdnjs.cloudflare.com; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    return resp


# --------------------------------------------------------------------------- device auth

def canonical_string(method, path, timestamp, nonce, body_bytes):
    return f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{hashlib.sha256(body_bytes).hexdigest()}"


def compute_signature(secret, method, path, timestamp, nonce, body_bytes):
    return hmac.new(
        secret.encode("utf-8"),
        canonical_string(method, path, timestamp, nonce, body_bytes).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def device_signed(bucket):
    """Authenticate + authorise + rate-limit + replay-protect a request from the phone.

    Failed attempts are throttled per address, but a request carrying a VALID signature is never
    blocked by that throttle - so nobody can lock her phone out of reporting an emergency by
    spamming bad requests."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = client_ip()

            def reject(status, code, **extra):
                if limiter.count(("authfail", ip), 60) >= AUTH_FAILURES_PER_MINUTE:
                    return error(429, "too_many_failed_attempts")      # not audited: don't let noise fill the log
                limiter.add(("authfail", ip))
                audit(f"device:{(request.headers.get('X-Device-Id') or '?')[:32]}", "AUTH_FAILED", None, code)
                return error(status, code, **extra)

            device_id = request.headers.get("X-Device-Id", "")
            ts_header = request.headers.get("X-Timestamp", "")
            nonce = request.headers.get("X-Nonce", "")
            signature = request.headers.get("X-Signature", "")
            if not (device_id and ts_header and nonce and signature):
                return reject(401, "missing_auth_headers")
            if not _NONCE_RE.match(nonce):
                return reject(401, "bad_nonce")
            try:
                ts = int(ts_header)
            except ValueError:
                return reject(401, "bad_timestamp")

            body = request.get_data(cache=True)
            expected = compute_signature(DEVICE_SECRET, request.method, request.path, ts_header, nonce, body)
            # Same generic answer for "wrong device" and "wrong signature": no oracle for guessing.
            if not (_same(device_id, DEVICE_ID) and _same(signature, expected)):
                return reject(401, "unauthorized")
            if abs(now() - ts) > MAX_CLOCK_SKEW_SECONDS:
                return reject(401, "timestamp_out_of_range", server_time=int(now()))

            try:
                db().execute("INSERT INTO seen_nonces (nonce, at) VALUES (?, ?)", (nonce, now()))
                db().commit()
            except sqlite3.IntegrityError:
                return reject(401, "replayed_request")

            if not limiter.allow(("device", device_id, bucket), DEVICE_LIMITS[bucket], 60):
                resp = error(429, "rate_limited")
                resp[0].headers["Retry-After"] = "30"
                return resp

            g.device_actor = f"device:{device_id}"
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# --------------------------------------------------------------------------- validation

def json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def valid_user_id(value):
    return isinstance(value, str) and bool(_USER_RE.match(value))


def valid_client_session_id(value):
    return isinstance(value, str) and bool(_ID_RE.match(value))


def parse_position(obj):
    """Validate one position. Returns (position_dict, None) or (None, error_code)."""
    try:
        lat = float(obj.get("latitude"))
        lng = float(obj.get("longitude"))
    except (TypeError, ValueError):
        return None, "invalid_coordinates"
    if not (math.isfinite(lat) and math.isfinite(lng)) or not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        return None, "invalid_coordinates"
    if lat == 0.0 and lng == 0.0:
        return None, "invalid_coordinates"            # (0,0) is the classic "no fix" placeholder

    accuracy = obj.get("accuracy_meters")
    if accuracy is not None:
        try:
            accuracy = float(accuracy)
        except (TypeError, ValueError):
            return None, "invalid_accuracy"
        if not math.isfinite(accuracy) or accuracy < 0 or accuracy > 100000:
            return None, "invalid_accuracy"

    measured = parse_time(obj.get("timestamp"))
    if measured is None:
        return None, "invalid_timestamp"
    t = now()
    if measured > t + LOCATION_MAX_FUTURE_SECONDS or measured < t - LOCATION_MAX_AGE_SECONDS:
        return None, "invalid_timestamp"

    source = obj.get("location_source", "CURRENT")
    if source not in LOCATION_SOURCES or source == "UNAVAILABLE":
        return None, "invalid_location_source"

    provider = obj.get("provider")
    provider = provider[:32] if isinstance(provider, str) else None
    return {
        "lat": lat, "lng": lng, "accuracy": accuracy, "measured": measured,
        "source": source, "provider": provider,
        "stale": 1 if (t - measured) > STALE_AFTER_SECONDS else 0,
    }, None


def get_or_create_session(client_sid, user_id, started_at, trigger_source=None, duration=None, app_version=None):
    """Sessions are keyed by the phone's own id (idempotency) but identified everywhere else by a
    server-generated random id, so a phone can never choose or guess another session's id."""
    row = db().execute("SELECT * FROM sessions WHERE client_session_id = ?", (client_sid,)).fetchone()
    if row:
        return row, False
    server_sid = "EM-" + secrets.token_urlsafe(18)
    t = now()
    db().execute(
        "INSERT INTO sessions (server_session_id, client_session_id, user_id, device_id, status, trigger_source, "
        "tracking_duration_minutes, app_version, started_at, received_at, last_event_at) "
        "VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?)",
        (server_sid, client_sid, user_id, DEVICE_ID, trigger_source, duration, (app_version or "")[:32] or None, started_at, t, t),
    )
    db().commit()
    return db().execute("SELECT * FROM sessions WHERE server_session_id = ?", (server_sid,)).fetchone(), True


def add_event(server_sid, event_type, detail=None, event_at=None):
    t = now()
    db().execute(
        "INSERT INTO events (server_session_id, event_type, detail, event_at, received_at) VALUES (?, ?, ?, ?, ?)",
        (server_sid, event_type, (detail or "")[:500] or None, event_at if event_at is not None else t, t),
    )


def touch_session(server_sid):
    db().execute("UPDATE sessions SET last_event_at = ? WHERE server_session_id = ?", (now(), server_sid))


# --------------------------------------------------------------------------- device endpoints

@emergency_bp.route("/api/emergency/trigger", methods=["POST"])
@device_signed("trigger")
def emergency_trigger():
    data = json_body()
    if data is None:
        return error(400, "invalid_json")
    if data.get("event_type") != "EMERGENCY_TRIGGERED":
        return error(400, "invalid_event_type")
    client_sid = data.get("emergency_session_id")
    if not valid_client_session_id(client_sid):
        return error(400, "invalid_session_id")
    if not valid_user_id(data.get("user_id")):
        return error(400, "invalid_user_id")

    started = parse_time(data.get("timestamp"))
    t = now()
    if started is None or started > t + LOCATION_MAX_FUTURE_SECONDS or started < t - LOCATION_MAX_AGE_SECONDS:
        return error(400, "invalid_timestamp")

    trigger_source = data.get("trigger_source")
    if trigger_source not in TRIGGER_SOURCES:
        return error(400, "invalid_trigger_source")
    duration = data.get("tracking_duration_minutes")
    duration = duration if isinstance(duration, int) and not isinstance(duration, bool) and 1 <= duration <= 240 else None

    # Location is optional on the trigger (it may be unavailable) - but if present it must be valid.
    position = None
    location_unavailable = bool(data.get("location_unavailable"))
    if not location_unavailable and "latitude" in data:
        position, err = parse_position(data)
        if err:
            return error(400, err)

    session_row, created = get_or_create_session(
        client_sid, data["user_id"], started, trigger_source, duration, data.get("app_version"))
    sid = session_row["server_session_id"]

    if created:
        add_event(sid, "EMERGENCY_TRIGGERED", f"source={trigger_source}", started)
        if position:
            _insert_location(sid, "trigger", position)
        else:
            add_event(sid, "LOCATION_UNAVAILABLE", "No location was available when the emergency was triggered.")
        db().commit()
        audit(g.device_actor, "DEVICE_TRIGGER", sid, f"source={trigger_source}")

    has_location = db().execute("SELECT 1 FROM locations WHERE server_session_id = ? LIMIT 1", (sid,)).fetchone() is not None
    return jsonify({
        "ok": True,
        "server_session_id": sid,
        "status": session_row["status"],
        "received_at": iso(session_row["received_at"]),
        "location_received": has_location,
        "duplicate": not created,
    })


def _insert_location(server_sid, client_event_id, position):
    cur = db().execute(
        "INSERT OR IGNORE INTO locations (server_session_id, client_event_id, latitude, longitude, accuracy_m, "
        "measured_at, received_at, provider, location_source, stale) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (server_sid, client_event_id, position["lat"], position["lng"], position["accuracy"], position["measured"],
         now(), position["provider"], position["source"], position["stale"]),
    )
    return cur.rowcount


@emergency_bp.route("/api/emergency/location", methods=["POST"])
@device_signed("location")
def emergency_location():
    data = json_body()
    if data is None:
        return error(400, "invalid_json")
    client_sid = data.get("emergency_session_id")
    if not valid_client_session_id(client_sid):
        return error(400, "invalid_session_id")
    if not valid_user_id(data.get("user_id")):
        return error(400, "invalid_user_id")
    events = data.get("events")
    if not isinstance(events, list) or not (1 <= len(events) <= MAX_BATCH_EVENTS):
        return error(400, "invalid_events")

    accepted, duplicates, rejected = 0, 0, []
    parsed = []
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            rejected.append({"index": i, "error": "invalid_event"}); continue
        eid = ev.get("client_event_id")
        if not valid_client_session_id(eid):
            rejected.append({"index": i, "error": "invalid_event_id"}); continue
        if ev.get("event_type", "EMERGENCY_LOCATION_UPDATE") != "EMERGENCY_LOCATION_UPDATE":
            rejected.append({"index": i, "error": "invalid_event_type"}); continue
        position, err = parse_position(ev)
        if err:
            rejected.append({"index": i, "client_event_id": eid, "error": err}); continue
        parsed.append((eid, position))

    if not parsed:
        return error(400, "no_valid_events", rejected=rejected)

    first_measured = min(p["measured"] for _, p in parsed)
    session_row, _ = get_or_create_session(client_sid, data["user_id"], first_measured)
    sid = session_row["server_session_id"]
    for eid, position in parsed:
        if _insert_location(sid, eid, position):
            accepted += 1
        else:
            duplicates += 1
    touch_session(sid)
    db().commit()
    audit(g.device_actor, "DEVICE_LOCATION_BATCH", sid, f"accepted={accepted} duplicates={duplicates} rejected={len(rejected)}")
    return jsonify({"ok": True, "server_session_id": sid, "accepted": accepted, "duplicates": duplicates, "rejected": rejected})


@emergency_bp.route("/api/emergency/event", methods=["POST"])
@device_signed("event")
def emergency_event():
    data = json_body()
    if data is None:
        return error(400, "invalid_json")
    if data.get("event_type") != "STATUS_UPDATE":
        return error(400, "invalid_event_type")
    client_sid = data.get("emergency_session_id")
    if not valid_client_session_id(client_sid):
        return error(400, "invalid_session_id")
    if not valid_user_id(data.get("user_id")):
        return error(400, "invalid_user_id")
    call_status, sms_status = data.get("call_status"), data.get("sms_status")
    if call_status not in CALL_STATUSES or sms_status not in SMS_STATUSES:
        return error(400, "invalid_status")

    session_row, _ = get_or_create_session(client_sid, data["user_id"], now())
    sid = session_row["server_session_id"]
    call_detail = data.get("call_detail") if isinstance(data.get("call_detail"), str) else None
    sms_detail = data.get("sms_detail") if isinstance(data.get("sms_detail"), str) else None
    db().execute(
        "UPDATE sessions SET call_status = ?, call_detail = ?, sms_status = ?, sms_detail = ? WHERE server_session_id = ?",
        (call_status, (call_detail or "")[:300] or None, sms_status, (sms_detail or "")[:300] or None, sid))
    add_event(sid, f"CALL_{call_status}", call_detail)
    if sms_status != "NOT_ATTEMPTED":
        add_event(sid, f"SMS_{sms_status}", sms_detail)
    touch_session(sid)
    db().commit()
    audit(g.device_actor, "DEVICE_STATUS_UPDATE", sid, f"call={call_status} sms={sms_status}")
    return jsonify({"ok": True, "server_session_id": sid})


@emergency_bp.route("/api/emergency/end", methods=["POST"])
@device_signed("end")
def emergency_end():
    data = json_body()
    if data is None:
        return error(400, "invalid_json")
    if data.get("event_type") != "EMERGENCY_ENDED":
        return error(400, "invalid_event_type")
    client_sid = data.get("emergency_session_id")
    if not valid_client_session_id(client_sid):
        return error(400, "invalid_session_id")
    if not valid_user_id(data.get("user_id")):
        return error(400, "invalid_user_id")
    status = data.get("status")
    if status not in END_STATUSES:
        return error(400, "invalid_status")
    ended = parse_time(data.get("timestamp"))
    t = now()
    if ended is None or ended > t + LOCATION_MAX_FUTURE_SECONDS or ended < t - LOCATION_MAX_AGE_SECONDS:
        ended = t

    session_row, _ = get_or_create_session(client_sid, data["user_id"], ended)
    sid = session_row["server_session_id"]
    if session_row["status"] == "ACTIVE":
        db().execute("UPDATE sessions SET status = ?, ended_at = ? WHERE server_session_id = ?", (status, ended, sid))
        add_event(sid, f"SESSION_{status}", None, ended)
        touch_session(sid)
        db().commit()
        audit(g.device_actor, "DEVICE_END", sid, status)
    return jsonify({"ok": True, "server_session_id": sid, "status": status if session_row["status"] == "ACTIVE" else session_row["status"]})


@emergency_bp.route("/api/emergency/selftest", methods=["POST"])
@device_signed("selftest")
def emergency_selftest():
    """Lets the app's Setup screen verify URL/device id/secret. Stores nothing and needs no location."""
    audit(g.device_actor, "DEVICE_SELFTEST")
    return jsonify({"ok": True, "server_time": int(now())})


# --------------------------------------------------------------------------- dashboard auth
# The login is its own signed, expiring, HttpOnly cookie, so the host app needs no secret_key.

class _ClockSigner(TimestampSigner):
    def get_timestamp(self):
        return int(now())


def _serializer():
    return URLSafeTimedSerializer(SECRET_KEY, salt="emergency-dashboard-v1", signer=_ClockSigner)


def is_logged_in():
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    try:
        data = _serializer().loads(token, max_age=SESSION_SECONDS)
    except BadSignature:                 # includes SignatureExpired
        return False
    return isinstance(data, dict) and data.get("a") == 1


def _cookie_flags():
    return {"httponly": True, "secure": COOKIE_SECURE, "samesite": "Lax", "path": "/"}


def require_login(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            if request.path.startswith("/api/"):
                return error(401, "not_logged_in")
            return redirect(url_for("emergency.login"))
        if request.path.startswith("/api/") and not limiter.allow(("dash", client_ip()), DASHBOARD_READS_PER_MINUTE, 60):
            return error(429, "rate_limited")
        return fn(*args, **kwargs)
    return wrapper


@emergency_bp.route("/emergency/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if is_logged_in():
            return redirect(url_for("emergency.dashboard"))
        return _render("emergency_login.html", error=None, login_url=url_for("emergency.login"))

    ip = client_ip()
    if limiter.count(("login", ip), LOGIN_LOCKOUT_SECONDS) >= LOGIN_MAX_FAILURES:
        audit("dashboard", "LOGIN_LOCKED_OUT")
        return _render("emergency_login.html", error="Too many attempts. Try again later.",
                       login_url=url_for("emergency.login")), 429

    if _same(request.form.get("password", ""), DASHBOARD_PASSWORD):
        limiter.clear(("login", ip))
        audit("dashboard", "LOGIN_OK")
        resp = redirect(url_for("emergency.dashboard"))
        resp.set_cookie(COOKIE_NAME, _serializer().dumps({"a": 1}), max_age=SESSION_SECONDS, **_cookie_flags())
        return resp

    limiter.add(("login", ip))
    audit("dashboard", "LOGIN_FAILED")
    return _render("emergency_login.html", error="Wrong password.", login_url=url_for("emergency.login")), 401


@emergency_bp.route("/emergency/logout", methods=["POST"])
def logout():
    if is_logged_in():
        audit("dashboard", "LOGOUT")
    resp = redirect(url_for("emergency.login"))
    resp.delete_cookie(COOKIE_NAME, **_cookie_flags())
    return resp


@emergency_bp.route("/emergency")
def index():
    return redirect(url_for("emergency.dashboard") if is_logged_in() else url_for("emergency.login"))


@emergency_bp.route("/emergency/dashboard")
@require_login
def dashboard():
    return _render("emergency_dashboard.html",
                   login_url=url_for("emergency.login"), logout_url=url_for("emergency.logout"))


# --------------------------------------------------------------------------- dashboard API

def _location_json(row):
    if row is None:
        return None
    return {
        "latitude": row["latitude"], "longitude": row["longitude"], "accuracy_m": row["accuracy_m"],
        "measured_at": iso(row["measured_at"]), "measured_epoch": row["measured_at"],
        "received_at": iso(row["received_at"]), "source": row["location_source"],
        "provider": row["provider"], "stale": bool(row["stale"]),
    }


def _session_json(row, latest=None, location_count=0):
    return {
        "server_session_id": row["server_session_id"],
        "status": row["status"],
        "trigger_source": row["trigger_source"],
        "tracking_duration_minutes": row["tracking_duration_minutes"],
        "started_at": iso(row["started_at"]), "started_epoch": row["started_at"],
        "ended_at": iso(row["ended_at"]),
        "last_event_at": iso(row["last_event_at"]), "last_event_epoch": row["last_event_at"],
        "call_status": row["call_status"], "call_detail": row["call_detail"],
        "sms_status": row["sms_status"], "sms_detail": row["sms_detail"],
        "latest_location": _location_json(latest), "location_count": location_count,
    }


@emergency_bp.route("/api/emergency/sessions")
@require_login
def list_sessions():
    limit = int_arg("limit", 20, 1, 100)
    rows = db().execute("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        latest = db().execute(
            "SELECT * FROM locations WHERE server_session_id = ? ORDER BY measured_at DESC LIMIT 1", (r["server_session_id"],)).fetchone()
        count = db().execute("SELECT COUNT(*) AS c FROM locations WHERE server_session_id = ?", (r["server_session_id"],)).fetchone()["c"]
        out.append(_session_json(r, latest, count))
    audit("dashboard", "VIEW_SESSIONS", None, f"count={len(out)}")
    return jsonify({"sessions": out, "server_time": int(now())})


@emergency_bp.route("/api/emergency/sessions/<server_sid>")
@require_login
def session_detail(server_sid):
    row = db().execute("SELECT * FROM sessions WHERE server_session_id = ?", (server_sid,)).fetchone()
    if row is None:
        return error(404, "not_found")
    locations = db().execute(
        "SELECT * FROM locations WHERE server_session_id = ? ORDER BY measured_at ASC LIMIT 1000", (server_sid,)).fetchall()
    events = db().execute(
        "SELECT event_type, detail, event_at FROM events WHERE server_session_id = ? ORDER BY event_at ASC, id ASC", (server_sid,)).fetchall()
    audit("dashboard", "VIEW_SESSION", server_sid)
    latest = locations[-1] if locations else None
    return jsonify({
        "session": _session_json(row, latest, len(locations)),
        "locations": [_location_json(l) for l in locations],
        "events": [{"type": e["event_type"], "detail": e["detail"], "at": iso(e["event_at"])} for e in events],
        "server_time": int(now()),
    })


@emergency_bp.route("/api/emergency/audit")
@require_login
def audit_log():
    limit = int_arg("limit", 100, 1, 500)
    rows = db().execute("SELECT * FROM audit ORDER BY at DESC, id DESC LIMIT ?", (limit,)).fetchall()
    audit("dashboard", "VIEW_AUDIT")
    return jsonify({"entries": [
        {"at": iso(r["at"]), "actor": r["actor"], "action": r["action"],
         "server_session_id": r["server_session_id"], "ip": r["ip"], "detail": r["detail"]} for r in rows
    ]})
