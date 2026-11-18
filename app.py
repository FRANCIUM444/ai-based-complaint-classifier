import csv
import hashlib
import hmac
import io
import os
import re
import secrets
import time
import sqlite3
import smtplib
import ssl
import uuid
from email.message import EmailMessage
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from html import escape

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session, Response, jsonify, send_from_directory, g,
                   abort)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from classifier import CATEGORIES, classify_complaint
from localization import translate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() in {"1", "true", "yes"},
    SESSION_COOKIE_SAMESITE="Lax",
)
DB_PATH = os.path.join(os.path.dirname(__file__), "complaints.db")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
MAX_IMAGE_SIZE = 5 * 1024 * 1024
MAX_IMAGE_COUNT = 3
DUPLICATE_SIMILARITY_THRESHOLD = float(
    os.environ.get("CAMPUSCARE_DUPLICATE_SIMILARITY_THRESHOLD", "0.72")
)
if not 0 < DUPLICATE_SIMILARITY_THRESHOLD <= 1:
    raise ValueError("Duplicate similarity threshold must be greater than 0 and at most 1.")
DUPLICATE_REVIEW_THRESHOLD = min(0.48, DUPLICATE_SIMILARITY_THRESHOLD)
DATABASE_SCHEMA_VERSION = 2
ESCALATION_INTERVAL_SECONDS = 60
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 15 * 60
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_last_escalation_check = 0.0
_escalation_thresholds = {
    "Pending": int(os.environ.get("CAMPUSCARE_ESCALATE_PENDING_HOURS", "48")),
    "In Progress": int(os.environ.get("CAMPUSCARE_ESCALATE_PROGRESS_HOURS", "72")),
}
if any(hours <= 0 for hours in _escalation_thresholds.values()):
    raise ValueError("Complaint escalation thresholds must be positive hours.")

# Demo credentials. Set ADMIN_USERNAME and ADMIN_PASSWORD environment variables before use.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
DUMMY_STUDENT_PASSWORD_HASH = generate_password_hash(secrets.token_urlsafe(32))

LEGACY_CATEGORY_NAMES = {
    "IT & Wi-Fi": "IT / Technical",
    "Water & Sanitation": "Sanitation",
    "Canteen & Food": "Canteen",
    "Transport": "Transportation",
}
CATEGORY_ALIASES = {
    canonical: tuple(legacy for legacy, current in LEGACY_CATEGORY_NAMES.items()
                     if current == canonical)
    for canonical in CATEGORIES
}
STATUSES = ["Pending", "In Progress", "Resolved"]
PRIORITIES = ["Low", "Medium", "High"]

# Update this single mapping when verified college department addresses are available.
DEPARTMENT_EMAILS = {
    "Academics": "academics@example.com",
    "Infrastructure": "maintenance@example.com",
    "IT / Technical": "it@example.com",
    "Sanitation": "housekeeping@example.com",
    "Canteen": "canteen@example.com",
    "Hostel": "hostel@example.com",
    "Transportation": "transport@example.com",
    "Administration": "administration@example.com",
    "Other": "support@example.com",
}


def canonical_category(category):
    return LEGACY_CATEGORY_NAMES.get(category, category if category in CATEGORIES else "Other")


DEPARTMENT_NAMES = {
    "Infrastructure": "Maintenance Department",
    "Academics": "Academic Affairs Department",
    "IT / Technical": "IT Department",
    "Sanitation": "Sanitation Department",
    "Canteen": "Canteen Department",
    "Hostel": "Hostel Department",
    "Transportation": "Transportation Department",
    "Administration": "Administration Department",
    "Other": "CampusCare Administration",
}


def configured_department_recipient(category):
    recipient = DEPARTMENT_EMAILS.get(canonical_category(category), "").strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient):
        return ""
    return recipient


def is_demo_recipient(recipient):
    if not recipient or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient):
        return True
    domain = recipient.rsplit("@", 1)[1].lower()
    if domain in {"example.com", "example.org", "example.net", "example.edu", "localhost"}:
        return True
    if domain.endswith((".example", ".invalid", ".test")):
        return True
    if any(marker in domain for marker in ("placeholder", "yourcollege", "youruniversity")):
        return True
    return False


def build_consolidated_email(group, complaints):
    priority = max(
        (complaint["priority"] for complaint in complaints),
        key=lambda value: {"Low": 0, "Medium": 1, "High": 2}.get(value, 0),
        default="Low",
    )
    location = group["location"] or "Not specified"
    student_count = len({
        complaint["student_id"] or f"complaint:{complaint['complaint_id']}"
        for complaint in complaints
    })
    subject_parts = [f"[CampusCare] Multiple Reports - {group['issue_title']}"]
    if group["location"]:
        subject_parts[0] += f" - {group['location']}"
    references = "\n".join(f"- Complaint #{row['complaint_id']}" for row in complaints)
    description_lines = "\n".join(
        f"- Complaint #{row['complaint_id']} ({row['created_at']}): {row['complaint_text']}"
        for row in complaints
    )
    text = (
        f"Dear {DEPARTMENT_NAMES.get(group['category'], 'Department')},\n\n"
        "CampusCare has received multiple reports regarding the same issue.\n\n"
        "CONSOLIDATED COMPLAINT SUMMARY\n\n"
        f"Issue: {group['issue_title']}\n"
        f"Summary: {group['summary']}\n"
        f"Location: {location}\n"
        f"Category: {group['category']}\n"
        f"Total Reports: {len(complaints)} from {student_count} students\n"
        f"Priority: {priority}\n\n"
        "INDIVIDUAL COMPLAINT REFERENCES:\n"
        f"{references}\n\n"
        "INDIVIDUAL REPORT DETAILS:\n"
        f"{description_lines}\n\n"
        "Please investigate the issue and take the necessary action.\n\n"
        "Regards,\nCampusCare Complaint Management System"
    )
    details_html = "".join(
        "<li><strong>Complaint #{} ({})</strong><br>{}</li>".format(
            escape(str(row["complaint_id"])),
            escape(str(row["created_at"])),
            escape(str(row["complaint_text"])),
        )
        for row in complaints
    )
    references_html = "".join(
        f"<li>Complaint #{escape(str(row['complaint_id']))}</li>"
        for row in complaints
    )
    html = (
        '<!doctype html><html><body style="margin:0;background:#f3f7f8;'
        'font-family:Arial,sans-serif;color:#183153"><div style="max-width:680px;'
        'margin:24px auto;background:#fff;border:1px solid #dce7ea;border-radius:12px;'
        'overflow:hidden"><div style="padding:22px 28px;background:#164e63;color:#fff">'
        '<strong>CampusCare</strong><div style="font-size:13px;margin-top:6px">'
        'Consolidated complaint notification</div></div><div style="padding:28px">'
        f"<p>Dear {escape(DEPARTMENT_NAMES.get(group['category'], 'Department'))},</p>"
        '<p>CampusCare has received multiple reports regarding the same issue.</p>'
        '<h2 style="font-size:18px">Consolidated Complaint Summary</h2>'
        '<table style="border-collapse:collapse;width:100%">'
        f"<tr><th align=\"left\">Issue</th><td>{escape(group['issue_title'])}</td></tr>"
        f"<tr><th align=\"left\">Summary</th><td>{escape(group['summary'])}</td></tr>"
        f"<tr><th align=\"left\">Location</th><td>{escape(location)}</td></tr>"
        f"<tr><th align=\"left\">Category</th><td>{escape(group['category'])}</td></tr>"
        f"<tr><th align=\"left\">Total Reports</th><td>{len(complaints)} reports from "
        f"{student_count} students</td></tr>"
        f"<tr><th align=\"left\">Priority</th><td>{escape(priority)}</td></tr>"
        "</table><h3 style=\"font-size:16px\">Individual Complaint References</h3>"
        f"<ul>{references_html}</ul><h3 style=\"font-size:16px\">Report Details</h3>"
        f"<ul>{details_html}</ul><p>Please investigate the issue and take the necessary action."
        "</p><p>Regards,<br>CampusCare Complaint Management System</p></div></div>"
        "</body></html>"
    )
    return f"{subject_parts[0]}", text, html


def send_consolidated_email(recipient, notification_id, subject, text, html):
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_user = os.environ.get("SMTP_USERNAME", "").strip()
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    from_address = os.environ.get("SMTP_FROM", smtp_user).strip()
    if not smtp_host or not from_address:
        return False, "SMTP settings are incomplete"
    try:
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        return False, "SMTP_PORT is invalid"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_address
    message["To"] = recipient
    message["X-CampusCare-Batch-ID"] = notification_id
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if smtp_user:
                server.login(smtp_user, smtp_password)
            refused = server.send_message(message)
            if refused:
                refused_recipients = ", ".join(sorted(refused))
                return False, f"SMTP refused recipient(s): {refused_recipients}"
        return True, "SMTP accepted notification"
    except (OSError, smtplib.SMTPException) as exc:
        app.logger.exception("Consolidated email failed for batch %s", notification_id)
        return False, f"{type(exc).__name__}: {exc}"



@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def backup_database_before_migration():
    if not os.path.isfile(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return
    existing = sqlite3.connect(DB_PATH)
    try:
        version = existing.execute("PRAGMA user_version").fetchone()[0]
    finally:
        existing.close()
    if version >= DATABASE_SCHEMA_VERSION:
        return
    backup_path = f"{DB_PATH}.backup.{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    source = sqlite3.connect(DB_PATH)
    backup = sqlite3.connect(backup_path)
    try:
        source.backup(backup)
    finally:
        backup.close()
        source.close()
    app.logger.info("Backed up database before migration to %s", backup_path)


def init_db():
    backup_database_before_migration()
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                student_email TEXT,
                department TEXT,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                updated_at TEXT,
                feedback_rating INTEGER,
                feedback_comment TEXT,
                anonymous INTEGER NOT NULL DEFAULT 0,
                email_status TEXT DEFAULT 'not_attempted',
                email_recipient TEXT
            )
        """)
        # Migrate an older version of the project database without deleting records.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(complaints)")}
        migrations = {
            "confidence": "ALTER TABLE complaints ADD COLUMN confidence REAL DEFAULT 0",
            "updated_at": "ALTER TABLE complaints ADD COLUMN updated_at TEXT",
            "feedback_rating": "ALTER TABLE complaints ADD COLUMN feedback_rating INTEGER",
            "feedback_comment": "ALTER TABLE complaints ADD COLUMN feedback_comment TEXT",
            "anonymous": "ALTER TABLE complaints ADD COLUMN anonymous INTEGER NOT NULL DEFAULT 0",
            "email_status": "ALTER TABLE complaints ADD COLUMN email_status TEXT DEFAULT 'not_attempted'",
            "email_recipient": "ALTER TABLE complaints ADD COLUMN email_recipient TEXT",
            "duplicate_of": "ALTER TABLE complaints ADD COLUMN duplicate_of INTEGER",
            "escalated_at": "ALTER TABLE complaints ADD COLUMN escalated_at TEXT",
        }
        for col, sql in migrations.items():
            if col not in columns:
                conn.execute(sql)
        if "student_id" not in columns:
            conn.execute("""
                ALTER TABLE complaints ADD COLUMN student_id INTEGER
                REFERENCES students(id)
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                college_email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                enrollment_number TEXT NOT NULL COLLATE NOCASE UNIQUE,
                department TEXT NOT NULL,
                academic_year TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_attempts (
                attempt_key TEXT PRIMARY KEY,
                failed_count INTEGER NOT NULL,
                window_started_at INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaints_student
            ON complaints(student_id, id DESC)
        """)
        conn.execute("UPDATE complaints SET updated_at = created_at WHERE updated_at IS NULL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaint_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id INTEGER NOT NULL REFERENCES complaints(id),
                stored_name TEXT NOT NULL UNIQUE,
                original_name TEXT NOT NULL,
                content_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaint_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id INTEGER NOT NULL REFERENCES complaints(id),
                event_type TEXT NOT NULL,
                status TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaint_events_complaint
            ON complaint_events(complaint_id, id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaint_attachments_complaint
            ON complaint_attachments(complaint_id)
        """)
        conn.execute("""
            INSERT INTO complaint_events (complaint_id, event_type, status, message, created_at)
            SELECT c.id, 'status', c.status, 'Complaint submitted', c.created_at
            FROM complaints AS c
            WHERE NOT EXISTS (
                SELECT 1 FROM complaint_events AS e
                WHERE e.complaint_id = c.id AND e.event_type = 'status'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaint_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                public_id TEXT NOT NULL UNIQUE,
                notification_id TEXT NOT NULL UNIQUE,
                related_group_id INTEGER REFERENCES complaint_groups(id),
                matched_group_id INTEGER REFERENCES complaint_groups(id),
                issue_title TEXT NOT NULL,
                summary TEXT NOT NULL,
                category TEXT NOT NULL,
                location TEXT NOT NULL DEFAULT '',
                approval_status TEXT NOT NULL DEFAULT 'pending_approval',
                email_status TEXT NOT NULL DEFAULT 'pending_approval',
                department_recipient TEXT NOT NULL DEFAULT '',
                manual_review INTEGER NOT NULL DEFAULT 0,
                review_note TEXT,
                similarity_score REAL,
                send_attempts INTEGER NOT NULL DEFAULT 0,
                approval_admin_id TEXT,
                approved_at TEXT,
                rejected_by_admin_id TEXT,
                rejected_at TEXT,
                rejection_reason TEXT,
                sent_at TEXT,
                smtp_result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaint_group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL REFERENCES complaint_groups(id),
                complaint_id INTEGER NOT NULL UNIQUE REFERENCES complaints(id),
                similarity_score REAL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaint_groups_status
            ON complaint_groups(approval_status, email_status, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complaint_group_members_group
            ON complaint_group_members(group_id, id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS complaint_notification_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL REFERENCES complaint_groups(id),
                notification_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                recipient TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                smtp_result TEXT,
                smtp_accepted INTEGER,
                UNIQUE(group_id, attempt_number)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_notification_attempts_group
            ON complaint_notification_attempts(group_id, attempt_number)
        """)
        conn.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")


@app.context_processor
def inject_language():
    g.language = session.get("language", "en")
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return {
        "t": lambda key, **values: translate(g.language, key, **values),
        "language": g.language,
        "category_key": category_key,
        "department_email_for_category": configured_department_recipient,
        "is_demo_recipient": is_demo_recipient,
        "csrf_token": session["csrf_token"],
        "student_session": session.get("role") == "student",
        "admin_session": session.get("role") == "admin",
    }


@app.before_request
def enforce_csrf():
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return
    expected = session.get("csrf_token", "")
    supplied = request.headers.get("X-CSRF-Token", "")
    if request.is_json:
        supplied = supplied or ""
    else:
        supplied = supplied or request.form.get("csrf_token", "")
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        return jsonify(error="Invalid or missing CSRF token"), 400


def current_student_id():
    if session.get("role") != "student":
        return None
    return session.get("student_id")


def student_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if session.get("role") != "student" or not session.get("student_id"):
            if session.get("role") == "admin":
                abort(403)
            return redirect(url_for("student_login", next=request.full_path))
        return fn(*args, **kwargs)
    return wrapped


def admin_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if session.get("role") == "student":
            abort(403)
        if session.get("role") != "admin" or not session.get("admin_logged_in"):
            flash(translate(session.get("language", "en"), "login_required"), "error")
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    return wrapped


def owner_required(fn):
    @wraps(fn)
    def wrapped(complaint_id, *args, **kwargs):
        if session.get("role") == "admin" and session.get("admin_logged_in"):
            return fn(complaint_id, *args, **kwargs)
        student_id = current_student_id()
        if not student_id:
            return redirect(url_for("student_login", next=request.full_path))
        with get_db() as conn:
            owned = conn.execute(
                "SELECT 1 FROM complaints WHERE id = ? AND student_id = ?",
                (complaint_id, student_id),
            ).fetchone()
        if not owned:
            abort(404)
        return fn(complaint_id, *args, **kwargs)
    return wrapped


def safe_next_url(candidate, fallback):
    if candidate.startswith("/") and not candidate.startswith(("//", "/\\")):
        if candidate.startswith("/admin") and session.get("role") != "admin":
            return fallback
        return candidate
    return fallback


def auth_attempt_key(email):
    identity = f"{request.remote_addr or 'unknown'}\0{email.lower()}"
    return hashlib.sha256((app.secret_key + identity).encode("utf-8")).hexdigest()


def login_is_limited(key):
    now = int(time.time())
    with get_db() as conn:
        conn.execute(
            "DELETE FROM auth_attempts WHERE window_started_at <= ?",
            (now - LOGIN_WINDOW_SECONDS,),
        )
        row = conn.execute(
            "SELECT failed_count, window_started_at FROM auth_attempts WHERE attempt_key = ?",
            (key,),
        ).fetchone()
        if not row:
            return False
        return row["failed_count"] >= LOGIN_MAX_ATTEMPTS


def record_login_failure(key):
    now = int(time.time())
    with get_db() as conn:
        conn.execute("""
            INSERT INTO auth_attempts (attempt_key, failed_count, window_started_at)
            VALUES (?, 1, ?)
            ON CONFLICT(attempt_key) DO UPDATE SET
                failed_count = CASE
                    WHEN excluded.window_started_at - auth_attempts.window_started_at >= ?
                    THEN 1 ELSE auth_attempts.failed_count + 1 END,
                window_started_at = CASE
                    WHEN excluded.window_started_at - auth_attempts.window_started_at >= ?
                    THEN excluded.window_started_at ELSE auth_attempts.window_started_at END
        """, (key, now, LOGIN_WINDOW_SECONDS, LOGIN_WINDOW_SECONDS))


def clear_login_failures(key):
    with get_db() as conn:
        conn.execute("DELETE FROM auth_attempts WHERE attempt_key = ?", (key,))


@app.before_request
def run_scheduled_escalations():
    global _last_escalation_check
    if request.endpoint == "static":
        return
    now = time.monotonic()
    if now - _last_escalation_check >= ESCALATION_INTERVAL_SECONDS:
        escalate_overdue_complaints()
        _last_escalation_check = now


def escalate_overdue_complaints(now=None):
    now = now or datetime.now()
    now_text = now.strftime("%Y-%m-%d %H:%M:%S")
    escalated_count = 0
    with get_db() as conn:
        for status, hours in _escalation_thresholds.items():
            cutoff = (now - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            overdue = conn.execute("""
                SELECT id FROM complaints
                WHERE status = ? AND escalated_at IS NULL AND created_at <= ?
            """, (status, cutoff)).fetchall()
            for row in overdue:
                complaint_id = row["id"]
                cursor = conn.execute("""
                    UPDATE complaints SET priority = 'High', escalated_at = ?
                    WHERE id = ? AND escalated_at IS NULL
                """, (now_text, complaint_id))
                if cursor.rowcount == 0:
                    continue
                conn.execute("""
                    INSERT INTO complaint_events
                    (complaint_id, event_type, status, message, created_at)
                    VALUES (?, 'escalation', ?, 'Complaint automatically escalated', ?)
                """, (complaint_id, status, now_text))
                escalated_count += 1
    return escalated_count


_LOCATION_PATTERNS = (
    re.compile(
        r"\b(?P<place>room|classroom|class|lecture room)\s*"
        r"(?:number|no\.?|#)?\s*(?P<number>[a-z]?\d+[a-z]?)\b", re.I,
    ),
    re.compile(
        r"\b(?P<place>block|building|lab|laboratory|floor)\s*"
        r"(?:number|no\.?|#)?\s*(?P<number>[a-z]?\d+[a-z]?)\b", re.I,
    ),
)
_LOCATION_TEXT_PATTERN = re.compile(
    r"\b(?:room|classroom|class|lecture room|block|building|lab|laboratory|floor)"
    r"\s*(?:number|no\.?|#)?\s*[a-z]?\d+[a-z]?\b",
    re.I,
)
_ISSUE_STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "been", "ceiling", "class", "classroom",
    "for", "from", "in", "is", "it", "my", "our", "the", "this", "to", "was",
    "were", "with",
}
_GENERIC_ISSUE_WORDS = {"malfunction", "issue", "problem", "not", "working"}


def extract_location(text):
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    for index, pattern in enumerate(_LOCATION_PATTERNS):
        match = pattern.search(normalized)
        if match:
            number = match.group("number").upper()
            place = "Room" if index == 0 else match.group("place").title()
            return f"{place} {number}"
    return ""


def normalize_issue_text(text):
    normalized = text.lower()
    normalized = re.sub(
        r"\b(?:(?:is|are|was|were|has|have)\s+)?(?:not working|does not work|doesn't work|"
        r"isn't working|aren't working|stopped working|out of order|broken|faulty)\b",
        " malfunction ",
        normalized,
    )
    normalized = re.sub(r"\b(?:problem|issue)\b", " malfunction ", normalized)
    normalized = _LOCATION_TEXT_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\bfans\b", "fan", normalized)
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _ISSUE_STOP_WORDS
    ]
    return " ".join(tokens)


def normalized_issue_title(text):
    tokens = normalize_issue_text(text).split()
    issue_words = [token for token in tokens if token not in _GENERIC_ISSUE_WORDS]
    if not issue_words:
        return "Reported Issue"
    title = " ".join(word.upper() if word == "wifi" else word.title()
                     for word in issue_words[:6])
    if "malfunction" in tokens:
        title += " Malfunction"
    return title


def issue_keywords(normalized_text):
    return set(normalized_text.split()) - _GENERIC_ISSUE_WORDS


def complaint_similarity(first, second):
    first_normalized = normalize_issue_text(first)
    second_normalized = normalize_issue_text(second)
    first_keywords = issue_keywords(first_normalized)
    second_keywords = issue_keywords(second_normalized)
    shared = first_keywords & second_keywords
    if not shared:
        return 0.0
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    vectors = vectorizer.fit_transform([first_normalized, second_normalized])
    text_similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0, 0])
    keyword_f1 = 2 * len(shared) / (len(first_keywords) + len(second_keywords))
    return 0.75 * text_similarity + 0.25 * keyword_f1


def find_group_match(conn, complaint_text, category, location):
    groups = conn.execute("""
        SELECT g.*
        FROM complaint_groups AS g
        WHERE g.approval_status != 'deleted'
        ORDER BY g.id DESC
    """).fetchall()
    possible_matches = []
    for group in groups:
        if canonical_category(group["category"]) != canonical_category(category):
            continue
        members = conn.execute("""
            SELECT c.id, c.complaint_text
            FROM complaint_group_members AS gm
            JOIN complaints AS c ON c.id = gm.complaint_id
            WHERE gm.group_id = ? AND c.status != 'Resolved'
            ORDER BY c.id
        """, (group["id"],)).fetchall()
        if not members:
            continue
        score, matching_complaint_id = max(
            (
                (complaint_similarity(complaint_text, member["complaint_text"]), member["id"])
                for member in members
            ),
            default=(0.0, None),
        )
        if score < DUPLICATE_REVIEW_THRESHOLD:
            continue
        is_pending = (
            group["approval_status"] == "pending_approval"
            and group["email_status"] == "pending_approval"
        )
        possible_matches.append((group, score, matching_complaint_id, is_pending))

    location_matches = [
        item for item in possible_matches
        if location and item[0]["location"] and item[0]["location"].casefold() == location.casefold()
    ]
    if location_matches:
        pending_matches = [item for item in location_matches if item[3]]
        best = max(pending_matches or location_matches, key=lambda item: item[1])
        if best[1] >= DUPLICATE_SIMILARITY_THRESHOLD and best[3]:
            return {"mode": "merge", "group": best[0], "score": best[1],
                    "complaint_id": best[2]}
        if best[1] >= DUPLICATE_SIMILARITY_THRESHOLD:
            return {"mode": "new_batch", "group": best[0], "score": best[1],
                    "complaint_id": best[2]}
        return {"mode": "review", "group": best[0], "score": best[1],
                "complaint_id": best[2]}

    unresolved_location = [
        item for item in possible_matches
        if not location or not item[0]["location"]
    ]
    if unresolved_location:
        best = max(unresolved_location, key=lambda item: item[1])
        return {"mode": "review" if best[3] else "new_batch",
                "group": best[0], "score": best[1],
                "complaint_id": best[2]}
    return None


HINDI_CLASSIFIER_HINTS = (
    ("वाई-फाई", "wifi is not working"),
    ("वाईफाई", "wifi is not working"),
    ("इंटरनेट", "internet connection is very slow"),
    ("छात्रावास", "hostel room has no hot water"),
    ("हॉस्टल", "hostel room has no hot water"),
    ("शौचालय", "toilets are dirty and smell bad"),
    ("कैंटीन", "canteen food is stale and poor quality"),
    ("खाना", "canteen food is poor quality"),
    ("परीक्षा", "exam timetable has a mistake"),
    ("छात्रवृत्ति", "scholarship form is pending in office"),
    ("फीस", "fee payment receipt is missing"),
    ("बस", "college bus arrived late"),
    ("परिवहन", "bus route does not cover our stop"),
    ("पंखा", "classroom fans and lights are broken"),
    ("धमकी", "someone is threatening students"),
    ("रैगिंग", "I want to report ragging by seniors"),
    ("उत्पीड़न", "someone is threatening students"),
    ("आग", "urgent smoke coming from an electrical panel"),
    ("धुआँ", "urgent smoke coming from an electrical panel"),
    ("बिजली का तार", "urgent exposed electrical wire in corridor"),
    ("पानी", "there is no water in the washroom"),
    ("स्वच्छता", "toilets are dirty and smell bad"),
    ("सफाई", "toilets are dirty and smell bad"),
)


def classify_multilingual_complaint(text):
    normalized = text.lower()
    if any("\u0900" <= char <= "\u097f" for char in text):
        english_hints = [
            hint for phrase, hint in HINDI_CLASSIFIER_HINTS
            if phrase in normalized
        ]
        if english_hints:
            result = classify_complaint(" ".join(english_hints))
        else:
            result = classify_complaint(text)
    else:
        result = classify_complaint(text)
    category, priority, confidence = result
    has_room_fan = bool(extract_location(text).startswith("Room ")) and bool(
        re.search(r"\b(?:fan|fans)\b", normalized)
    )
    if has_room_fan and not re.search(r"\b(?:hostel|dorm|dormitory)\b", normalized):
        category = "Infrastructure"
    return category, priority, confidence


def read_image_uploads(files):
    uploads = [item for item in files if item and item.filename]
    if len(uploads) > MAX_IMAGE_COUNT:
        raise ValueError("upload_count_error")

    accepted = []
    signatures = {
        ".jpg": ("image/jpeg", lambda data: data.startswith(b"\xff\xd8\xff")),
        ".jpeg": ("image/jpeg", lambda data: data.startswith(b"\xff\xd8\xff")),
        ".png": ("image/png", lambda data: data.startswith(b"\x89PNG\r\n\x1a\n")),
        ".webp": ("image/webp", lambda data: len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"),
    }
    for upload in uploads:
        extension = os.path.splitext(upload.filename)[1].lower()
        signature = signatures.get(extension)
        if not signature:
            raise ValueError("upload_type_error")
        data = upload.stream.read(MAX_IMAGE_SIZE + 1)
        upload.stream.seek(0)
        if not data or len(data) > MAX_IMAGE_SIZE:
            raise ValueError("upload_size_error")
        content_type, matches = signature
        if not matches(data):
            raise ValueError("upload_signature_error")
        original_name = secure_filename(upload.filename) or f"image{extension}"
        accepted.append({
            "stored_name": f"{uuid.uuid4().hex}{extension}",
            "original_name": original_name[:255],
            "content_type": content_type,
            "file_size": len(data),
            "data": data,
        })
    return accepted


def assistant_reply(message, language):
    normalized = message.lower()
    devanagari_digits = "०१२३४५६७८९"
    message = message.translate(str.maketrans(devanagari_digits, "0123456789"))
    normalized = message.lower()
    status_intent = any(term in normalized for term in
                        ("status", "track", "where is", "स्थिति", "स्टेटस", "ट्रैक"))
    complaint_id = re.search(r"#?\b(\d{1,10})\b", message) if status_intent else None
    if complaint_id:
        student_id = current_student_id()
        if not student_id:
            return translate(language, "assistant_login_status")
        with get_db() as conn:
            row = conn.execute(
                "SELECT status FROM complaints WHERE id = ? AND student_id = ?",
                (int(complaint_id.group(1)), student_id)
            ).fetchone()
        if row:
            return translate(language, "assistant_status_found",
                             id=complaint_id.group(1),
                             status=translate(language, row["status"].lower().replace(" ", "_")))
        return translate(language, "assistant_status_missing")
    if status_intent:
        if not current_student_id():
            return translate(language, "assistant_login_status")
        if any(term in normalized for term in ("track", "ट्रैक")):
            return translate(language, "assistant_track")
        return translate(language, "assistant_status_prompt")
    if any(term in normalized for term in
           ("emergency", "danger", "unsafe", "fire", "violence", "आपात", "खतरा", "असुरक्षित")):
        return translate(language, "assistant_urgent")
    if any(term in normalized for term in
           ("anonymous", "anonymously", "गुमनाम", "बेनाम")):
        return translate(language, "assistant_submit")
    if any(term in normalized for term in
           ("submit", "complaint", "शिकायत", "समस्या", "दर्ज")):
        return translate(language, "assistant_submit")
    if any(term in normalized for term in ("category", "classify", "श्रेणी", "वर्गीकरण")):
        return translate(language, "assistant_categories")
    if any(term in normalized for term in ("help", "कैसे", "मदद")):
        return translate(language, "assistant_help")
    if any(term in normalized for term in ("track", "ट्रैक")):
        return translate(language, "assistant_track")
    if len(message) >= 12 and any("a" <= char.lower() <= "z" for char in message):
        category, priority, _ = classify_multilingual_complaint(message)
        if language == "hi":
            return f"यह समस्या संभवतः “{translate(language, category_key(category))}” श्रेणी की है। सुझाई गई प्राथमिकता: {translate(language, priority.lower())}।"
        return f"This sounds like a “{category}” concern with suggested {priority.lower()} priority. Submit it so the team can review."
    return translate(language, "assistant_help")


def category_key(category):
    return {
        "Academics": "academics",
        "Infrastructure": "infrastructure",
        "IT / Technical": "it_technical",
        "Sanitation": "sanitation",
        "Canteen": "canteen",
        "Hostel": "hostel",
        "Transportation": "transportation",
        "Administration": "administration",
        "Other": "other",
    }.get(canonical_category(category), "other")


@app.route("/")
def index():
    if current_student_id():
        return redirect(url_for("student_dashboard"))
    return render_template("landing.html")


@app.route("/submit", methods=["GET"])
@student_required
def submit_page():
    with get_db() as conn:
        student = conn.execute(
            "SELECT full_name, college_email, department FROM students WHERE id = ?",
            (current_student_id(),),
        ).fetchone()
    if not student:
        session.clear()
        return redirect(url_for("student_login"))
    return render_template("index.html", categories=CATEGORIES, student=student)


@app.route("/language/<language>", methods=["POST"])
def set_language(language):
    if language not in {"en", "hi"}:
        return "Unsupported language", 404
    session["language"] = language
    destination = request.form.get("next", "/")
    if not destination.startswith("/") or destination.startswith(("//", "/\\")):
        destination = "/"
    return redirect(destination)


@app.route("/assistant", methods=["POST"])
def assistant():
    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify(error=translate(session.get("language", "en"), "assistant_error")), 400
    message = payload.get("message", "")
    if not isinstance(message, str):
        return jsonify(error=translate(session.get("language", "en"), "assistant_error")), 400
    message = message.strip()
    language = payload.get("language", session.get("language", "en"))
    if language not in {"en", "hi"}:
        language = session.get("language", "en")
    if not message:
        return jsonify(error=translate(language, "assistant_empty")), 400
    if len(message) > 500:
        return jsonify(error=translate(language, "assistant_too_long")), 400
    return jsonify(reply=assistant_reply(message, language))


@app.route("/submit", methods=["POST"])
@student_required
def submit():
    student_id = current_student_id()
    complaint = request.form.get("complaint_text", "").strip()

    if not complaint or len(complaint) < 8:
        flash(translate(session.get("language", "en"), "complaint_min_error"), "error")
        return redirect(url_for("submit_page"))

    try:
        uploads = read_image_uploads(request.files.getlist("images"))
    except ValueError as exc:
        flash(translate(session.get("language", "en"), str(exc)), "error")
        return redirect(url_for("submit_page"))
    category, priority, confidence = classify_multilingual_complaint(complaint)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stored_paths = []

    try:
        with get_db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            student = conn.execute("""
                SELECT full_name, college_email, department
                FROM students WHERE id = ?
            """, (student_id,)).fetchone()
            if not student:
                session.clear()
                return redirect(url_for("student_login"))
            name, email, department = (
                student["full_name"], student["college_email"], student["department"]
            )
            location = extract_location(complaint)
            match = find_group_match(conn, complaint, category, location)
            duplicate = (
                (match["complaint_id"], match["score"]) if match else None
            )
            duplicate_id = duplicate[0] if duplicate else None
            cur = conn.execute("""
                INSERT INTO complaints
                (student_name, student_email, department, complaint_text, category, priority,
                 status, created_at, confidence, updated_at, anonymous, duplicate_of, student_id,
                 email_status)
                VALUES (?, ?, ?, ?, ?, ?, 'Pending', ?, ?, ?, 0, ?, ?, 'pending_approval')
            """, (name, email, department, complaint, category, priority, now,
                  confidence, now, duplicate_id, student_id))
            complaint_id = cur.lastrowid
            if match and match["mode"] == "merge":
                group_id = match["group"]["id"]
                group_public_id = match["group"]["public_id"]
                conn.execute("""
                    UPDATE complaint_groups SET updated_at = ?
                    WHERE id = ? AND approval_status = 'pending_approval'
                      AND email_status = 'pending_approval'
                """, (now, group_id))
            else:
                issue_title = normalized_issue_title(complaint)
                manual_review = bool(not location or (match and match["mode"] == "review"))
                if match and not location:
                    review_note = "Similar report found, but the location is unclear."
                elif match:
                    review_note = (
                        f"Similarity {match['score']:.0%} is below the automatic "
                        "grouping threshold; review the related group."
                    )
                elif not location:
                    review_note = "Location could not be identified; verify it during review."
                else:
                    review_note = None
                group_public_id = f"CCG-{uuid.uuid4().hex[:10].upper()}"
                group_cursor = conn.execute("""
                    INSERT INTO complaint_groups
                    (public_id, notification_id, related_group_id, matched_group_id,
                     issue_title, summary, category, location, approval_status, email_status,
                     department_recipient, manual_review, review_note, similarity_score,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_approval', 'pending_approval',
                            ?, ?, ?, ?, ?, ?)
                """, (
                    group_public_id,
                    uuid.uuid4().hex,
                    match["group"]["id"] if match else None,
                    match["group"]["id"] if match else None,
                    issue_title,
                    f"Reports indicate {issue_title.lower()} at {location or 'an unspecified location'}.",
                    category,
                    location,
                    configured_department_recipient(category),
                    int(manual_review),
                    review_note,
                    match["score"] if match else None,
                    now,
                    now,
                ))
                group_id = group_cursor.lastrowid
            conn.execute("""
                INSERT INTO complaint_group_members
                (group_id, complaint_id, similarity_score, created_at)
                VALUES (?, ?, ?, ?)
            """, (group_id, complaint_id, match["score"] if match else None, now))
            group_report_count = conn.execute(
                "SELECT COUNT(*) FROM complaint_group_members WHERE group_id = ?",
                (group_id,),
            ).fetchone()[0]
            conn.execute("""
                INSERT INTO complaint_events
                (complaint_id, event_type, status, message, created_at)
                VALUES (?, 'status', 'Pending', 'Complaint submitted', ?)
            """, (complaint_id, now))
            if uploads:
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                for upload in uploads:
                    file_path = os.path.join(UPLOAD_FOLDER, upload["stored_name"])
                    with open(file_path, "xb") as image_file:
                        image_file.write(upload["data"])
                    stored_paths.append(file_path)
                    conn.execute("""
                        INSERT INTO complaint_attachments
                        (complaint_id, stored_name, original_name, content_type, file_size, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (complaint_id, upload["stored_name"], upload["original_name"],
                          upload["content_type"], upload["file_size"], now))
                conn.execute("""
                    INSERT INTO complaint_events
                    (complaint_id, event_type, status, message, created_at)
                    VALUES (?, 'attachment', 'Pending', 'Image(s) attached', ?)
                """, (complaint_id, now))
    except Exception:
        for file_path in stored_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        raise

    return render_template("submitted.html", complaint_id=complaint_id,
                           category=category, priority=priority,
                           confidence=round(confidence * 100),
                           email_status="pending_approval",
                           group_public_id=group_public_id,
                           group_report_count=group_report_count,
                           manual_review=bool(not location or (match and match["mode"] == "review")),
                           duplicate=duplicate,
                           image_count=len(uploads),
                           language=session.get("language", "en"))


@app.route("/track", methods=["GET", "POST"])
@student_required
def track():
    complaint = None
    searched = False
    timeline = []
    if request.method == "POST":
        searched = True
        raw_id = request.form.get("complaint_id", "").strip()
        if raw_id.isdigit():
            with get_db() as conn:
                complaint = conn.execute("""
                    SELECT id, category, priority, status, created_at, complaint_text,
                           feedback_rating, feedback_comment, student_id
                    FROM complaints WHERE id = ? AND student_id = ?
                """, (int(raw_id), current_student_id())).fetchone()
                if complaint:
                    timeline = conn.execute("""
                        SELECT event_type, status, message, created_at
                        FROM complaint_events WHERE complaint_id = ? ORDER BY id
                    """, (int(raw_id),)).fetchall()
    return render_template("track.html", complaint=complaint, searched=searched,
                           timeline=timeline)


@app.route("/api/complaints/<int:complaint_id>/timeline")
@owner_required
def complaint_timeline(complaint_id):
    with get_db() as conn:
        complaint = conn.execute(
            "SELECT status FROM complaints WHERE id = ?", (complaint_id,)
        ).fetchone()
        if not complaint:
            return jsonify(error="Complaint not found"), 404
        events = conn.execute("""
            SELECT event_type, status, message, created_at
            FROM complaint_events WHERE complaint_id = ? ORDER BY id
        """, (complaint_id,)).fetchall()
    return jsonify(
        complaint_id=complaint_id,
        current_status=complaint["status"],
        events=[dict(event) for event in events],
    )


@app.route("/feedback/<int:complaint_id>", methods=["POST"])
@student_required
@owner_required
def feedback(complaint_id):
    raw_rating = request.form.get("rating", "")
    comment = request.form.get("comment", "").strip()[:1000]
    if raw_rating not in {"1", "2", "3", "4", "5"}:
        flash(translate(session.get("language", "en"), "rating_error"), "error")
        return redirect(url_for("track"))
    with get_db() as conn:
        row = conn.execute(
            "SELECT status FROM complaints WHERE id = ? AND student_id = ?",
            (complaint_id, current_student_id()),
        ).fetchone()
        if not row:
            flash(translate(session.get("language", "en"), "complaint_missing_error"), "error")
        elif row["status"] != "Resolved":
            flash(translate(session.get("language", "en"), "feedback_unresolved_error"), "error")
        else:
            conn.execute("""
                UPDATE complaints SET feedback_rating = ?, feedback_comment = ?
                WHERE id = ?
            """, (int(raw_rating), comment, complaint_id))
            conn.execute("""
                INSERT INTO complaint_events
                (complaint_id, event_type, status, message, created_at)
                VALUES (?, 'rating', 'Resolved', 'Student feedback received', ?)
            """, (complaint_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            flash(translate(session.get("language", "en"), "feedback_saved"), "success")
    return redirect(url_for("track"))


@app.route("/admin/attachments/<int:attachment_id>")
def view_attachment(attachment_id):
    with get_db() as conn:
        attachment = conn.execute("""
            SELECT a.stored_name, a.original_name, a.content_type, c.student_id
            FROM complaint_attachments AS a
            JOIN complaints AS c ON c.id = a.complaint_id
            WHERE a.id = ?
        """, (attachment_id,)).fetchone()
    if not attachment:
        return "Attachment not found", 404
    if session.get("role") != "admin" or not session.get("admin_logged_in"):
        student_id = current_student_id()
        if not student_id:
            return redirect(url_for("student_login", next=request.full_path))
        if attachment["student_id"] != student_id:
            abort(404)
    return send_from_directory(
        UPLOAD_FOLDER, attachment["stored_name"],
        mimetype=attachment["content_type"], as_attachment=False,
        download_name=attachment["original_name"]
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("role") == "student":
        abort(403)
    if session.get("role") == "admin" and session.get("admin_logged_in"):
        return redirect(url_for("admin"))
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        attempt_key = auth_attempt_key(username)
        if login_is_limited(attempt_key):
            flash(translate(session.get("language", "en"), "rate_limited"), "error")
            return render_template("login.html"), 429
        # Environment-based password is compared directly for simple demo setup.
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            language = session.get("language", "en")
            session.clear()
            session["language"] = language
            session["csrf_token"] = secrets.token_urlsafe(32)
            session.permanent = True
            session["role"] = "admin"
            session["admin_logged_in"] = True
            session["admin_username"] = username
            clear_login_failures(attempt_key)
            flash(translate(language, "login_welcome"), "success")
            return redirect(safe_next_url(request.args.get("next", ""), url_for("admin")))
        record_login_failure(attempt_key)
        flash(translate(session.get("language", "en"), "login_error"), "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    if session.get("role") != "admin" or not session.get("admin_logged_in"):
        abort(403)
    language = session.get("language", "en")
    session.clear()
    session["language"] = language
    session["csrf_token"] = secrets.token_urlsafe(32)
    flash(translate(language, "logout_success"), "success")
    return redirect(url_for("index"))


def valid_student_profile(full_name, email, enrollment_number, department, academic_year):
    if not all((full_name, email, enrollment_number, department, academic_year)):
        return "profile_required_error"
    if len(full_name) > 120 or len(email) > 254 or len(department) > 100 or len(academic_year) > 40:
        return "profile_length_error"
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "invalid_email"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/-]{2,31}", enrollment_number):
        return "enrollment_format_error"
    return None


@app.route("/student/register", methods=["GET", "POST"])
def student_register():
    if current_student_id():
        return redirect(url_for("student_dashboard"))
    if session.get("role") == "admin":
        abort(403)
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("college_email", "").strip().lower()
        enrollment_number = request.form.get("enrollment_number", "").strip().upper()
        department = request.form.get("department", "").strip()
        academic_year = request.form.get("academic_year", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        error = valid_student_profile(
            full_name, email, enrollment_number, department, academic_year
        )
        if error:
            flash(translate(session.get("language", "en"), error), "error")
        elif len(password) < 10 or len(password) > 128:
            flash(translate(session.get("language", "en"), "password_length"), "error")
        elif password != confirm_password:
            flash(translate(session.get("language", "en"), "password_mismatch"), "error")
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with get_db() as conn:
                    duplicate_email = conn.execute(
                        "SELECT 1 FROM students WHERE college_email = ?", (email,)
                    ).fetchone()
                    duplicate_enrollment = conn.execute(
                        "SELECT 1 FROM students WHERE enrollment_number = ?",
                        (enrollment_number,),
                    ).fetchone()
                    if duplicate_email:
                        registration_error = "duplicate_email"
                    elif duplicate_enrollment:
                        registration_error = "duplicate_enrollment"
                    else:
                        registration_error = None
                        conn.execute("""
                            INSERT INTO students
                            (full_name, college_email, enrollment_number, department,
                             academic_year, password_hash, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (full_name, email, enrollment_number, department,
                              academic_year, generate_password_hash(password), now, now))
            except sqlite3.IntegrityError:
                with get_db() as conn:
                    duplicate_email = conn.execute(
                        "SELECT 1 FROM students WHERE college_email = ?", (email,)
                    ).fetchone()
                if duplicate_email:
                    flash(translate(session.get("language", "en"), "duplicate_email"), "error")
                else:
                    flash(translate(session.get("language", "en"), "duplicate_enrollment"), "error")
            else:
                if registration_error:
                    flash(
                        translate(session.get("language", "en"), registration_error),
                        "error",
                    )
                else:
                    flash(
                        translate(session.get("language", "en"), "registration_success"),
                        "success",
                    )
                    return redirect(url_for("student_login"))
    return render_template("student_register.html")


@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    if current_student_id():
        return redirect(url_for("student_dashboard"))
    if session.get("role") == "admin":
        abort(403)
    next_url = request.args.get("next", "")
    show_signup_reminder = False
    if request.method == "POST":
        email = request.form.get("college_email", "").strip().lower()
        password = request.form.get("password", "")
        next_url = request.form.get("next", next_url)
        attempt_key = auth_attempt_key(email)
        limited = login_is_limited(attempt_key)
        if limited:
            flash(translate(session.get("language", "en"), "rate_limited"), "error")
            return render_template("student_login.html", next_url=next_url), 429
        with get_db() as conn:
            student = conn.execute("""
                SELECT id, full_name AS student_name, password_hash
                FROM students WHERE college_email = ?
            """, (email,)).fetchone()
        valid_password_length = 1 <= len(password) <= 128
        password_to_check = password if valid_password_length else ""
        password_hash = (
            student["password_hash"] if student and valid_password_length
            else DUMMY_STUDENT_PASSWORD_HASH
        )
        password_verified = check_password_hash(password_hash, password_to_check)
        password_ok = bool(student and valid_password_length and password_verified)
        if not limited and password_ok:
            language = session.get("language", "en")
            session.clear()
            session["language"] = language
            session["csrf_token"] = secrets.token_urlsafe(32)
            session.permanent = True
            session["role"] = "student"
            session["student_id"] = student["id"]
            session["student_name"] = student["student_name"]
            clear_login_failures(attempt_key)
            flash(translate(language, "login_success_student"), "success")
            return redirect(safe_next_url(next_url, url_for("student_dashboard")))
        record_login_failure(attempt_key)
        if student is None:
            show_signup_reminder = True
        else:
            flash(translate(session.get("language", "en"), "incorrect_password"), "error")
    return render_template(
        "student_login.html",
        next_url=next_url,
        show_signup_reminder=show_signup_reminder,
        submitted_email=request.form.get("college_email", "").strip(),
    )


@app.route("/student/logout", methods=["POST"])
@student_required
def student_logout():
    language = session.get("language", "en")
    session.clear()
    session["language"] = language
    session["csrf_token"] = secrets.token_urlsafe(32)
    flash(translate(language, "logout_success_student"), "success")
    return redirect(url_for("index"))


@app.route("/student/dashboard")
@student_required
def student_dashboard():
    student_id = current_student_id()
    with get_db() as conn:
        student = conn.execute("""
            SELECT id, full_name, college_email, enrollment_number, department,
                   academic_year, created_at
            FROM students WHERE id = ?
        """, (student_id,)).fetchone()
        if not student:
            session.clear()
            return redirect(url_for("student_login"))
        counts = {
            "total": conn.execute(
                "SELECT COUNT(*) FROM complaints WHERE student_id=?", (student_id,)
            ).fetchone()[0],
        }
        for key, status in (("pending", "Pending"), ("progress", "In Progress"),
                            ("resolved", "Resolved")):
            counts[key] = conn.execute(
                "SELECT COUNT(*) FROM complaints WHERE student_id=? AND status=?",
                (student_id, status),
            ).fetchone()[0]
        complaints = conn.execute("""
            SELECT id, category, priority, status, created_at, complaint_text
            FROM complaints WHERE student_id = ? ORDER BY id DESC LIMIT 10
        """, (student_id,)).fetchall()
    return render_template(
        "student_dashboard.html", student=student, complaints=complaints, **counts
    )


def get_student_password_hash(student_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT password_hash FROM students WHERE id = ?", (student_id,)
        ).fetchone()
    if not row:
        abort(401)
    return row["password_hash"]


@app.route("/student/profile", methods=["GET", "POST"])
@student_required
def student_profile():
    student_id = current_student_id()
    with get_db() as conn:
        student = conn.execute("""
            SELECT id, full_name, college_email, enrollment_number, department,
                   academic_year
            FROM students WHERE id = ?
        """, (student_id,)).fetchone()
    if not student:
        session.clear()
        return redirect(url_for("student_login"))
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("college_email", "").strip().lower()
        enrollment_number = request.form.get("enrollment_number", "").strip().upper()
        department = request.form.get("department", "").strip()
        academic_year = request.form.get("academic_year", "").strip()
        error = valid_student_profile(
            full_name, email, enrollment_number, department, academic_year
        )
        new_password = request.form.get("new_password", "")
        current_password = request.form.get("current_password", "")
        if error:
            flash(translate(session.get("language", "en"), error), "error")
        elif new_password and (len(new_password) < 10 or len(new_password) > 128):
            flash(translate(session.get("language", "en"), "password_length"), "error")
        elif new_password and not check_password_hash(
            get_student_password_hash(student_id), current_password
        ):
            flash(translate(session.get("language", "en"), "current_password_required"), "error")
        else:
            password_hash = (
                generate_password_hash(new_password) if new_password
                else get_student_password_hash(student_id)
            )
            try:
                with get_db() as conn:
                    conn.execute("""
                        UPDATE students SET full_name=?, college_email=?, enrollment_number=?,
                            department=?, academic_year=?, password_hash=?, updated_at=?
                        WHERE id=?
                    """, (full_name, email, enrollment_number, department, academic_year,
                          password_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          student_id))
            except sqlite3.IntegrityError:
                with get_db() as conn:
                    duplicate_email = conn.execute(
                        "SELECT 1 FROM students WHERE college_email=? AND id!=?",
                        (email, student_id),
                    ).fetchone()
                flash(translate(
                    session.get("language", "en"),
                    "duplicate_profile_email" if duplicate_email
                    else "duplicate_profile_enrollment"
                ), "error")
            else:
                session["student_name"] = full_name
                flash(translate(session.get("language", "en"), "profile_saved"), "success")
                return redirect(url_for("student_profile"))
        student = {
            "id": student_id, "full_name": full_name, "college_email": email,
            "enrollment_number": enrollment_number, "department": department,
            "academic_year": academic_year,
        }
    return render_template("student_profile.html", student=student)


@app.route("/student/complaints/<int:complaint_id>")
@student_required
@owner_required
def student_complaint_detail(complaint_id):
    with get_db() as conn:
        complaint = conn.execute("""
            SELECT id, category, priority, status, created_at, complaint_text,
                   feedback_rating, feedback_comment
            FROM complaints WHERE id=? AND student_id=?
        """, (complaint_id, current_student_id())).fetchone()
        timeline = conn.execute("""
            SELECT event_type, status, message, created_at
            FROM complaint_events WHERE complaint_id=? ORDER BY id
        """, (complaint_id,)).fetchall()
    return render_template(
        "student_complaint.html", complaint=complaint, timeline=timeline
    )


def load_consolidated_group(conn, public_id):
    group = conn.execute("""
        SELECT g.*, COUNT(c.id) AS report_count,
               COUNT(DISTINCT COALESCE(c.student_id, c.id)) AS student_count
        FROM complaint_groups AS g
        LEFT JOIN complaint_group_members AS gm ON gm.group_id = g.id
        LEFT JOIN complaints AS c ON c.id = gm.complaint_id
        WHERE g.public_id = ?
        GROUP BY g.id
    """, (public_id,)).fetchone()
    if not group:
        return None, []
    complaints = conn.execute("""
        SELECT c.id AS complaint_id, c.student_id, c.complaint_text, c.category,
               c.priority, c.status, c.created_at, gm.similarity_score
        FROM complaint_group_members AS gm
        JOIN complaints AS c ON c.id = gm.complaint_id
        WHERE gm.group_id = ?
        ORDER BY c.created_at, c.id
    """, (group["id"],)).fetchall()
    return group, complaints


def send_approved_group(public_id, retry=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    admin_id = session.get("admin_username")
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        group, complaints = load_consolidated_group(conn, public_id)
        if not group:
            return "missing"
        if retry:
            if group["approval_status"] != "approved" or group["email_status"] != "failed":
                return "not_retryable"
        elif group["approval_status"] != "pending_approval" or group["email_status"] != "pending_approval":
            return "not_pending"

        recipient = configured_department_recipient(group["category"])
        if is_demo_recipient(recipient):
            if retry:
                conn.execute("""
                    UPDATE complaint_groups
                    SET email_status = 'demo_mode', department_recipient = ?,
                        smtp_result = 'Demo Mode – Email Not Actually Sent',
                        updated_at = ?
                    WHERE id = ? AND approval_status = 'approved'
                      AND email_status = 'failed'
                """, (recipient, now, group["id"]))
            else:
                conn.execute("""
                    UPDATE complaint_groups
                    SET approval_status = 'approved', email_status = 'demo_mode',
                        approval_admin_id = ?, approved_at = ?, department_recipient = ?,
                        smtp_result = 'Demo Mode – Email Not Actually Sent',
                        updated_at = ?
                    WHERE id = ? AND approval_status = 'pending_approval'
                      AND email_status = 'pending_approval'
                """, (admin_id, now, recipient, now, group["id"]))
            conn.execute("""
                UPDATE complaints SET email_status = 'demo_mode', email_recipient = ?
                WHERE id IN (
                    SELECT complaint_id FROM complaint_group_members WHERE group_id = ?
                )
            """, (recipient, group["id"]))
            return "demo_mode"

        if retry:
            cursor = conn.execute("""
                UPDATE complaint_groups
                SET email_status = 'sending', department_recipient = ?,
                    send_attempts = send_attempts + 1, smtp_result = NULL, updated_at = ?
                WHERE id = ? AND approval_status = 'approved' AND email_status = 'failed'
            """, (recipient, now, group["id"]))
        else:
            cursor = conn.execute("""
                UPDATE complaint_groups
                SET approval_status = 'approved', email_status = 'sending',
                    approval_admin_id = ?, approved_at = ?, department_recipient = ?,
                    send_attempts = send_attempts + 1, smtp_result = NULL, updated_at = ?
                WHERE id = ? AND approval_status = 'pending_approval'
                  AND email_status = 'pending_approval'
            """, (admin_id, now, recipient, now, group["id"]))
        if cursor.rowcount != 1:
            return "not_pending"
        group, complaints = load_consolidated_group(conn, public_id)
        if not complaints:
            raise RuntimeError(f"Consolidated complaint group {public_id} has no complaints")
        notification_id = group["notification_id"]
        approval_timestamp = group["approved_at"]
        attempt_number = group["send_attempts"]
        conn.execute("""
            INSERT INTO complaint_notification_attempts
            (group_id, notification_id, attempt_number, recipient, started_at)
            VALUES (?, ?, ?, ?, ?)
        """, (group["id"], notification_id, attempt_number, recipient, now))

    subject, text, html = build_consolidated_email(group, complaints)
    sent, smtp_result = send_consolidated_email(
        recipient, notification_id, subject, text, html
    )
    final_status = "sent" if sent else "failed"
    finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        group_update = conn.execute("""
            UPDATE complaint_groups
            SET email_status = ?, sent_at = ?, smtp_result = ?, updated_at = ?
            WHERE public_id = ? AND email_status = 'sending'
              AND notification_id = ? AND approved_at = ?
        """, (
            final_status, finished_at if sent else None, smtp_result, finished_at,
            public_id, notification_id, approval_timestamp,
        ))
        if group_update.rowcount != 1:
            raise RuntimeError(
                f"Could not finalize notification batch {notification_id} after SMTP delivery"
            )
        attempt_update = conn.execute("""
            UPDATE complaint_notification_attempts
            SET completed_at = ?, smtp_result = ?, smtp_accepted = ?
            WHERE notification_id = ? AND attempt_number = ?
        """, (
            finished_at, smtp_result, int(sent), notification_id, attempt_number,
        ))
        if attempt_update.rowcount != 1:
            raise RuntimeError(
                f"Could not record SMTP result for notification batch {notification_id}"
            )
        conn.execute("""
            UPDATE complaints
            SET email_status = ?, email_recipient = ?
            WHERE id IN (
                SELECT complaint_id FROM complaint_group_members
                WHERE group_id = (SELECT id FROM complaint_groups WHERE public_id = ?)
            )
        """, (final_status, recipient, public_id))
    return final_status


@app.route("/admin/consolidated/<string:public_id>")
@admin_required
def consolidated_group_detail(public_id):
    with get_db() as conn:
        group, complaints = load_consolidated_group(conn, public_id)
        attempts = conn.execute("""
            SELECT attempt_number, recipient, started_at, completed_at,
                   smtp_result, smtp_accepted
            FROM complaint_notification_attempts
            WHERE group_id = (SELECT id FROM complaint_groups WHERE public_id = ?)
            ORDER BY attempt_number
        """, (public_id,)).fetchall()
    if not group:
        abort(404)
    return render_template(
        "consolidated_group.html", group=group, complaints=complaints,
        attempts=attempts,
        department_recipient=configured_department_recipient(group["category"]),
        department_name=DEPARTMENT_NAMES.get(group["category"], "Department"),
    )


@app.route("/admin/consolidated/<string:public_id>/preview")
@admin_required
def consolidated_group_preview(public_id):
    with get_db() as conn:
        group, complaints = load_consolidated_group(conn, public_id)
    if not group:
        abort(404)
    subject, text, html = build_consolidated_email(group, complaints)
    recipient = configured_department_recipient(group["category"])
    return render_template(
        "consolidated_email_preview.html", group=group, complaints=complaints,
        email_subject=subject, email_text=text, email_html=html,
        recipient=recipient, department_name=DEPARTMENT_NAMES.get(
            group["category"], "Department"
        ),
    )


@app.route("/admin/consolidated/<string:public_id>/edit", methods=["POST"])
@admin_required
def edit_consolidated_group(public_id):
    title = request.form.get("issue_title", "").strip()
    summary = request.form.get("summary", "").strip()
    if (
        not title or not summary or len(title) > 160 or len(summary) > 2000
        or any(ord(char) < 32 or ord(char) == 127 for char in title)
    ):
        flash("Enter an issue title (up to 160 characters) and summary (up to 2,000 characters).", "error")
        return redirect(url_for("consolidated_group_detail", public_id=public_id))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cursor = conn.execute("""
            UPDATE complaint_groups SET issue_title = ?, summary = ?, updated_at = ?
            WHERE public_id = ? AND approval_status = 'pending_approval'
              AND email_status = 'pending_approval'
        """, (title, summary, now, public_id))
    if cursor.rowcount:
        flash("Consolidated issue updated. Original student complaints were not changed.", "success")
    else:
        flash("Only pending groups can be edited.", "error")
    return redirect(url_for("consolidated_group_detail", public_id=public_id))


@app.route("/admin/consolidated/<string:public_id>/approve", methods=["POST"])
@admin_required
def approve_consolidated_group(public_id):
    result = send_approved_group(public_id)
    if result == "missing":
        abort(404)
    messages = {
        "sent": ("Consolidated notification sent successfully.", "success"),
        "failed": ("Approval was recorded, but email delivery failed. You may retry this batch.", "error"),
        "demo_mode": ("Demo Mode – Email Not Actually Sent", "success"),
        "not_pending": ("This group is no longer awaiting approval and was not sent again.", "error"),
    }
    message, category = messages.get(result, ("The notification could not be processed.", "error"))
    flash(message, category)
    return redirect(url_for("consolidated_group_detail", public_id=public_id))


@app.route("/admin/consolidated/<string:public_id>/retry", methods=["POST"])
@admin_required
def retry_consolidated_group(public_id):
    result = send_approved_group(public_id, retry=True)
    if result == "missing":
        abort(404)
    messages = {
        "sent": ("Consolidated notification retry sent successfully.", "success"),
        "failed": ("Email delivery failed again. The approved batch remains available for retry.", "error"),
        "demo_mode": ("Demo Mode – Email Not Actually Sent", "error"),
        "not_retryable": ("Only an approved notification that failed delivery can be retried.", "error"),
    }
    message, category = messages.get(result, ("The notification could not be processed.", "error"))
    flash(message, category)
    return redirect(url_for("consolidated_group_detail", public_id=public_id))


@app.route("/admin/consolidated/<string:public_id>/reject", methods=["POST"])
@admin_required
def reject_consolidated_group(public_id):
    reason = request.form.get("reason", "").strip()
    if not reason or len(reason) > 1000:
        flash("A rejection reason is required (up to 1,000 characters).", "error")
        return redirect(url_for("consolidated_group_detail", public_id=public_id))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cursor = conn.execute("""
            UPDATE complaint_groups
            SET approval_status = 'rejected', email_status = 'rejected',
                rejected_by_admin_id = ?, rejected_at = ?, rejection_reason = ?,
                smtp_result = 'Rejected by administrator', updated_at = ?
            WHERE public_id = ? AND approval_status = 'pending_approval'
              AND email_status = 'pending_approval'
        """, (session.get("admin_username"), now, reason, now, public_id))
        if cursor.rowcount:
            conn.execute("""
                UPDATE complaints SET email_status = 'rejected'
                WHERE id IN (
                    SELECT complaint_id FROM complaint_group_members
                    WHERE group_id = (SELECT id FROM complaint_groups WHERE public_id = ?)
                )
            """, (public_id,))
    if cursor.rowcount:
        flash("Consolidated notification rejected. No department email was sent.", "success")
    else:
        flash("Only a pending group can be rejected.", "error")
    return redirect(url_for("consolidated_group_detail", public_id=public_id))


@app.route("/admin")
@admin_required
def admin():
    search = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip()
    priority = request.args.get("priority", "").strip()
    status = request.args.get("status", "").strip()
    date_range = request.args.get("range", "").strip()

    clauses, params = [], []
    if search:
        clauses.append("(CAST(id AS TEXT) LIKE ? OR student_name LIKE ? OR complaint_text LIKE ? OR department LIKE ?)")
        term = f"%{search}%"
        params.extend([term, term, term, term])
    if category in CATEGORIES:
        category_names = (category, *CATEGORY_ALIASES[category])
        placeholders = ",".join("?" for _ in category_names)
        clauses.append(f"category IN ({placeholders})")
        params.extend(category_names)
    if priority in PRIORITIES:
        clauses.append("priority = ?")
        params.append(priority)
    if status in STATUSES:
        clauses.append("status = ?")
        params.append(status)
    if date_range in {"today", "week", "month"}:
        now = datetime.now()
        if date_range == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_range == "week":
            start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        clauses.append("created_at >= ?")
        params.append(start.strftime("%Y-%m-%d %H:%M:%S"))

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_db() as conn:
        complaints = conn.execute(
            "SELECT * FROM complaints" + where + " ORDER BY id DESC", params
        ).fetchall()
        attachments_by_complaint = {}
        if complaints:
            ids = [row["id"] for row in complaints]
            placeholders = ",".join("?" for _ in ids)
            attachments = conn.execute(f"""
                SELECT id, complaint_id, original_name
                FROM complaint_attachments
                WHERE complaint_id IN ({placeholders})
                ORDER BY id
            """, ids).fetchall()
            for attachment in attachments:
                attachments_by_complaint.setdefault(attachment["complaint_id"], []).append(attachment)
        total = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Pending'").fetchone()[0]
        progress = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='In Progress'").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM complaints WHERE status='Resolved'").fetchone()[0]
        high = conn.execute("SELECT COUNT(*) FROM complaints WHERE priority='High'").fetchone()[0]
        avg_rating = conn.execute("SELECT AVG(feedback_rating) FROM complaints WHERE feedback_rating IS NOT NULL").fetchone()[0]
        stored_category_rows = conn.execute(
            "SELECT category, COUNT(*) AS count FROM complaints GROUP BY category ORDER BY count DESC"
        ).fetchall()
        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM complaints GROUP BY status"
        ).fetchall()
        priority_rows = conn.execute(
            "SELECT priority, COUNT(*) AS count FROM complaints GROUP BY priority"
        ).fetchall()
        avg_resolution = conn.execute("""
            SELECT AVG((julianday(updated_at)-julianday(created_at))*24)
            FROM complaints WHERE status='Resolved' AND updated_at IS NOT NULL
        """).fetchone()[0]
        pending_groups = conn.execute("""
            SELECT g.*, COUNT(c.id) AS report_count,
                   COUNT(DISTINCT COALESCE(c.student_id, c.id)) AS student_count,
                   GROUP_CONCAT(c.id, ', ') AS complaint_ids
            FROM complaint_groups AS g
            LEFT JOIN complaint_group_members AS gm ON gm.group_id = g.id
            LEFT JOIN complaints AS c ON c.id = gm.complaint_id
            WHERE g.approval_status = 'pending_approval'
            GROUP BY g.id
            ORDER BY g.created_at DESC
        """).fetchall()
        group_analytics = {
            "total": conn.execute("SELECT COUNT(*) FROM complaint_groups").fetchone()[0],
            "awaiting": conn.execute(
                "SELECT COUNT(*) FROM complaint_groups WHERE approval_status='pending_approval'"
            ).fetchone()[0],
            "approved": conn.execute(
                "SELECT COUNT(*) FROM complaint_groups WHERE approval_status='approved'"
            ).fetchone()[0],
            "sent": conn.execute(
                "SELECT COUNT(*) FROM complaint_groups WHERE email_status='sent'"
            ).fetchone()[0],
            "failed": conn.execute(
                "SELECT COUNT(*) FROM complaint_groups WHERE email_status='failed'"
            ).fetchone()[0],
        }
        frequent_issues = conn.execute("""
            SELECT g.issue_title, g.category, COUNT(c.id) AS report_count
            FROM complaint_groups AS g
            JOIN complaint_group_members AS gm ON gm.group_id = g.id
            JOIN complaints AS c ON c.id = gm.complaint_id
            GROUP BY g.issue_title, g.category
            ORDER BY report_count DESC, g.issue_title
            LIMIT 5
        """).fetchall()
        department_counts = conn.execute("""
            SELECT g.category, COUNT(c.id) AS report_count
            FROM complaint_groups AS g
            JOIN complaint_group_members AS gm ON gm.group_id = g.id
            JOIN complaints AS c ON c.id = gm.complaint_id
            GROUP BY g.category
            ORDER BY report_count DESC, g.category
            LIMIT 5
        """).fetchall()

    category_counts = {}
    for row in stored_category_rows:
        category = canonical_category(row["category"])
        category_counts[category] = category_counts.get(category, 0) + row["count"]
    category_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
    chart_data = {
        "categories": [translate(session.get("language", "en"), category_key(category)) for category in category_counts],
        "categoryCounts": list(category_counts.values()),
        "statuses": [translate(session.get("language", "en"), r["status"].lower().replace(" ", "_")) for r in status_rows],
        "statusCounts": [r["count"] for r in status_rows],
        "priorities": [translate(session.get("language", "en"), r["priority"].lower()) for r in priority_rows],
        "priorityCounts": [r["count"] for r in priority_rows],
    }
    return render_template(
        "admin.html", complaints=complaints, total=total, pending=pending,
        progress=progress, resolved=resolved, high=high, avg_rating=avg_rating,
        avg_resolution=avg_resolution, chart_data=chart_data,
        pending_groups=pending_groups, group_analytics=group_analytics,
        frequent_issues=frequent_issues, department_counts=department_counts,
        DEPARTMENT_NAMES=DEPARTMENT_NAMES,
        categories=CATEGORIES, priorities=PRIORITIES, statuses=STATUSES,
        attachments_by_complaint=attachments_by_complaint,
        filters={"q": search, "category": category, "priority": priority,
                 "status": status, "range": date_range}
    )


@app.route("/admin/update/<int:complaint_id>", methods=["POST"])
@admin_required
def update_status(complaint_id):
    status = request.form.get("status", "")
    if status not in STATUSES:
        flash(translate(session.get("language", "en"), "invalid_status"), "error")
        return redirect(url_for("admin"))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        complaint = conn.execute(
            "SELECT status FROM complaints WHERE id = ?", (complaint_id,)
        ).fetchone()
        if not complaint:
            flash(translate(session.get("language", "en"), "complaint_missing_error"), "error")
            return redirect(url_for("admin"))
        conn.execute("UPDATE complaints SET status = ?, updated_at = ? WHERE id = ?",
                     (status, now, complaint_id))
        if complaint["status"] != status:
            conn.execute("""
                INSERT INTO complaint_events
                (complaint_id, event_type, status, message, created_at)
                VALUES (?, 'status', ?, ?, ?)
            """, (complaint_id, status, f"Status updated to {status}", now))
    flash(translate(session.get("language", "en"), "status_updated",
                    id=complaint_id,
                    status=translate(session.get("language", "en"), status.lower().replace(" ", "_"))),
          "success")
    return redirect(url_for("admin", **request.args))


@app.route("/admin/export.csv")
@admin_required
def export_csv():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, student_name, student_email, department, complaint_text, category,
                   priority, status, created_at, updated_at, confidence, feedback_rating,
                   feedback_comment, anonymous
            FROM complaints ORDER BY id DESC
        """).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Student", "Email", "Department", "Complaint", "Category",
                     "Priority", "Status", "Created At", "Updated At",
                     "Model Confidence", "Feedback Rating", "Feedback Comment", "Anonymous"])
    for row in rows:
        writer.writerow(list(row))
    return Response(
        "\ufeff" + output.getvalue(), mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=campuscare_complaints.csv"}
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(413)
def request_too_large(_error):
    flash(translate(session.get("language", "en"), "request_too_large"), "error")
    student_id = current_student_id()
    if student_id:
        with get_db() as conn:
            student = conn.execute(
                "SELECT full_name, college_email, department FROM students WHERE id = ?",
                (student_id,),
            ).fetchone()
        if student:
            return render_template(
                "index.html", categories=CATEGORIES, student=student
            ), 413
        session.clear()
        return redirect(url_for("student_login"))
    return render_template("landing.html"), 413


@app.cli.command("escalate-complaints")
def escalate_complaints_command():
    count = escalate_overdue_complaints()
    print(f"Escalated {count} overdue complaint(s).")


init_db()

if __name__ == "__main__":
    app.run(debug=True)
