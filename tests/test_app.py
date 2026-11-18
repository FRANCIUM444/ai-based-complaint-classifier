import io
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import app as campuscare


class CampusCareFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = campuscare.DB_PATH
        self.original_upload_folder = campuscare.UPLOAD_FOLDER
        campuscare.DB_PATH = os.path.join(self.temp_dir.name, "test.db")
        campuscare.UPLOAD_FOLDER = os.path.join(self.temp_dir.name, "uploads")
        campuscare._last_escalation_check = 0
        campuscare.app.config.update(TESTING=True)
        campuscare.init_db()
        self.client = campuscare.app.test_client()

    def tearDown(self):
        campuscare.DB_PATH = self.original_db_path
        campuscare.UPLOAD_FOLDER = self.original_upload_folder
        campuscare._last_escalation_check = 0
        self.temp_dir.cleanup()

    def csrf_token(self, client=None):
        client = client or self.client
        client.get("/")
        with client.session_transaction() as state:
            return state["csrf_token"]

    def post_form(self, path, data=None, client=None, **kwargs):
        client = client or self.client
        values = dict(data or {})
        values["csrf_token"] = self.csrf_token(client)
        return client.post(path, data=values, **kwargs)

    def register_student(self, client=None, email="student@example.edu",
                         enrollment="CSE2024001", name="Test Student"):
        client = client or self.client
        response = self.post_form("/student/register", {
            "full_name": name,
            "college_email": email,
            "enrollment_number": enrollment,
            "department": "CSE",
            "academic_year": "2nd year",
            "password": "Secure-test-pass-1",
            "confirm_password": "Secure-test-pass-1",
        }, client=client)
        return response

    def login_student(self, client=None, email="student@example.edu"):
        client = client or self.client
        return self.post_form("/student/login", {
            "college_email": email,
            "password": "Secure-test-pass-1",
        }, client=client)

    def create_student(self, client=None, email="student@example.edu",
                       enrollment="CSE2024001", name="Test Student"):
        client = client or self.client
        self.register_student(client, email, enrollment, name)
        return self.login_student(client, email)

    def submit(self, complaint_text, extra=None, client=None):
        client = client or self.client
        data = {"complaint_text": complaint_text}
        data.update(extra or {})
        return self.post_form(
            "/submit", data=data, client=client, follow_redirects=True
        )

    def seed_complaint(self, status="Pending", created_at=None, priority="Low",
                       student_id=None):
        created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with campuscare.get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO complaints
                (student_name, complaint_text, category, priority, status, created_at,
                 updated_at, student_id)
                VALUES ('Test Student', 'Water supply is not working in the hostel block',
                        'Hostel', ?, ?, ?, ?, ?)
            """, (priority, status, created_at, created_at, student_id))
            complaint_id = cursor.lastrowid
            conn.execute("""
                INSERT INTO complaint_events
                (complaint_id, event_type, status, message, created_at)
                VALUES (?, 'status', ?, 'Complaint submitted', ?)
            """, (complaint_id, status, created_at))
        return complaint_id

    def login_admin(self, client=None):
        client = client or self.client
        return self.post_form("/login", {
            "username": campuscare.ADMIN_USERNAME,
            "password": campuscare.ADMIN_PASSWORD,
        }, client=client)

    def test_student_registration_hashes_password_and_rejects_duplicates(self):
        mismatched = self.post_form("/student/register", {
            "full_name": "Test Student",
            "college_email": "student@example.edu",
            "enrollment_number": "CSE2024001",
            "department": "CSE",
            "academic_year": "2nd year",
            "password": "Secure-test-pass-1",
            "confirm_password": "different-pass-1",
        })
        self.assertIn(b"passwords do not match", mismatched.data)
        invalid_email = self.post_form("/student/register", {
            "full_name": "Test Student",
            "college_email": "not-an-email",
            "enrollment_number": "CSE2024001",
            "department": "CSE",
            "academic_year": "2nd year",
            "password": "Secure-test-pass-1",
            "confirm_password": "Secure-test-pass-1",
        })
        self.assertIn(b"valid college email", invalid_email.data)
        response = self.register_student()
        self.assertEqual(response.status_code, 302)
        with campuscare.get_db() as conn:
            student = conn.execute(
                "SELECT password_hash FROM students WHERE college_email=?",
                ("student@example.edu",),
            ).fetchone()
        self.assertTrue(campuscare.check_password_hash(
            student["password_hash"], "Secure-test-pass-1"
        ))
        self.assertNotEqual(student["password_hash"], "Secure-test-pass-1")

        duplicate_email = self.register_student(
            email="STUDENT@example.edu", enrollment="CSE2024002"
        )
        self.assertEqual(duplicate_email.status_code, 200)
        self.assertIn(b"already uses this email", duplicate_email.data)
        duplicate_enrollment = self.register_student(
            email="other@example.edu", enrollment="cse2024001"
        )
        self.assertEqual(duplicate_enrollment.status_code, 200)
        self.assertIn(b"already uses this enrollment", duplicate_enrollment.data)

    def test_complete_student_registration_login_logout_flow(self):
        response = self.register_student()
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/student/login")
        with self.client.session_transaction() as state:
            self.assertNotEqual(state.get("role"), "student")
            self.assertNotIn("student_id", state)

        login_page = self.client.get(response.location, follow_redirects=True)
        self.assertEqual(login_page.status_code, 200)
        self.assertIn(
            b"Your CampusCare account has been created successfully! "
            b"Please log in to continue.",
            login_page.data,
        )
        with campuscare.get_db() as conn:
            student = conn.execute("""
                SELECT id, full_name, college_email, password_hash
                FROM students WHERE college_email = ?
            """, ("student@example.edu",)).fetchone()
            login_row = conn.execute("""
                SELECT id, full_name AS student_name, password_hash
                FROM students WHERE college_email = ?
            """, ("student@example.edu",)).fetchone()
        self.assertIsNotNone(student)
        self.assertEqual(
            login_row.keys(), ["id", "student_name", "password_hash"]
        )
        self.assertEqual(student["full_name"], "Test Student")
        self.assertTrue(campuscare.check_password_hash(
            student["password_hash"], "Secure-test-pass-1"
        ))

        login = self.login_student()
        self.assertEqual(login.status_code, 302)
        self.assertEqual(login.location, "/student/dashboard")
        with self.client.session_transaction() as state:
            self.assertEqual(state["role"], "student")
            self.assertEqual(state["student_id"], student["id"])
            self.assertEqual(state["student_name"], "Test Student")
        dashboard = self.client.get(login.location)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Welcome, Test Student", dashboard.data)

        logout = self.post_form("/student/logout", follow_redirects=True)
        self.assertEqual(logout.status_code, 200)
        with self.client.session_transaction() as state:
            self.assertNotIn("role", state)
            self.assertNotIn("student_id", state)
            self.assertNotIn("student_name", state)
        self.assertEqual(self.client.get("/student/dashboard").status_code, 302)

    def test_student_login_logout_and_session_rotation(self):
        self.register_student()
        with self.client.session_transaction() as state:
            old_csrf = state["csrf_token"]
            state["pre_login_marker"] = "discard this"
        response = self.login_student()
        self.assertEqual(response.status_code, 302)
        self.assertIn("/student/dashboard", response.location)
        session_cookie = response.headers["Set-Cookie"]
        self.assertIn("HttpOnly", session_cookie)
        self.assertIn("SameSite=Lax", session_cookie)
        self.assertIn("Expires=", session_cookie)
        if campuscare.app.config["SESSION_COOKIE_SECURE"]:
            self.assertIn("Secure", session_cookie)
        with self.client.session_transaction() as state:
            self.assertEqual(state["role"], "student")
            self.assertEqual(state["student_id"], 1)
            self.assertEqual(state["student_name"], "Test Student")
            self.assertNotIn("pre_login_marker", state)
            self.assertNotEqual(state["csrf_token"], old_csrf)
            self.assertTrue(state.permanent)
        dashboard = self.client.get("/student/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Welcome, Test Student", dashboard.data)

        invalid_client = campuscare.app.test_client()
        self.register_student(invalid_client, "second@example.edu", "CSE2024002")
        bad_login = self.post_form("/student/login", {
            "college_email": "second@example.edu",
            "password": "wrong-password",
        }, client=invalid_client)
        self.assertEqual(bad_login.status_code, 200)
        self.assertIn(b"Incorrect password. Please try again.", bad_login.data)
        self.assertNotIn(b"signupReminder", bad_login.data)

        logged_out = self.post_form("/student/logout", client=self.client)
        self.assertEqual(logged_out.status_code, 302)
        self.assertEqual(self.client.get("/student/dashboard").status_code, 302)

    def test_unregistered_email_displays_signup_modal_and_registration_link(self):
        response = self.post_form("/student/login", {
            "college_email": "new-student@example.edu",
            "password": "any-password",
        })
        self.assertEqual(response.status_code, 200)
        rendered_html = response.data.decode("utf-8").replace("&#39;", "'")
        for text in (
            "Welcome to CampusCare!",
            "Oops! It looks like you don't have an account yet.",
            "Please create your student account first to access CampusCare features, submit complaints and track their progress.",
            "Create Account",
            "Try Again",
            "Already have an account?",
            "Log in here.",
        ):
            self.assertIn(text, rendered_html)
        self.assertIn(b'href="/student/register"', response.data)
        self.assertIn(b'id="signupReminder"', response.data)
        self.assertIn(b'id="signupTryAgain"', response.data)
        self.assertIn(b"modal.hidden = true", response.data)
        self.assertIn(b"email.focus()", response.data)
        self.assertEqual(self.client.get("/student/register").status_code, 200)
        with campuscare.get_db() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM students").fetchone()[0], 0)
        self.assertEqual(self.client.get("/student/dashboard").status_code, 302)

    def test_student_login_rate_limit_and_csrf(self):
        self.register_student()
        missing_csrf = self.client.post("/student/login", data={
            "college_email": "student@example.edu",
            "password": "wrong-password",
        })
        self.assertEqual(missing_csrf.status_code, 400)
        for _ in range(campuscare.LOGIN_MAX_ATTEMPTS):
            response = self.post_form("/student/login", {
                "college_email": "student@example.edu",
                "password": "wrong-password",
            })
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Incorrect password. Please try again.", response.data)
        limited = self.post_form("/student/login", {
            "college_email": "student@example.edu",
            "password": "wrong-password",
        })
        self.assertEqual(limited.status_code, 429)

    def test_student_routes_are_private_and_return_to_requested_page(self):
        paths = (
            "/submit", "/track", "/student/dashboard", "/student/profile",
            "/student/complaints/1", "/api/complaints/1/timeline",
        )
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302, path)
            self.assertIn("/student/login", response.location)
        response = self.client.get("/student/profile?tab=details")
        self.assertIn("next=", response.location)
        self.register_student()
        login = self.post_form("/student/login", {
            "college_email": "student@example.edu",
            "password": "Secure-test-pass-1",
            "next": "/student/profile",
        })
        self.assertEqual(login.status_code, 302)
        self.assertEqual(login.location, "/student/profile")

    def test_students_cannot_access_other_student_complaints_or_admin(self):
        self.create_student()
        with campuscare.get_db() as conn:
            owner_id = conn.execute(
                "SELECT id FROM students WHERE college_email='student@example.edu'"
            ).fetchone()["id"]
        complaint_id = self.seed_complaint(student_id=owner_id)
        other_client = campuscare.app.test_client()
        self.create_student(other_client, "other@example.edu", "CSE2024002", "Other Student")
        self.assertEqual(
            other_client.get(f"/student/complaints/{complaint_id}").status_code, 404
        )
        self.assertEqual(
            other_client.get(f"/api/complaints/{complaint_id}/timeline").status_code, 404
        )
        search = self.post_form(
            "/track", {"complaint_id": str(complaint_id)}, client=other_client
        )
        self.assertNotIn(b"Water supply is not working", search.data)
        self.assertEqual(other_client.get("/admin").status_code, 403)
        self.assertEqual(other_client.get("/admin/export.csv").status_code, 403)
        self.assertEqual(other_client.get("/login").status_code, 403)
        self.assertEqual(
            other_client.get(f"/student/complaints/{complaint_id}").status_code, 404
        )
        self.assertEqual(
            self.client.get("/student/profile").status_code, 200
        )

    def test_student_profile_and_complaint_submission_use_account_identity(self):
        self.create_student()
        profile = self.client.get("/student/profile")
        self.assertEqual(profile.status_code, 200)
        self.assertNotIn(b"password_hash", profile.data)
        text = "The classroom projector is not working correctly"
        response = self.post_form("/submit", {
            "student_name": "Impersonated Name",
            "student_email": "attacker@example.net",
            "department": "Fake Department",
            "complaint_text": text,
        })
        self.assertEqual(response.status_code, 200)
        with campuscare.get_db() as conn:
            complaint = conn.execute("""
                SELECT id, student_name, student_email, department, student_id
                FROM complaints WHERE complaint_text=?
            """, (text,)).fetchone()
        self.assertEqual(complaint["student_name"], "Test Student")
        self.assertEqual(complaint["student_email"], "student@example.edu")
        self.assertEqual(complaint["department"], "CSE")
        self.assertIsNotNone(complaint["student_id"])
        detail = self.client.get(f"/student/complaints/{complaint['id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(text.encode(), detail.data)
        dashboard = self.client.get("/student/dashboard")
        self.assertIn(b"stat-total", dashboard.data)
        self.assertIn(b"<strong>1</strong>", dashboard.data)

    def test_student_can_update_profile_and_password(self):
        self.create_student()
        response = self.post_form("/student/profile", {
            "full_name": "Updated Student",
            "college_email": "student@example.edu",
            "enrollment_number": "CSE2024001",
            "department": "Computer Science",
            "academic_year": "3rd year",
            "current_password": "Secure-test-pass-1",
            "new_password": "A-different-test-pass-2",
        })
        self.assertEqual(response.status_code, 302)
        with campuscare.get_db() as conn:
            student = conn.execute(
                "SELECT full_name, password_hash FROM students WHERE college_email=?",
                ("student@example.edu",),
            ).fetchone()
        self.assertEqual(student["full_name"], "Updated Student")
        self.assertTrue(campuscare.check_password_hash(
            student["password_hash"], "A-different-test-pass-2"
        ))
        new_client = campuscare.app.test_client()
        new_login = self.post_form("/student/login", {
            "college_email": "student@example.edu",
            "password": "A-different-test-pass-2",
        }, client=new_client)
        self.assertEqual(new_login.status_code, 302)

    def test_admin_login_remains_separate_from_student_role(self):
        admin_client = campuscare.app.test_client()
        self.assertEqual(self.login_admin(admin_client).status_code, 302)
        self.assertEqual(admin_client.get("/admin").status_code, 200)
        self.assertEqual(admin_client.get("/student/dashboard").status_code, 403)
        self.assertEqual(
            self.post_form("/logout", client=admin_client).status_code, 302
        )
        self.assertEqual(admin_client.get("/admin").status_code, 302)
        student_client = campuscare.app.test_client()
        self.create_student(student_client)
        self.assertEqual(student_client.get("/admin").status_code, 403)

    def test_existing_pages_render_and_language_switches(self):
        for path in ("/", "/about", "/login", "/student/login", "/student/register"):
            self.assertEqual(self.client.get(path).status_code, 200)
        self.assertEqual(self.client.get("/track").status_code, 302)
        response = self.post_form(
            "/language/hi", data={"next": "/"}, follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("शिकायत भेजें".encode(), response.data)
        chatbot = self.client.post("/assistant", json={
            "message": "शिकायत कैसे भेजें", "language": "hi"
        }, headers={"X-CSRF-Token": self.csrf_token()})
        self.assertEqual(chatbot.status_code, 200)
        self.assertIn("लॉगिन करें".encode(), chatbot.json["reply"].encode())
        self.assertEqual(
            campuscare.classify_multilingual_complaint("वाई-फाई काम नहीं कर रहा")[0],
            "IT / Technical",
        )

    def test_category_names_legacy_records_and_department_routing(self):
        self.assertEqual(campuscare.CATEGORIES, [
            "Academics", "Infrastructure", "IT / Technical", "Sanitation",
            "Canteen", "Hostel", "Transportation", "Administration", "Other",
        ])
        self.assertEqual(campuscare.canonical_category("IT & Wi-Fi"), "IT / Technical")
        self.assertEqual(campuscare.category_key("Water & Sanitation"), "sanitation")

        expected_recipients = {
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
        self.assertEqual(campuscare.DEPARTMENT_EMAILS, expected_recipients)
        for category, recipient in expected_recipients.items():
            self.assertEqual(campuscare.configured_department_recipient(category), recipient)
            self.assertTrue(campuscare.is_demo_recipient(recipient))

        recipients = {
            category: f"{index}@college.edu"
            for index, category in enumerate(campuscare.CATEGORIES)
        }
        with patch.dict(campuscare.DEPARTMENT_EMAILS, recipients):
            for category, recipient in recipients.items():
                self.assertEqual(
                    campuscare.configured_department_recipient(category), recipient
                )
            for legacy, canonical in campuscare.LEGACY_CATEGORY_NAMES.items():
                self.assertEqual(
                    campuscare.configured_department_recipient(legacy),
                    recipients[canonical],
                )

        with campuscare.get_db() as conn:
            conn.execute("""
                INSERT INTO complaints
                (student_name, complaint_text, category, priority, status, created_at, updated_at)
                VALUES ('Legacy Student', 'Legacy Wi-Fi complaint', 'IT & Wi-Fi',
                        'Low', 'Pending', '2026-01-01 10:00:00', '2026-01-01 10:00:00')
            """)
        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        filtered = admin_client.get("/admin?category=IT%20%2F%20Technical")
        self.assertEqual(filtered.status_code, 200)
        self.assertIn(b"Legacy Wi-Fi complaint", filtered.data)
        self.assertIn(b"IT / Technical", filtered.data)
        self.assertIn(b'"categories": ["IT / Technical"]', filtered.data)
        self.assertIn(b'"categoryCounts": [1]', filtered.data)

    def test_admin_dashboard_and_csv_keep_authentication(self):
        self.seed_complaint()
        self.assertEqual(self.client.get("/admin").status_code, 302)
        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        dashboard = admin_client.get("/admin")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"TOTAL COMPLAINTS", dashboard.data)
        self.post_form("/language/hi", {"next": "/admin"}, admin_client)
        hindi_dashboard = admin_client.get("/admin")
        self.assertEqual(hindi_dashboard.status_code, 200)
        self.assertIn("कुल शिकायतें".encode(), hindi_dashboard.data)
        csv_export = admin_client.get("/admin/export.csv")
        self.assertEqual(csv_export.status_code, 200)
        self.assertIn("campuscare_complaints.csv", csv_export.headers["Content-Disposition"])

    def test_submission_preserves_classification_email_and_duplicate_warning(self):
        self.create_student()
        text = "The hostel water supply is not working in our block"
        first = self.submit(text)
        self.assertEqual(first.status_code, 200)
        second = self.submit(text)
        self.assertEqual(second.status_code, 200)
        self.assertIn(b"Possible duplicate", second.data)
        with campuscare.get_db() as conn:
            rows = conn.execute(
                "SELECT id, category, priority, duplicate_of FROM complaints ORDER BY id"
            ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["duplicate_of"], rows[0]["id"])
        self.assertTrue(rows[0]["category"])
        self.assertIn(b"Pending administrator approval", second.data)
        with campuscare.get_db() as conn:
            groups = conn.execute("""
                SELECT g.manual_review, COUNT(gm.id) AS report_count
                FROM complaint_groups AS g
                JOIN complaint_group_members AS gm ON gm.group_id = g.id
                GROUP BY g.id ORDER BY g.id
            """).fetchall()
        self.assertEqual(len(groups), 2)
        self.assertEqual([group["report_count"] for group in groups], [1, 1])
        self.assertTrue(all(group["manual_review"] for group in groups))

    def test_room_114_fan_reports_wait_for_admin_and_send_one_email(self):
        examples = [
            "Fan not working in Room 114.",
            "Room 114 fan is not working.",
            "Ceiling fan not working in classroom 114.",
            "Fan problem in Room 114.",
        ]
        students = []
        with patch.dict(
            campuscare.DEPARTMENT_EMAILS,
            {"Infrastructure": "maintenance@college.edu"},
        ), patch.dict(os.environ, {
            "SMTP_HOST": "smtp.college.edu",
            "SMTP_FROM": "campuscare@college.edu",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "",
        }):
            smtp_server = MagicMock()
            smtp_server.send_message.return_value = {}
            smtp_context = MagicMock()
            smtp_context.__enter__.return_value = smtp_server
            with patch.object(campuscare.smtplib, "SMTP", return_value=smtp_context) as smtp:
                complaint_ids = []
                for index, text in enumerate(examples, 1):
                    student = campuscare.app.test_client()
                    self.create_student(
                        student,
                        email=f"student{index}@example.edu",
                        enrollment=f"CSE20240{index}",
                        name=f"Student {index}",
                    )
                    students.append(student)
                    result = self.post_form(
                        "/submit", {"complaint_text": text}, client=student
                    )
                    self.assertEqual(result.status_code, 200)
                    with campuscare.get_db() as conn:
                        complaint_ids.append(conn.execute(
                            "SELECT id FROM complaints WHERE complaint_text = ?",
                            (text,),
                        ).fetchone()["id"])
                smtp.assert_not_called()

                with campuscare.get_db() as conn:
                    groups = conn.execute(
                        "SELECT id, public_id, notification_id FROM complaint_groups"
                    ).fetchall()
                    member_count = conn.execute(
                        "SELECT COUNT(*) FROM complaint_group_members"
                    ).fetchone()[0]
                    saved_complaints = conn.execute(
                        "SELECT id, complaint_text, status FROM complaints ORDER BY id"
                    ).fetchall()
                self.assertEqual(len(groups), 1)
                self.assertEqual(member_count, 4)
                self.assertEqual([row["id"] for row in saved_complaints], complaint_ids)
                self.assertTrue(all(row["status"] == "Pending" for row in saved_complaints))

                admin_client = campuscare.app.test_client()
                self.login_admin(admin_client)
                dashboard = admin_client.get("/admin")
                self.assertIn(b"Consolidated Complaints", dashboard.data)
                self.assertIn(b"4 reports", dashboard.data)
                public_id = groups[0]["public_id"]
                detail = admin_client.get(f"/admin/consolidated/{public_id}")
                self.assertEqual(detail.status_code, 200)
                self.assertTrue(all(f"#{complaint_id}".encode() in detail.data
                                    for complaint_id in complaint_ids))
                preview = admin_client.get(
                    f"/admin/consolidated/{public_id}/preview"
                )
                self.assertEqual(preview.status_code, 200)
                self.assertIn(b"Multiple Reports - Fan Malfunction - Room 114", preview.data)
                self.assertIn(b"4 reports from 4 students", preview.data.lower())
                smtp.assert_not_called()
                self.post_form(
                    f"/admin/consolidated/{public_id}/edit",
                    {
                        "issue_title": "Fan Repair Needed",
                        "summary": "Four students report an inoperative fan in Room 114.",
                    },
                    client=admin_client,
                )
                with campuscare.get_db() as conn:
                    original_descriptions = [
                        row["complaint_text"] for row in conn.execute(
                            "SELECT complaint_text FROM complaints ORDER BY id"
                        ).fetchall()
                    ]
                self.assertEqual(original_descriptions, examples)
                edited_preview = admin_client.get(
                    f"/admin/consolidated/{public_id}/preview"
                )
                self.assertIn(b"Multiple Reports - Fan Repair Needed - Room 114",
                              edited_preview.data)
                self.assertIn(b"Four students report an inoperative fan",
                              edited_preview.data)

                approved = self.post_form(
                    f"/admin/consolidated/{public_id}/approve", client=admin_client
                )
                self.assertEqual(approved.status_code, 302)
                smtp.assert_called_once()
                message = smtp_server.send_message.call_args.args[0]
                self.assertEqual(message["To"], "maintenance@college.edu")
                self.assertEqual(
                    message["X-CampusCare-Batch-ID"], groups[0]["notification_id"]
                )
                plain_body = message.get_body(preferencelist=("plain",)).get_content()
                html_body = message.get_body(preferencelist=("html",)).get_content()
                self.assertIn("Total Reports: 4 from 4 students", plain_body)
                self.assertIn("4 reports from 4 students", html_body)
                self.assertIn("Issue: Fan Repair Needed", plain_body)
                self.assertIn("Four students report an inoperative fan", plain_body)
                for complaint_id in complaint_ids:
                    self.assertIn(f"Complaint #{complaint_id}", plain_body)
                again = self.post_form(
                    f"/admin/consolidated/{public_id}/approve", client=admin_client
                )
                self.assertEqual(again.status_code, 302)
                smtp.assert_called_once()
                with campuscare.get_db() as conn:
                    group = conn.execute("""
                        SELECT approval_status, email_status, send_attempts,
                               approval_admin_id, approved_at, sent_at
                        FROM complaint_groups WHERE public_id = ?
                    """, (public_id,)).fetchone()
                    attempt = conn.execute("""
                        SELECT recipient, attempt_number, smtp_accepted
                        FROM complaint_notification_attempts
                        WHERE notification_id = ?
                    """, (groups[0]["notification_id"],)).fetchone()
                self.assertEqual(group["approval_status"], "approved")
                self.assertEqual(group["email_status"], "sent")
                self.assertEqual(group["send_attempts"], 1)
                self.assertEqual(group["approval_admin_id"], campuscare.ADMIN_USERNAME)
                self.assertIsNotNone(group["approved_at"])
                self.assertIsNotNone(group["sent_at"])
                self.assertEqual(attempt["recipient"], "maintenance@college.edu")
                self.assertEqual(attempt["attempt_number"], 1)
                self.assertEqual(attempt["smtp_accepted"], 1)
                new_student = campuscare.app.test_client()
                self.create_student(
                    new_student, email="student5@example.edu",
                    enrollment="CSE202405", name="Student 5",
                )
                self.post_form(
                    "/submit",
                    {"complaint_text": "The fan in Room 114 is broken."},
                    client=new_student,
                )
                smtp.assert_called_once()
                with campuscare.get_db() as conn:
                    next_batch = conn.execute("""
                        SELECT g.id, g.related_group_id, g.email_status,
                               COUNT(gm.id) AS report_count
                        FROM complaint_groups AS g
                        JOIN complaint_group_members AS gm ON gm.group_id = g.id
                        GROUP BY g.id ORDER BY g.id DESC LIMIT 1
                    """).fetchone()
                self.assertEqual(next_batch["email_status"], "pending_approval")
                self.assertEqual(next_batch["report_count"], 1)
                self.assertEqual(next_batch["related_group_id"], groups[0]["id"])

    def test_student_cannot_approve_and_rejected_group_is_never_sent(self):
        self.create_student()
        self.submit("The ceiling fan is broken in Room 114.")
        with campuscare.get_db() as conn:
            public_id = conn.execute(
                "SELECT public_id FROM complaint_groups"
            ).fetchone()["public_id"]
        student_attempt = self.post_form(
            f"/admin/consolidated/{public_id}/approve"
        )
        self.assertEqual(student_attempt.status_code, 403)

        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        with patch.object(campuscare, "send_consolidated_email") as send_email:
            rejected = self.post_form(
                f"/admin/consolidated/{public_id}/reject",
                {"reason": "Not a shared maintenance issue"},
                client=admin_client,
            )
            self.assertEqual(rejected.status_code, 302)
            send_email.assert_not_called()
            self.post_form(
                f"/admin/consolidated/{public_id}/approve", client=admin_client
            )
            send_email.assert_not_called()
        with campuscare.get_db() as conn:
            group = conn.execute("""
                SELECT approval_status, email_status, rejection_reason,
                       rejected_by_admin_id, rejected_at
                FROM complaint_groups WHERE public_id = ?
            """, (public_id,)).fetchone()
        self.assertEqual(group["approval_status"], "rejected")
        self.assertEqual(group["email_status"], "rejected")
        self.assertEqual(group["rejection_reason"], "Not a shared maintenance issue")
        self.assertEqual(group["rejected_by_admin_id"], campuscare.ADMIN_USERNAME)
        self.assertIsNotNone(group["rejected_at"])

    def test_failed_approved_email_can_retry_same_batch_without_duplicate_after_send(self):
        self.create_student()
        self.submit("The classroom fan is not working in Room 114.")
        with campuscare.get_db() as conn:
            public_id = conn.execute(
                "SELECT public_id FROM complaint_groups"
            ).fetchone()["public_id"]
            notification_id = conn.execute(
                "SELECT notification_id FROM complaint_groups"
            ).fetchone()["notification_id"]
        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        with patch.dict(
            campuscare.DEPARTMENT_EMAILS,
            {"Infrastructure": "maintenance@college.edu"},
        ), patch.object(
            campuscare, "send_consolidated_email",
            side_effect=[
                (False, "SMTPConnectError: test failure"),
                (True, "SMTP accepted notification"),
            ],
        ) as send_email:
            self.post_form(
                f"/admin/consolidated/{public_id}/approve", client=admin_client
            )
            with campuscare.get_db() as conn:
                failed = conn.execute("""
                    SELECT approval_status, email_status, send_attempts,
                           approval_admin_id, approved_at
                    FROM complaint_groups WHERE public_id = ?
                """, (public_id,)).fetchone()
            self.assertEqual(failed["approval_status"], "approved")
            self.assertEqual(failed["email_status"], "failed")
            self.assertEqual(failed["send_attempts"], 1)
            self.assertEqual(failed["approval_admin_id"], campuscare.ADMIN_USERNAME)
            self.assertIsNotNone(failed["approved_at"])
            self.post_form(
                f"/admin/consolidated/{public_id}/retry", client=admin_client
            )
            self.assertEqual(send_email.call_count, 2)
            self.assertEqual(
                send_email.call_args_list[0].args[1],
                send_email.call_args_list[1].args[1],
            )
            with campuscare.get_db() as conn:
                sent = conn.execute("""
                    SELECT email_status, send_attempts, notification_id
                    FROM complaint_groups WHERE public_id = ?
                """, (public_id,)).fetchone()
            self.assertEqual(sent["email_status"], "sent")
            self.assertEqual(sent["send_attempts"], 2)
            self.assertEqual(sent["notification_id"], notification_id)
            with campuscare.get_db() as conn:
                attempts = conn.execute("""
                    SELECT attempt_number, smtp_result, smtp_accepted
                    FROM complaint_notification_attempts
                    WHERE notification_id = ? ORDER BY attempt_number
                """, (notification_id,)).fetchall()
            self.assertEqual(len(attempts), 2)
            self.assertEqual([attempt["smtp_accepted"] for attempt in attempts], [0, 1])
            self.post_form(
                f"/admin/consolidated/{public_id}/retry", client=admin_client
            )
            self.assertEqual(send_email.call_count, 2)

    def test_same_category_alone_does_not_group_different_issues_or_rooms(self):
        examples = [
            "The fan is not working in Room 114.",
            "The fan is not working in Room 115.",
            "The classroom light is not working in Room 114.",
        ]
        for index, text in enumerate(examples, 1):
            student = campuscare.app.test_client()
            self.create_student(
                student,
                email=f"location{index}@example.edu",
                enrollment=f"LOC20240{index}",
                name=f"Location Student {index}",
            )
            response = self.post_form(
                "/submit", {"complaint_text": text}, client=student
            )
            self.assertEqual(response.status_code, 200)
        with campuscare.get_db() as conn:
            groups = conn.execute("""
                SELECT g.location, COUNT(gm.id) AS report_count
                FROM complaint_groups AS g
                JOIN complaint_group_members AS gm ON gm.group_id = g.id
                GROUP BY g.id ORDER BY g.id
            """).fetchall()
        self.assertEqual(len(groups), 3)
        self.assertEqual([group["report_count"] for group in groups], [1, 1, 1])
        self.assertEqual([group["location"] for group in groups],
                         ["Room 114", "Room 115", "Room 114"])

    def test_temporary_department_address_previews_and_approves_in_demo_mode(self):
        self.create_student()
        self.submit("The classroom fan is broken in Room 114.")
        with campuscare.get_db() as conn:
            public_id = conn.execute(
                "SELECT public_id FROM complaint_groups"
            ).fetchone()["public_id"]
        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        with patch.object(campuscare, "send_consolidated_email") as send_email:
            preview = admin_client.get(
                f"/admin/consolidated/{public_id}/preview"
            )
            self.assertEqual(preview.status_code, 200)
            self.assertIn(b"maintenance@example.com", preview.data)
            self.assertIn("Demo Mode – Email Not Actually Sent".encode(), preview.data)
            self.assertIn(b"Approve (Demo Mode)", preview.data)
            response = self.post_form(
                f"/admin/consolidated/{public_id}/approve", client=admin_client
            )
            self.assertEqual(response.status_code, 302)
            send_email.assert_not_called()
            detail = admin_client.get(
                f"/admin/consolidated/{public_id}"
            )
            self.assertIn("Demo Mode – Email Not Actually Sent".encode(), detail.data)
        with campuscare.get_db() as conn:
            group = conn.execute("""
                SELECT approval_status, email_status, department_recipient,
                       approval_admin_id, approved_at, smtp_result, send_attempts
                FROM complaint_groups WHERE public_id = ?
            """, (public_id,)).fetchone()
        self.assertEqual(group["approval_status"], "approved")
        self.assertEqual(group["email_status"], "demo_mode")
        self.assertEqual(group["department_recipient"], "maintenance@example.com")
        self.assertEqual(group["approval_admin_id"], campuscare.ADMIN_USERNAME)
        self.assertIsNotNone(group["approved_at"])
        self.assertEqual(group["smtp_result"], "Demo Mode – Email Not Actually Sent")
        self.assertEqual(group["send_attempts"], 0)

    def test_smtp_refusal_is_not_reported_as_sent(self):
        smtp_server = MagicMock()
        smtp_server.send_message.return_value = {
            "maintenance@college.edu": (550, b"mailbox unavailable")
        }
        smtp_context = MagicMock()
        smtp_context.__enter__.return_value = smtp_server
        with patch.dict(os.environ, {
            "SMTP_HOST": "smtp.college.edu",
            "SMTP_FROM": "campuscare@college.edu",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "",
        }), patch.object(
            campuscare.smtplib, "SMTP", return_value=smtp_context
        ):
            sent, result = campuscare.send_consolidated_email(
                "maintenance@college.edu", "batch-1", "Subject", "Body", "<p>Body</p>"
            )
        self.assertFalse(sent)
        self.assertIn("SMTP refused recipient", result)

    def test_image_upload_is_saved_and_admin_only(self):
        self.create_student()
        response = self.post_form("/submit", data={
            "complaint_text": "The projector is not working in the classroom",
            "images": (io.BytesIO(b"\x89PNG\r\n\x1a\nfixture-data"), "classroom.png"),
        }, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"image(s) attached", response.data)
        with campuscare.get_db() as conn:
            attachment = conn.execute(
                "SELECT id, stored_name FROM complaint_attachments"
            ).fetchone()
        self.assertTrue(os.path.isfile(os.path.join(campuscare.UPLOAD_FOLDER, attachment["stored_name"])))
        owner_image = self.client.get(f"/admin/attachments/{attachment['id']}")
        self.assertEqual(owner_image.status_code, 200)
        owner_image.close()
        other_client = campuscare.app.test_client()
        self.create_student(other_client, "other@example.edu", "CSE2024002", "Other Student")
        self.assertEqual(
            other_client.get(f"/admin/attachments/{attachment['id']}").status_code, 404
        )
        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        image_response = admin_client.get(f"/admin/attachments/{attachment['id']}")
        self.assertEqual(image_response.status_code, 200)
        self.assertTrue(image_response.data.startswith(b"\x89PNG"))
        image_response.close()

    def test_invalid_image_is_rejected_without_creating_complaint(self):
        self.create_student()
        response = self.post_form("/submit", data={
            "complaint_text": "The projector is not working in the classroom",
            "images": (io.BytesIO(b"not an image"), "fake.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        with campuscare.get_db() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0], 0)

    def test_status_timeline_updates_and_feedback(self):
        self.create_student()
        with campuscare.get_db() as conn:
            student_id = conn.execute(
                "SELECT id FROM students WHERE college_email='student@example.edu'"
            ).fetchone()["id"]
        complaint_id = self.seed_complaint(student_id=student_id)
        self.assertEqual(
            self.client.get(f"/api/complaints/{complaint_id}/timeline").status_code, 200
        )
        admin_client = campuscare.app.test_client()
        self.login_admin(admin_client)
        updated = self.post_form(
            f"/admin/update/{complaint_id}", data={"status": "Resolved"},
            client=admin_client
        )
        self.assertEqual(updated.status_code, 302)
        timeline = self.client.get(
            f"/api/complaints/{complaint_id}/timeline"
        ).json
        self.assertEqual(timeline["current_status"], "Resolved")
        self.assertEqual(len(timeline["events"]), 2)
        feedback = self.post_form(
            f"/feedback/{complaint_id}", data={"rating": "5", "comment": "Helpful"}
        )
        self.assertEqual(feedback.status_code, 302)
        with campuscare.get_db() as conn:
            event = conn.execute(
                "SELECT event_type FROM complaint_events WHERE complaint_id=? ORDER BY id DESC",
                (complaint_id,),
            ).fetchone()
            rating = conn.execute(
                "SELECT feedback_rating FROM complaints WHERE id=?", (complaint_id,)
            ).fetchone()[0]
        self.assertEqual(event["event_type"], "rating")
        self.assertEqual(rating, 5)

    def test_old_open_complaint_is_escalated_once(self):
        now = datetime.now()
        old_date = (now - timedelta(hours=49)).strftime("%Y-%m-%d %H:%M:%S")
        complaint_id = self.seed_complaint(created_at=old_date)
        campuscare.escalate_overdue_complaints(now=now)
        campuscare.escalate_overdue_complaints(now=now)
        with campuscare.get_db() as conn:
            complaint = conn.execute(
                "SELECT priority, escalated_at FROM complaints WHERE id=?",
                (complaint_id,),
            ).fetchone()
            events = conn.execute("""
                SELECT COUNT(*) FROM complaint_events
                WHERE complaint_id=? AND event_type='escalation'
            """, (complaint_id,)).fetchone()[0]
        self.assertEqual(complaint["priority"], "High")
        self.assertIsNotNone(complaint["escalated_at"])
        self.assertEqual(events, 1)

    def test_legacy_database_migration_preserves_existing_row(self):
        legacy_path = os.path.join(self.temp_dir.name, "legacy.db")
        connection = sqlite3.connect(legacy_path)
        connection.execute("""
            CREATE TABLE complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                student_email TEXT,
                department TEXT,
                complaint_text TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            INSERT INTO complaints
            (student_name, complaint_text, category, priority, status, created_at)
            VALUES ('Legacy Student', 'Legacy complaint text', 'Other', 'Low', 'Pending', '2025-01-01 10:00:00')
        """)
        connection.commit()
        connection.close()
        current_path = campuscare.DB_PATH
        campuscare.DB_PATH = legacy_path
        try:
            campuscare.init_db()
        finally:
            campuscare.DB_PATH = current_path
        migrated = sqlite3.connect(legacy_path)
        try:
            row = migrated.execute(
                "SELECT student_name, complaint_text, status FROM complaints WHERE id=1"
            ).fetchone()
            student_id = migrated.execute(
                "SELECT student_id FROM complaints WHERE id=1"
            ).fetchone()[0]
            columns = {record[1] for record in migrated.execute("PRAGMA table_info(complaints)")}
            event_count = migrated.execute(
                "SELECT COUNT(*) FROM complaint_events WHERE complaint_id=1"
            ).fetchone()[0]
            foreign_key_errors = migrated.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            migrated.close()
        self.assertEqual(row, ("Legacy Student", "Legacy complaint text", "Pending"))
        self.assertIn("duplicate_of", columns)
        self.assertIn("escalated_at", columns)
        self.assertIn("student_id", columns)
        self.assertIsNone(student_id)
        self.assertEqual(event_count, 1)
        self.assertEqual(foreign_key_errors, [])
        backups = [
            filename for filename in os.listdir(os.path.dirname(legacy_path))
            if filename.startswith("legacy.db.backup.")
        ]
        self.assertEqual(len(backups), 1)
        campuscare.DB_PATH = legacy_path
        try:
            campuscare.init_db()
        finally:
            campuscare.DB_PATH = current_path
        self.assertEqual(len([
            filename for filename in os.listdir(os.path.dirname(legacy_path))
            if filename.startswith("legacy.db.backup.")
        ]), 1)


if __name__ == "__main__":
    unittest.main()
