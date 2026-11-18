# CampusCare Advanced — College Complaint Classifier

A Flask + SQLite mini project with student accounts, private complaint tracking, text classification, a separate admin login, analytics charts, search/filters, CSV export, and student feedback.

## Run on Windows CMD
1. Extract the ZIP file.
2. Open CMD in the `CampusCare_Advanced` folder (the folder containing `app.py`).
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Start the app:
   `python app.py`
5. Open `http://127.0.0.1:5000`

## Demo admin login
- Username: `admin`
- Password: `admin123`

Change these demo credentials and configure a stable secret key before deployment:
```cmd
set ADMIN_USERNAME=youradmin
set ADMIN_PASSWORD=use-a-strong-password
set SECRET_KEY=replace-with-a-long-random-secret
set SESSION_COOKIE_SECURE=true
python app.py
```
`SESSION_COOKIE_SECURE=true` requires HTTPS. Keep `SECRET_KEY` stable and private; without it, CampusCare generates a random key at startup and existing sessions will be invalidated at restart.

## Features
- Student registration and sign-in with Werkzeug password hashes
- Required student authentication for complaint submission, tracking, feedback, profiles, and private attachments
- Student dashboard with complaint counts and private recent history
- Separate student and administrator roles
- TF-IDF + Logistic Regression category prediction
- Nine complaint categories: Academics, Infrastructure, IT / Technical, Sanitation, Canteen, Hostel, Transportation, Administration, and Other
- Confidence-based fallback to Other for low-confidence predictions
- Rule-based priority estimate
- SQLite storage and complaint ID tracking
- Admin login and logout
- Dashboard charts (Chart.js loaded from CDN, internet required for charts)
- Search, category, priority, status and date filters
- Complaint status updates
- CSV export for reports
- Student feedback (1–5 rating and comment) after resolution, restricted to the complaint owner
- Consolidated complaint batches with explicit administrator approval before any department email
- Migration helper for the earlier version's SQLite database

## CampusCare feature additions

- **Campus assistant:** A local FAQ/status assistant that answers in English or Hindi and uses the existing classifier for complaint-category guidance. It does not send student messages to an external AI service.
- **Duplicate grouping and approval:** Complaint text is normalized and compared with character n-gram TF-IDF plus issue-keyword overlap. A candidate must match the predicted category and extracted location; category alone never groups complaints. The default automatic-match threshold is `0.72` and can be changed with `CAMPUSCARE_DUPLICATE_SIMILARITY_THRESHOLD`. Uncertain matches and reports without a location stay in separate groups for admin review. Every original complaint remains individually stored and trackable.
- **Consolidated notifications:** The admin dashboard shows pending groups, original reports, similarity information, analytics, a full HTML/plain-text email preview, editable group title/summary, and Reject or Approve & Send actions. Approval is enforced on authenticated admin-only backend routes. A successful send is recorded once per unique batch; SMTP failures can be retried on the same batch.
- **Complaint images:** Up to three JPG, PNG, or WebP images can be attached (5 MB each). Files are stored in `uploads/`, linked in SQLite, and served only to authenticated admins.
- **Live status timeline:** Submission, status changes, attachments, feedback, and escalations are recorded as complaint events. The tracking page refreshes its timeline every 20 seconds.
- **Automatic escalation:** Pending complaints are escalated after 48 hours and in-progress complaints after 72 hours by raising priority to High and adding a timeline event. These thresholds can be changed with `CAMPUSCARE_ESCALATE_PENDING_HOURS` and `CAMPUSCARE_ESCALATE_PROGRESS_HOURS`. Web requests check for overdue complaints at most once per minute; for quiet sites, schedule the command below.
- **English/Hindi:** Use the language switch in the navigation. The UI, assistant, and common Hindi complaint phrases are supported locally. English complaints continue through the existing TF-IDF/Logistic Regression classifier; Hindi classification uses a small local phrase map and then the same classifier, so uncommon Hindi wording may be categorized less accurately.
- **Student satisfaction:** The existing 1–5 rating and optional comment after resolution are retained and added to the live event timeline and dashboard averages.

To run escalation checks on a schedule even when the site is idle, configure Windows Task Scheduler to run this command from the project directory (for example, every 15 minutes):

```cmd
python -m flask --app app escalate-complaints
```

Run the feature and student-auth regression tests with an isolated temporary database:

```cmd
python -m unittest discover -s tests -v
```

Student accounts and new complaint/group fields are created with idempotent additive migrations. Before the current schema migration, the app creates a timestamped SQLite backup next to `complaints.db`. Existing complaint rows and IDs are preserved; historical complaints without a student account remain available to administrators. Include uploaded image files with `complaints.db` in independent production backups.

Student passwords are stored as salted Werkzeug hashes. CSRF tokens are required for state-changing form and assistant requests. Login attempts are limited to five failures per email/IP combination within a 15-minute window. Use HTTPS and set `SESSION_COOKIE_SECURE=true` outside local development.

## Notes and limitations
- The classifier retrains its TF-IDF + Logistic Regression pipeline when `classifier.py` is imported, using 186 built-in labeled examples (20 per category, with 22 each for Academics, Hostel, and Transportation). Predictions below 0.16 maximum class probability are labeled `Other`; the returned confidence remains the highest model probability. Its 45-example curated holdout is useful as a regression check, not a substitute for evaluation on representative, independently labeled campus complaints. Model confidence is approximate and is not a guarantee of correctness.
- Historical category names such as `IT & Wi-Fi`, `Water & Sanitation`, `Canteen & Food`, and `Transport` are displayed and filtered as their canonical category equivalents. Existing complaint records are not rewritten.
- Priority is estimated by simple keyword rules and must not replace direct reporting of emergencies or safety issues.
- The default admin credentials are for local demonstration only. Set strong `ADMIN_USERNAME` and `ADMIN_PASSWORD` values before deployment.
- Department email delivery is never automatic. A student submission only creates or updates a pending group; a logged-in administrator must preview and explicitly approve it. The temporary `@example.com` department addresses are displayed in the preview but approvals remain in Demo Mode: no delivery is attempted and the batch is not marked Sent. Replace the centralized `DEPARTMENT_EMAILS` mapping in `app.py` with verified department addresses to enable delivery.

## Configure department email notifications

The temporary category-to-address mapping lives in `DEPARTMENT_EMAILS` in `app.py`. It currently contains `academics@example.com`, `maintenance@example.com`, `it@example.com`, `housekeeping@example.com`, `canteen@example.com`, `hostel@example.com`, `transport@example.com`, `administration@example.com`, and `support@example.com`. Since `example.com` is a reserved example domain, every notification is explicitly labeled **Demo Mode – Email Not Actually Sent** and SMTP delivery is disabled for these addresses.

When official addresses are available, update the corresponding entries in this single mapping. Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `SMTP_FROM` in the deployment environment (for Gmail, use an App Password with 2-Step Verification enabled). Keep SMTP credentials private. The app records the batch ID, recipient, approving admin, timestamps, attempt count, and SMTP result. Emails are sent only when an authenticated administrator approves a notification. The configurable duplicate threshold defaults to `0.72`; for example, set `CAMPUSCARE_DUPLICATE_SIMILARITY_THRESHOLD=0.78` to require a closer text match.
- Collect only the student profile information needed to manage complaints. Complaint submission is no longer anonymous and requires a student account.
