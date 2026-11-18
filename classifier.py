"""Educational complaint classifier using TF-IDF and Logistic Regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = [
    "Academics",
    "Infrastructure",
    "IT / Technical",
    "Sanitation",
    "Canteen",
    "Hostel",
    "Transportation",
    "Administration",
    "Other",
]
MIN_CATEGORY_CONFIDENCE = 0.16

TRAINING_DATA = [
    ("internal marks are incorrect on my result", "Academics"),
    ("my exam marks were added incorrectly", "Academics"),
    ("the grade for my assignment is missing", "Academics"),
    ("the examination timetable has a scheduling error", "Academics"),
    ("our lecturer has not completed the course syllabus", "Academics"),
    ("the online learning portal has no lecture notes", "Academics"),
    ("my attendance record is wrong for last week", "Academics"),
    ("the final exam result has not been published", "Academics"),
    ("the professor cancelled class without notice", "Academics"),
    ("please review the answer key for the midterm", "Academics"),
    ("my coursework submission is not showing as received", "Academics"),
    ("there is no instructor assigned to our course", "Academics"),
    ("the class schedule overlaps with another lecture", "Academics"),
    ("I need a correction to my semester marksheet", "Academics"),
    ("the lab practical exam was not held as scheduled", "Academics"),
    ("my internal assessment grade is lower than recorded", "Academics"),
    ("the course material is missing from the class page", "Academics"),
    ("the examination room list has the wrong student number", "Academics"),
    ("we have not received feedback on our submitted project", "Academics"),
    ("a required elective is not available for registration", "Academics"),
    ("the exam grade has not been posted in the course results", "Academics"),
    ("my semester exam grade is still missing", "Academics"),

    ("classroom ceiling fans are not working", "Infrastructure"),
    ("several lights in the lecture hall have stopped working", "Infrastructure"),
    ("broken desks make the classroom difficult to use", "Infrastructure"),
    ("rainwater is leaking through the classroom roof", "Infrastructure"),
    ("the elevator in the academic block is out of service", "Infrastructure"),
    ("a window pane is broken in the science building", "Infrastructure"),
    ("the classroom door does not close properly", "Infrastructure"),
    ("the stairs handrail is loose in block B", "Infrastructure"),
    ("the lecture hall air conditioning has failed", "Infrastructure"),
    ("cracks have appeared in the corridor wall", "Infrastructure"),
    ("the library study tables are damaged", "Infrastructure"),
    ("the ceiling in the seminar room is sagging", "Infrastructure"),
    ("the campus pathway has a large unsafe pothole", "Infrastructure"),
    ("the electrical sockets in the classroom are damaged", "Infrastructure"),
    ("the lecture room has no working lights", "Infrastructure"),
    ("the main gate is broken and will not open", "Infrastructure"),
    ("the auditorium seats are loose and damaged", "Infrastructure"),
    ("a ceiling panel fell in the computer building", "Infrastructure"),
    ("the ramp outside the lecture block is damaged", "Infrastructure"),
    ("the drinking fountain fixture is physically broken", "Infrastructure"),

    ("college Wi-Fi is not working in the library", "IT / Technical"),
    ("the campus internet disconnects every few minutes", "IT / Technical"),
    ("I cannot sign in to the student portal", "IT / Technical"),
    ("the college website returns an error when I log in", "IT / Technical"),
    ("computers in the IT lab will not start", "IT / Technical"),
    ("the learning management system is unavailable", "IT / Technical"),
    ("my college email account cannot send messages", "IT / Technical"),
    ("the classroom projector will not connect to a laptop", "IT / Technical"),
    ("the online fee portal is stuck during payment", "IT / Technical"),
    ("the campus network password is not accepted", "IT / Technical"),
    ("the printer in the computer lab is offline", "IT / Technical"),
    ("the student app crashes whenever I open it", "IT / Technical"),
    ("the lecture recording will not play on the portal", "IT / Technical"),
    ("the biometric attendance scanner is not responding", "IT / Technical"),
    ("the campus computer has a broken keyboard", "IT / Technical"),
    ("the online course registration page keeps timing out", "IT / Technical"),
    ("the smart board is not displaying the lesson", "IT / Technical"),
    ("the library computer cannot access the internet", "IT / Technical"),
    ("the campus website is showing outdated information", "IT / Technical"),
    ("the help desk has not fixed my account access issue", "IT / Technical"),

    ("the washrooms are not cleaned regularly", "Sanitation"),
    ("the toilets smell bad and need cleaning", "Sanitation"),
    ("there is no soap in the restroom dispensers", "Sanitation"),
    ("garbage bins in the corridor have not been emptied", "Sanitation"),
    ("the washroom floor is dirty and slippery", "Sanitation"),
    ("the restroom taps are leaking and wasting water", "Sanitation"),
    ("the campus washroom has no running water", "Sanitation"),
    ("flies are gathering around the waste bins", "Sanitation"),
    ("the toilets are blocked in the main building", "Sanitation"),
    ("cleaning staff have not visited the common area", "Sanitation"),
    ("the handwashing area has no paper towels", "Sanitation"),
    ("the washroom sanitary bins are overflowing", "Sanitation"),
    ("dust and litter have collected in the hallway", "Sanitation"),
    ("the drinking water station needs hygienic cleaning", "Sanitation"),
    ("the restroom mirrors and sinks are covered in grime", "Sanitation"),
    ("there is a persistent sewage smell near the washroom", "Sanitation"),
    ("the classroom trash has not been collected", "Sanitation"),
    ("the bathroom cleaning schedule is not being followed", "Sanitation"),
    ("soap dispensers are empty in the student toilets", "Sanitation"),
    ("the campus courtyard is full of uncollected litter", "Sanitation"),

    ("the canteen food quality is poor", "Canteen"),
    ("the lunch served in the cafeteria was cold", "Canteen"),
    ("food at the college canteen tastes stale", "Canteen"),
    ("the canteen meal prices are too high", "Canteen"),
    ("I found an insect in my cafeteria meal", "Canteen"),
    ("the canteen queue takes too long at lunchtime", "Canteen"),
    ("the vegetarian options at the canteen are limited", "Canteen"),
    ("the canteen staff served an undercooked meal", "Canteen"),
    ("the cafeteria tables are dirty after meals", "Canteen"),
    ("the canteen runs out of food before the lunch break", "Canteen"),
    ("the food portion at the cafeteria is too small", "Canteen"),
    ("the canteen drinking cups are not clean", "Canteen"),
    ("the college cafeteria does not list food allergens", "Canteen"),
    ("the canteen menu has no affordable breakfast", "Canteen"),
    ("the food counter is closed during the posted hours", "Canteen"),
    ("the cafeteria meal gave several students stomach trouble", "Canteen"),
    ("the canteen cashier charged me twice for lunch", "Canteen"),
    ("the food served in the dining hall is often burnt", "Canteen"),
    ("the cafeteria serving line is poorly managed", "Canteen"),
    ("the canteen food containers are not covered", "Canteen"),

    ("the hostel water supply is unavailable", "Hostel"),
    ("my hostel room fan has stopped working", "Hostel"),
    ("the hostel warden has not responded to my request", "Hostel"),
    ("there is no hot water in the hostel shower", "Hostel"),
    ("the hostel room key does not open the door", "Hostel"),
    ("the hostel laundry machines are out of service", "Hostel"),
    ("the dormitory has a power outage every evening", "Hostel"),
    ("the hostel study room is locked after hours", "Hostel"),
    ("the hostel common room furniture is damaged", "Hostel"),
    ("the hostel room window will not lock", "Hostel"),
    ("the hostel internet is unavailable on my floor", "Hostel"),
    ("my room assignment in the hostel is incorrect", "Hostel"),
    ("the hostel curfew notice was not communicated", "Hostel"),
    ("there is a leak in the hostel bedroom ceiling", "Hostel"),
    ("the hostel water heater is not working", "Hostel"),
    ("the dormitory entrance security light is broken", "Hostel"),
    ("the hostel maintenance request is still pending", "Hostel"),
    ("my hostel room has a damaged wardrobe", "Hostel"),
    ("the hostel common area is too noisy at night", "Hostel"),
    ("the hostel gate is not being secured overnight", "Hostel"),
    ("the dorm laundry machine is broken and needs repair", "Hostel"),
    ("the hostel laundry service is unavailable this week", "Hostel"),

    ("the college bus is frequently late", "Transportation"),
    ("the morning campus shuttle did not arrive", "Transportation"),
    ("the bus route does not stop near our hostel", "Transportation"),
    ("the college bus is overcrowded every day", "Transportation"),
    ("the driver skipped the scheduled campus stop", "Transportation"),
    ("the shuttle schedule conflicts with evening classes", "Transportation"),
    ("the college bus brakes make a worrying noise", "Transportation"),
    ("my transport fee was charged twice", "Transportation"),
    ("there is no bus service after the late lecture", "Transportation"),
    ("the campus bus left before the posted departure time", "Transportation"),
    ("the bus stop has no shelter during rain", "Transportation"),
    ("the college shuttle is not wheelchair accessible", "Transportation"),
    ("the driver did not follow the assigned route", "Transportation"),
    ("the transport office has not issued my bus pass", "Transportation"),
    ("the campus bus has broken air conditioning", "Transportation"),
    ("the evening shuttle is consistently delayed", "Transportation"),
    ("the bus timetable is missing from the stop", "Transportation"),
    ("students are standing because the bus has no seats left", "Transportation"),
    ("the campus parking shuttle is not running today", "Transportation"),
    ("the bus conductor refused to accept my valid pass", "Transportation"),
    ("my college bus pass has not been issued", "Transportation"),
    ("the campus bus pass application is still pending", "Transportation"),

    ("my college ID card has not been issued", "Administration"),
    ("the student office has not processed my enrollment form", "Administration"),
    ("my tuition payment receipt is missing from the account", "Administration"),
    ("the administration office is not answering email", "Administration"),
    ("my student name is misspelled in the official record", "Administration"),
    ("I have not received the requested enrollment certificate", "Administration"),
    ("the scholarship application is pending with the office", "Administration"),
    ("the college has not approved my fee refund", "Administration"),
    ("the registrar has not updated my contact information", "Administration"),
    ("my transcript request has exceeded the stated wait time", "Administration"),
    ("the office gave me the wrong graduation letter", "Administration"),
    ("I cannot register because my student record is incomplete", "Administration"),
    ("the administration portal lists an incorrect fee balance", "Administration"),
    ("my library card application has not been processed", "Administration"),
    ("the office has not confirmed receipt of my transfer documents", "Administration"),
    ("I need an official letter confirming my student status", "Administration"),
    ("the student affairs office lost my submitted form", "Administration"),
    ("my campus access pass has not been renewed", "Administration"),
    ("the college has not corrected my admission record", "Administration"),
    ("I was charged an administrative fee that was not explained", "Administration"),

    ("I lost my wallet near the campus sports field", "Other"),
    ("someone found a set of keys beside the auditorium", "Other"),
    ("please report a stray dog near the campus entrance", "Other"),
    ("the music from a campus event is too loud", "Other"),
    ("I would like to suggest adding more student clubs", "Other"),
    ("a bicycle was stolen from the outdoor rack", "Other"),
    ("the campus garden has been damaged by visitors", "Other"),
    ("there is an unsafe crowd near the festival stage", "Other"),
    ("I need information about joining the photography club", "Other"),
    ("the campus sports field is closed without explanation", "Other"),
    ("a personal package was delivered to the wrong building", "Other"),
    ("students are being disturbed by loud construction outside campus", "Other"),
    ("I want to report lost property from the student lounge", "Other"),
    ("the campus event registration list has my name missing", "Other"),
    ("there is a stray animal inside the parking area", "Other"),
    ("I have a suggestion for the annual cultural festival", "Other"),
    ("someone left a bag unattended in the common area", "Other"),
    ("the bicycle stand is being used to store unrelated equipment", "Other"),
    ("I need directions to the alumni office", "Other"),
    ("the campus chess club needs a larger meeting space", "Other"),
]

MODEL = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), lowercase=True, strip_accents="unicode"
    )),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
MODEL.fit([text for text, _ in TRAINING_DATA], [label for _, label in TRAINING_DATA])

HIGH_PRIORITY_TERMS = [
    "fire", "smoke", "electric shock", "exposed wire", "threat", "violence",
    "ragging", "harassment", "unsafe", "accident", "dangerous", "emergency",
    "insects in the food", "contaminated", "no drinking water",
]
MEDIUM_PRIORITY_TERMS = [
    "not working", "broken", "leaking", "dirty", "slow", "late",
    "missing", "pending", "poor quality", "no water", "out of order",
]


def classify_complaint(text):
    cleaned = text.strip()
    probabilities = MODEL.predict_proba([cleaned])[0]
    best_index = probabilities.argmax()
    category = MODEL.classes_[best_index]
    confidence = float(probabilities[best_index])
    if confidence < MIN_CATEGORY_CONFIDENCE:
        category = "Other"
        other_index = list(MODEL.classes_).index("Other")
        confidence = float(probabilities[other_index])
    lower = cleaned.lower()
    if any(term in lower for term in HIGH_PRIORITY_TERMS):
        priority = "High"
    elif any(term in lower for term in MEDIUM_PRIORITY_TERMS):
        priority = "Medium"
    else:
        priority = "Low"
    return category, priority, confidence
