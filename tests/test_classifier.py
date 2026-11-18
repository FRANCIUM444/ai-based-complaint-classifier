import unittest
from collections import Counter

import classifier


HELD_OUT_COMPLAINTS = {
    "Academics": [
        "My internal marks are incorrect on the transcript.",
        "The exam grade has not been posted.",
        "Our lecture was cancelled without notice.",
        "The midterm answer key contains a mistake.",
        "My attendance percentage is wrong.",
    ],
    "Infrastructure": [
        "The classroom ceiling fans are not working.",
        "A broken desk is in the lecture room.",
        "Rain is leaking through the auditorium roof.",
        "The elevator in the science building is out of service.",
        "Several hallway lights are broken.",
    ],
    "IT / Technical": [
        "College Wi-Fi is not working.",
        "The student portal keeps rejecting my login.",
        "The computer lab printer is offline.",
        "The lecture projector cannot connect to my laptop.",
        "My campus email account will not send messages.",
    ],
    "Sanitation": [
        "The washrooms are not cleaned.",
        "The toilets smell and need cleaning.",
        "The restroom has no soap.",
        "Garbage bins have not been emptied.",
        "The bathroom floor is dirty.",
    ],
    "Canteen": [
        "The food quality at the canteen is poor.",
        "The cafeteria meal was served cold.",
        "I found an insect in my lunch.",
        "Canteen prices are too high.",
        "The food counter ran out of meals.",
    ],
    "Hostel": [
        "The hostel water supply is unavailable.",
        "There is no hot water in my dorm shower.",
        "The hostel warden has not replied.",
        "My hostel room key does not work.",
        "The dorm laundry machine is broken.",
    ],
    "Transportation": [
        "The college bus is frequently late.",
        "The campus shuttle missed its stop.",
        "The bus is overcrowded every morning.",
        "The evening campus bus did not arrive.",
        "My bus pass has not been issued.",
    ],
    "Administration": [
        "My college ID card has not been issued.",
        "The registrar has not corrected my student record.",
        "I have not received my enrollment certificate.",
        "My scholarship paperwork is pending at the office.",
        "The administration has not sent my fee receipt.",
    ],
    "Other": [
        "I lost my wallet by the sports field.",
        "A stray dog is near the campus entrance.",
        "The campus event music is too loud.",
        "I want to report lost keys near the auditorium.",
        "I have a suggestion for the cultural festival.",
    ],
}


class ComplaintClassifierTests(unittest.TestCase):
    def test_training_data_and_model_cover_all_nine_categories(self):
        counts = Counter(label for _, label in classifier.TRAINING_DATA)
        self.assertEqual(set(counts), set(classifier.CATEGORIES))
        self.assertEqual(set(classifier.MODEL.classes_), set(classifier.CATEGORIES))
        self.assertTrue(all(count >= 20 for count in counts.values()), counts)

    def test_predicts_five_held_out_examples_per_category(self):
        self.assertEqual(set(HELD_OUT_COMPLAINTS), set(classifier.CATEGORIES))
        mistakes = []
        for expected, complaints in HELD_OUT_COMPLAINTS.items():
            self.assertEqual(len(complaints), 5, expected)
            for text in complaints:
                predicted, _, confidence = classifier.classify_complaint(text)
                if predicted != expected:
                    mistakes.append((expected, predicted, confidence, text))
        self.assertEqual(mistakes, [])

    def test_low_confidence_unrelated_inputs_use_other_fallback(self):
        unrelated = [
            "The moon looks beautiful tonight.",
            "I am planning a trip during winter break.",
            "asdj qwer 9382",
        ]
        for text in unrelated:
            category, _, confidence = classifier.classify_complaint(text)
            self.assertLess(confidence, classifier.MIN_CATEGORY_CONFIDENCE, text)
            self.assertEqual(category, "Other", text)


if __name__ == "__main__":
    unittest.main()
