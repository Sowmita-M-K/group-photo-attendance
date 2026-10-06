import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import date


def process_group_photo(
    photo_path,
    known_encodings,
    known_names,
    tolerance=0.5
):
    image = face_recognition.load_image_file(photo_path)

    face_locations = face_recognition.face_locations(image)

    face_encodings = face_recognition.face_encodings(
        image,
        face_locations
    )

    output_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    present_names = set()
    detected_names = []

    for location, face_encoding in zip(
        face_locations,
        face_encodings
    ):
        top, right, bottom, left = location

        name = "Unknown"

        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(
                known_encodings,
                face_encoding
            )

            best_index = int(np.argmin(distances))
            best_distance = distances[best_index]

            if best_distance <= tolerance:
                name = known_names[best_index]
                present_names.add(name)

        detected_names.append(name)

        if name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            output_image,
            (left, top),
            (right, bottom),
            color,
            2
        )

        cv2.putText(
            output_image,
            name,
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return output_image, present_names, detected_names


def create_attendance(
    student_names,
    present_names
):
    rows = []

    for student in sorted(set(student_names)):
        status = (
            "Present"
            if student in present_names
            else "Absent"
        )

        rows.append(
            {
                "Date": str(date.today()),
                "Name": student,
                "Status": status
            }
        )

    return pd.DataFrame(rows)