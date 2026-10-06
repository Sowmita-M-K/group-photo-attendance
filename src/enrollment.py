import os
import pickle
import face_recognition

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def create_embeddings(known_folder="data/known"):
    encodings = []
    names = []
    skipped = []

    if not os.path.exists(known_folder):
        raise FileNotFoundError(
            f"Folder not found: {known_folder}"
        )

    for person_name in os.listdir(known_folder):
        person_folder = os.path.join(
            known_folder,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(
                person_folder,
                filename
            )

            try:
                image = face_recognition.load_image_file(
                    image_path
                )

                face_encodings = (
                    face_recognition.face_encodings(image)
                )

                if len(face_encodings) == 1:
                    encodings.append(face_encodings[0])
                    names.append(person_name)
                else:
                    skipped.append(
                        f"{image_path} "
                        f"({len(face_encodings)} faces found)"
                    )

            except Exception as error:
                skipped.append(
                    f"{image_path} ({error})"
                )

    return encodings, names, skipped


def save_embeddings(
    encodings,
    names,
    output_file="data/output/embeddings.pkl"
):
    output_folder = os.path.dirname(output_file)

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    data = {
        "encodings": encodings,
        "names": names
    }

    with open(output_file, "wb") as file:
        pickle.dump(data, file)


def load_embeddings(
    input_file="data/output/embeddings.pkl"
):
    with open(input_file, "rb") as file:
        data = pickle.load(file)

    return data["encodings"], data["names"]


if __name__ == "__main__":
    encodings, names, skipped = create_embeddings()

    if not encodings:
        print("No valid face encodings were found.")
        print(
            "Add clear individual photos inside "
            "data/known/StudentName/"
        )
    else:
        save_embeddings(encodings, names)

        print(
            f"Created {len(encodings)} face encodings."
        )
        print(
            f"Students found: {sorted(set(names))}"
        )

    if skipped:
        print("\nSkipped files:")
        for item in skipped:
            print(f"- {item}")