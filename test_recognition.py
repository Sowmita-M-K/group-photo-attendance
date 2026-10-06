import os
import cv2

from src.enrollment import load_embeddings
from src.recognition import process_group_photo


PHOTO_PATH = "data/uploads/group.jpg"
EMBEDDINGS_PATH = "data/output/embeddings.pkl"
OUTPUT_PATH = "data/output/test_result.jpg"


if not os.path.exists(PHOTO_PATH):
    raise FileNotFoundError(
        f"Group photo not found: {PHOTO_PATH}"
    )

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(
        f"Embeddings not found: {EMBEDDINGS_PATH}"
    )

known_encodings, known_names = load_embeddings(
    EMBEDDINGS_PATH
)

output_image, present_names, detected_names = (
    process_group_photo(
        PHOTO_PATH,
        known_encodings,
        known_names
    )
)

cv2.imwrite(
    OUTPUT_PATH,
    output_image
)

print("Detected names:", detected_names)
print(
    "Present students:",
    sorted(present_names)
)
print(
    f"Saved result to: {OUTPUT_PATH}"
)