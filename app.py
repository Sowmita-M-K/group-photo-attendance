from pathlib import Path
from datetime import date

import cv2
import pandas as pd
import streamlit as st

from src.enrollment import (
    create_embeddings,
    save_embeddings,
    load_embeddings
)

from src.recognition import (
    process_group_photo,
    create_attendance
)


KNOWN_FOLDER = Path("data/known")
UPLOAD_FOLDER = Path("data/uploads")
OUTPUT_FOLDER = Path("data/output")
EMBEDDINGS_FILE = OUTPUT_FOLDER / "embeddings.pkl"

KNOWN_FOLDER.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="Group Photo Attendance",
    page_icon="📸",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        color: #17365d;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        color: #5b677a;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e1e7ef;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .section-title {
        color: #17365d;
        font-size: 1.3rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.6rem;
    }

    .present-box {
        background-color: #eaf8ef;
        border-left: 5px solid #20a05a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #145c35;
    }

    .absent-box {
        background-color: #fff0f0;
        border-left: 5px solid #d64545;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #8d2020;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="dashboard-title">Group Photo Attendance</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    "Upload a group photo to identify students and generate attendance."
    "</div>",
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("System Controls")

    st.write(
        "Student folders are read from:"
    )
    st.code("data/known/StudentName/")

    st.divider()

    if st.button(
        "Rebuild Face Database",
        use_container_width=True
    ):
        try:
            encodings, names, skipped = (
                create_embeddings(str(KNOWN_FOLDER))
            )

            if not encodings:
                st.error(
                    "No valid faces found in the known-student folders."
                )
            else:
                save_embeddings(
                    encodings,
                    names,
                    str(EMBEDDINGS_FILE)
                )

                st.success(
                    f"Saved {len(encodings)} face encodings "
                    f"for {len(set(names))} students."
                )

                if skipped:
                    st.warning(
                        f"{len(skipped)} image(s) were skipped."
                    )

        except Exception as error:
            st.error(f"Database error: {error}")

    st.divider()

    st.caption(
        f"Attendance date: {date.today()}"
    )


st.markdown(
    '<div class="section-title">Upload Group Photo</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG group photo",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:
    st.info(
        "Upload a group photo to begin attendance processing."
    )
    st.stop()


upload_path = UPLOAD_FOLDER / uploaded_file.name

with open(upload_path, "wb") as file:
    file.write(uploaded_file.getbuffer())

st.success(
    f"Uploaded: {uploaded_file.name}"
)


if not EMBEDDINGS_FILE.exists():
    st.warning(
        "Face database not found. Click 'Rebuild Face Database' "
        "in the sidebar first."
    )
    st.stop()


if st.button(
    "Process Attendance",
    type="primary",
    use_container_width=True
):
    with st.spinner(
        "Detecting and matching faces..."
    ):
        known_encodings, known_names = load_embeddings(
            str(EMBEDDINGS_FILE)
        )

        output_image, present_names, detected_names = (
            process_group_photo(
                str(upload_path),
                known_encodings,
                known_names,
                tolerance=0.5
            )
        )

        marked_path = OUTPUT_FOLDER / "marked_group_photo.jpg"

        cv2.imwrite(
            str(marked_path),
            output_image
        )

        attendance_df = create_attendance(
            known_names,
            present_names
        )

        attendance_df = attendance_df[
            ["Name", "Status", "Date"]
        ]

        csv_path = OUTPUT_FOLDER / "attendance.csv"

        attendance_df.to_csv(
            csv_path,
            index=False
        )

        st.session_state["output_image"] = str(marked_path)
        st.session_state["attendance_df"] = attendance_df
        st.session_state["detected_names"] = detected_names
        st.session_state["present_names"] = sorted(
            present_names
        )

        st.success(
            "Attendance processed successfully."
        )


if "attendance_df" in st.session_state:
    attendance_df = st.session_state["attendance_df"]
    detected_names = st.session_state["detected_names"]
    present_names = st.session_state["present_names"]

    all_students = sorted(
        set(attendance_df["Name"].tolist())
    )

    absent_names = [
        student
        for student in all_students
        if student not in present_names
    ]

    st.markdown(
        '<div class="section-title">Attendance Overview</div>',
        unsafe_allow_html=True
    )

    total_students = len(all_students)
    present_count = len(present_names)
    absent_count = len(absent_names)
    detected_count = len(detected_names)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric(
            "Total Students",
            total_students
        )

    with metric_2:
        st.metric(
            "Present",
            present_count
        )

    with metric_3:
        st.metric(
            "Absent",
            absent_count
        )

    with metric_4:
        st.metric(
            "Faces Detected",
            detected_count
        )

    st.divider()

    image_column, summary_column = st.columns(
        [1.5, 1]
    )

    with image_column:
        st.markdown(
            '<div class="section-title">'
            "Processed Group Photo"
            "</div>",
            unsafe_allow_html=True
        )

        st.image(
            st.session_state["output_image"],
            use_container_width=True
        )

    with summary_column:
        st.markdown(
            '<div class="section-title">'
            "Present Students"
            "</div>",
            unsafe_allow_html=True
        )

        if present_names:
            present_text = "<br>".join(
                f"✅ {name}"
                for name in present_names
            )

            st.markdown(
                f'<div class="present-box">{present_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("No enrolled student was recognized.")

        st.markdown(
            '<div class="section-title">'
            "Absent Students"
            "</div>",
            unsafe_allow_html=True
        )

        if absent_names:
            absent_text = "<br>".join(
                f"❌ {name}"
                for name in absent_names
            )

            st.markdown(
                f'<div class="absent-box">{absent_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.success("No absences recorded.")

    st.markdown(
        '<div class="section-title">'
        "Attendance Table"
        "</div>",
        unsafe_allow_html=True
    )

    st.dataframe(
        attendance_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = attendance_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Attendance CSV",
        data=csv_data,
        file_name=f"attendance_{date.today()}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Green boxes indicate recognized students. "
        "Red boxes indicate unknown faces."
    )