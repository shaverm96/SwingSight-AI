import streamlit as st

from swingsight.club_detection import CLUB_CATEGORIES, CLUB_LABELS, detect_club_category
from swingsight.config import load_config
from swingsight.io_utils import save_uploaded_file
from swingsight.pipeline import analyze_swing


st.set_page_config(page_title="SwingSight AI", layout="wide")

config = load_config()

st.title("SwingSight AI: Club-Aware Golf Swing Coach")

st.header("1) Club Selection")
club_image = st.file_uploader("Upload a club image", type=["jpg", "jpeg", "png"])

predicted_category = None
predicted_confidence = None

if club_image is not None:
    predicted_category, predicted_confidence = detect_club_category(club_image, config)

if predicted_category is None:
    predicted_category = CLUB_CATEGORIES[0]

selected_category = st.selectbox(
    "Confirm club category",
    options=CLUB_CATEGORIES,
    index=CLUB_CATEGORIES.index(predicted_category),
    format_func=lambda c: CLUB_LABELS.get(c, c),
)

if predicted_confidence is not None:
    st.caption(f"Detected: {CLUB_LABELS.get(predicted_category)} (conf: {predicted_confidence:.2f})")

st.header("2) Swing Video")
video_file = st.file_uploader("Upload a swing video", type=["mp4", "mov", "avi"])

run_analysis = st.button("Run analysis", type="primary", disabled=video_file is None)

if run_analysis and video_file is not None:
    outputs_dir = config["paths"]["outputs_dir"]
    video_path = save_uploaded_file(video_file, outputs_dir, "swing_video")

    club_image_path = None
    if club_image is not None:
        club_image_path = save_uploaded_file(club_image, outputs_dir, "club_image")

    results = analyze_swing(
        club_category=selected_category,
        video_path=video_path,
        club_image_path=club_image_path,
        config=config,
    )

    st.header("Results")
    st.subheader("Detected Club")
    st.write(CLUB_LABELS.get(results["club_category"], results["club_category"]))

    st.subheader("Swing Scorecard")
    st.json(results["metrics"])

    st.subheader("Recommendations")
    for item in results["feedback"]:
        st.write(f"- {item}")

    if results.get("annotated_video_path"):
        st.subheader("Annotated Swing Video")
        st.video(results["annotated_video_path"])
