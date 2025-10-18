import streamlit as st
import tempfile
import os
import cv2
import time
from anomaly_detection import process_video

# Page setup
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("AI-Powered Surveillance System")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

run_thresh = st.sidebar.slider("Running Threshold (pixels/sec)", 50, 600, 200, 10)
loiter_time = st.sidebar.slider("Loitering Time (sec)", 5, 60, 8, 1)
loiter_radius = st.sidebar.slider("Loitering Radius (pixels)", 20, 150, 50, 5)
window = st.sidebar.slider("Tracking Window (sec)", 5, 60, 25, 1)

input_mode = st.sidebar.radio("Input Source", ["Upload Video", "Webcam"])

# Handle input
input_video = None
if input_mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video (.avi, .mp4, .mov, .mkv)",
                                     type=["avi", "mp4", "mov", "mkv"])
    if uploaded_file is not None:
        # Always save as .mp4 so downstream code uses the right codec
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_file.read())
        input_video = temp_input.name
elif input_mode == "Webcam":
    input_video = 0

# Detection button
if st.button("▶️ Start Detection") and input_video is not None:
    st.write("Processing video... Please wait ⏳")

    # Use an mp4 output with browser-compatible codec
    output_path = os.path.join(
        tempfile.gettempdir(),
        f"anomaly_output_{int(time.time())}.mp4"
    )

    # Run detection
    process_video(
        source=input_video,
        output=output_path,
        model_name="yolov8m.pt",
        tracker_cfg="bytetrack.yaml",
        run_thresh_px_per_s=run_thresh,
        loiter_time_s=loiter_time,
        loiter_radius_px=loiter_radius,
        window_s=window,
        show_window=False,
    )

    st.success("✅ Processing complete!")

    # Ensure video is readable by browser
    if os.path.exists(output_path):
        try:
            # Re-encode to H.264 (safe for browser)
            fixed_path = os.path.splitext(output_path)[0] + "_fixed.mp4"
            os.system(
                f"ffmpeg -y -i {output_path} -vcodec libx264 -acodec aac {fixed_path}"
            )
            st.video(fixed_path)
        except Exception as e:
            st.error(f"⚠️ Could not convert video: {e}")
    else:
        st.error("❌ No output video found. Check process_video logic.")

else:
    st.info("Upload a video or select webcam, then click 'Start Detection'.")
