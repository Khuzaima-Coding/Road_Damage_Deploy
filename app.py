
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

st.title("Road Damage Detection Model")


# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error("Could not load the YOLO model. Check that best.pt exists.")
    st.exception(e)
    st.stop()


# ---------------- SESSION STATE ----------------

if "processed_video" not in st.session_state:
    st.session_state.processed_video = None


# ---------------- SELECT INPUT TYPE ----------------

input_type = st.radio(
    "Choose input type",
    ["Image", "Video"]
)


# ==================================================
#                 IMAGE DETECTION
# ==================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

            if st.button("Detect Road Damage", key="detect_image"):

                with st.spinner("Detecting road damage..."):

                    results = model.predict(
                        source=image,
                        device="cpu",
                        verbose=False
                    )

                    # YOLO returns the plotted image in BGR format
                    result_image = results[0].plot()

                st.image(
                    result_image,
                    channels="BGR",
                    caption="Detection Result",
                    use_container_width=True
                )

        except Exception as e:
            st.error("Image detection failed.")
            st.exception(e)


# ==================================================
#                 VIDEO DETECTION
# ==================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "Detect Road Damage in Video",
            key="detect_video"
        ):

            # Remove the previous result before processing a new one
            st.session_state.processed_video = None

            input_path = None
            output_path = None
            h264_path = None
            cap = None
            writer = None

            try:

                # ---------- SAVE INPUT VIDEO ----------

                video_bytes = uploaded_video.getvalue()

                suffix = (
                    os.path.splitext(uploaded_video.name)[1]
                    or ".mp4"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp:

                    temp.write(video_bytes)
                    input_path = temp.name

                # ---------- OPEN INPUT VIDEO ----------

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    raise RuntimeError(
                        "OpenCV could not open the uploaded video."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )
                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )
                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if width <= 0 or height <= 0:
                    raise ValueError(
                        "The uploaded video has invalid dimensions."
                    )

                # Use a fallback frame rate if metadata is missing
                if fps <= 0:
                    fps = 30.0

                # ---------- CREATE OUTPUT VIDEO ----------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp:

                    output_path = temp.name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        "OpenCV could not create the output video."
                    )

                # ---------- PROCESS VIDEO FRAMES ----------

                progress = st.progress(0)
                status = st.empty()

                frame_count = 0

                while True:

                    success, frame = cap.read()

                    if not success:
                        break

                    # Run YOLO detection on this frame
                    results = model.predict(
                        source=frame,
                        device="cpu",
                        verbose=False
                    )

                    # Draw bounding boxes and labels
                    annotated_frame = results[0].plot()

                    # Write annotated frame to output video
                    writer.write(annotated_frame)

                    frame_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                    status.write(
                        f"Processed {frame_count} frames"
                    )

                # Release resources before converting the video
                cap.release()
                cap = None

                writer.release()
                writer = None

                if frame_count == 0:
                    raise RuntimeError(
                        "No video frames could be processed."
                    )

                if (
                    not os.path.exists(output_path)
                    or os.path.getsize(output_path) == 0
                ):
                    raise RuntimeError(
                        "The processed video is missing or empty."
                    )

                # ---------- CONVERT TO H.264 MP4 ----------

                h264_path = (
                    os.path.splitext(output_path)[0]
                    + "_h264.mp4"
                )

                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

                ffmpeg_result = subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i", output_path,
                        "-c:v", "libx264",
                        "-preset", "ultrafast",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",
                        h264_path
                    ],
                    capture_output=True,
                    text=True
                )

                if ffmpeg_result.returncode != 0:
                    raise RuntimeError(
                        "FFmpeg conversion failed:\n"
                        + ffmpeg_result.stderr[-4000:]
                    )

                if (
                    not os.path.exists(h264_path)
                    or os.path.getsize(h264_path) == 0
                ):
                    raise RuntimeError(
                        "The converted video is missing or empty."
                    )

                # ---------- SAVE RESULT IN SESSION STATE ----------

                with open(h264_path, "rb") as file:
                    st.session_state.processed_video = file.read()

                progress.progress(1.0)
                status.success(
                    f"Detection completed! "
                    f"Processed {frame_count} frames."
                )

                st.success("Road damage detection completed!")

            except Exception as e:

                st.error("Video processing failed.")
                st.exception(e)

            finally:

                # Release resources if an error occurs
                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                # Delete temporary files
                for path in [
                    input_path,
                    output_path,
                    h264_path
                ]:

                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

    # ---------- DISPLAY PROCESSED VIDEO ----------

    if st.session_state.processed_video is not None:

        st.subheader("Detected Video")

        st.video(
            st.session_state.processed_video,
            format="video/mp4"
        )

        st.download_button(
            label="Download Detected Video",
            data=st.session_state.processed_video,
            file_name="road_damage_result.mp4",
            mime="video/mp4",
            key="download_video"
        )
