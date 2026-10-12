
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import subprocess
import imageio_ffmpeg

# YOLO model load
model = YOLO("best.pt")

st.title("Road Damage Detection Model")

# Select input type
input_type = st.radio("Choose input type", ["Image", "Video"])

# ---------------- IMAGE DETECTION ----------------

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Original Image")

        # YOLO prediction
        if st.button("Detect Road Damage"):

            results = model.predict(
                image
            )

            # Get image with bounding boxes
            result_image = results[0].plot()

            # Display result (YOLO plot is BGR)
            st.image(
                result_image,
                channels="BGR",
                caption="Detection Result"
            )


# ---------------- VIDEO DETECTION ----------------


else:
    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    # Save the result across Streamlit reruns
    if "processed_video" not in st.session_state:
        st.session_state.processed_video = None

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Detect Road Damage in Video", key="detect_video"):

            input_path = None
            output_path = None
            h264_path = None
            cap = None
            writer = None

            try:
                # Save the uploaded video temporarily
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp:
                    temp.write(uploaded_video.getvalue())
                    input_path = temp.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("Could not open the uploaded video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps <= 0 or width <= 0 or height <= 0:
                    st.error("Invalid video properties.")
                    st.stop()

                # Create temporary output video
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp:
                    output_path = temp.name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    st.error("Could not create the output video.")
                    st.stop()

                progress = st.progress(0)
                status = st.empty()
                frame_count = 0

                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    results = model.predict(
                        frame,
                        device="cpu",
                        verbose=False
                    )

                    # Draw detections on the frame
                    annotated_frame = results[0].plot()
                    writer.write(annotated_frame)

                    frame_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                    status.write(f"Processed {frame_count} frames")

                cap.release()
                cap = None

                writer.release()
                writer = None

                if frame_count == 0:
                    st.error("No frames were processed.")
                    st.stop()

                # Convert to browser-friendly H.264 MP4
                h264_path = output_path.replace(".mp4", "_h264.mp4")
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

                subprocess.run(
                    [
                        ffmpeg_path,
                        "-y",
                        "-i", output_path,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-an",
                        h264_path
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Store video bytes in session state
                with open(h264_path, "rb") as f:
                    st.session_state.processed_video = f.read()

                st.success("Video detection completed!")

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                for path in [input_path, output_path, h264_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

    # Display the saved result on every rerun
    if st.session_state.processed_video is not None:
        st.subheader("Detected Video")

        st.video(st.session_state.processed_video)

        st.download_button(
            "Download Detected Video",
            data=st.session_state.processed_video,
            file_name="road_damage_result.mp4",
            mime="video/mp4",
            key="download_video"
        )

