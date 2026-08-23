import time

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from imot.color_mask import mask_and_stats, overlay_red_mask, preset_hsv
from imot.video import VideoSource, iter_frames, open_uploaded_video, open_webcam
from imot.yolo_infer import default_weights, run_yolo


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _read_uploaded_image_bgr(uploaded_img) -> np.ndarray:
    data = np.frombuffer(uploaded_img.getbuffer(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _read_camera_input_bgr(camera_file) -> np.ndarray:
    data = np.frombuffer(camera_file.getbuffer(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode camera image.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _open_source(source: str, uploaded, webcam_index: int) -> VideoSource:
    if source == "Upload video":
        return open_uploaded_video(uploaded.getbuffer().tobytes(), uploaded.name)
    return open_webcam(int(webcam_index))


st.set_page_config(page_title="IMOT: Colour + Object Detection", layout="wide")
st.title("IMOT: Colour-Based Behavioral Analysis + Object Detection")

with st.sidebar:
    st.subheader("Input")
    source = st.radio("Source", ["Webcam Snapshot", "Upload image", "Upload video", "Webcam"], index=1)
    uploaded = None
    uploaded_img = None
    camera_snapshot = None
    webcam_index = 0
    if source == "Webcam Snapshot":
        camera_snapshot = st.camera_input("Take a picture")
    elif source == "Upload image":
        uploaded_img = st.file_uploader("Image file", type=["png", "jpg", "jpeg", "webp", "bmp"])
    elif source == "Upload video":
        uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    else:
        webcam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)

    st.divider()
    st.subheader("Processing")
    max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=300, step=50)
    frame_stride = st.number_input("Frame stride", min_value=1, value=1, step=1)
    preview_fps = st.slider("Preview FPS", 1, 30, 12)

tab_color, tab_yolo, tab_combined = st.tabs(["Colur Detection (HSV)", "Object Detection (YOLOv8)", "Combined"])

if source == "Upload video" and uploaded is None:
    st.info("Upload a video to start, or switch to Webcam / Upload image in the sidebar.")
    st.stop()
if source == "Upload image" and uploaded_img is None:
    st.info("Upload an image to start, or switch to Upload video / Webcam in the sidebar.")
    st.stop()
if source == "Webcam Snapshot" and camera_snapshot is None:
    st.info("Take a webcam snapshot to start, or switch input source in the sidebar.")
    st.stop()


with tab_color:
    st.subheader("HSV Colour Mask")

    col1, col2 = st.columns([1.2, 1])
    with col2:
        mode = st.selectbox(
            "Detection Mode",
            ["All Colors", "Red", "Blue", "Green", "Every Color Except White", "Custom HSV"],
            index=0,
        )

        custom_low = None
        custom_high = None
        if mode == "Custom HSV":
            st.markdown("### Custom HSV Range")
            h_low = st.slider("Hue Low", 0, 179, 0)
            s_low = st.slider("Saturation Low", 0, 255, 42)
            v_low = st.slider("Value Low", 0, 255, 0)
            h_high = st.slider("Hue High", 0, 179, 179)
            s_high = st.slider("Saturation High", 0, 255, 255)
            v_high = st.slider("Value High", 0, 255, 255)
            custom_low = np.array([h_low, s_low, v_low], dtype=np.uint8)
            custom_high = np.array([h_high, s_high, v_high], dtype=np.uint8)

        run_color = st.button("Run Colour Detection", type="primary")

    if run_color:
        rows: list[dict] = []
        frame_ph = col1.empty()
        table_ph = col2.empty()

        def run_one_frame(frame_bgr: np.ndarray, idx: int) -> dict[str, dict]:
            if mode == "All Colors":
                out: dict[str, dict] = {}
                for name in ["Red", "Blue", "Green", "Every Color Except White"]:
                    low, high = preset_hsv(name)
                    s = mask_and_stats(frame_bgr, low, high)
                    out[name] = s
                return out
            if mode == "Custom HSV":
                assert custom_low is not None and custom_high is not None
                return {"Custom": mask_and_stats(frame_bgr, custom_low, custom_high)}
            low, high = preset_hsv(mode)
            return {mode: mask_and_stats(frame_bgr, low, high)}

        def downloads_for_results(frame_bgr: np.ndarray, results: dict[str, dict]) -> None:
            st.markdown("---")
            st.subheader("Download Results (PNG)")
            for key, s in results.items():
                res_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=s["mask"])
                ok, buf = cv2.imencode(".png", res_bgr)
                if not ok:
                    continue
                st.download_button(
                    label=f"Download {key} Result",
                    data=buf.tobytes(),
                    file_name=f"{key.lower().replace(' ', '_')}_result.png",
                    mime="image/png",
                )

        def display_results(frame_bgr: np.ndarray, idx_label: str, results: dict[str, dict]) -> None:
            st.markdown("---")
            st.subheader("Results")

            if mode == "All Colors":
                a, b, c = st.columns(3)
                with a:
                    st.image(_bgr_to_rgb(frame_bgr), caption="Original", use_container_width=True)
                with b:
                    st.image(results["Red"]["mask"], caption="Red Mask", use_container_width=True)
                    st.image(_bgr_to_rgb(cv2.bitwise_and(frame_bgr, frame_bgr, mask=results["Red"]["mask"])), caption="Red Detection", use_container_width=True)
                with c:
                    st.image(results["Blue"]["mask"], caption="Blue Mask", use_container_width=True)
                    st.image(_bgr_to_rgb(cv2.bitwise_and(frame_bgr, frame_bgr, mask=results["Blue"]["mask"])), caption="Blue Detection", use_container_width=True)

                d, e, f = st.columns(3)
                with d:
                    st.image(results["Green"]["mask"], caption="Green Mask", use_container_width=True)
                    st.image(_bgr_to_rgb(cv2.bitwise_and(frame_bgr, frame_bgr, mask=results["Green"]["mask"])), caption="Green Detection", use_container_width=True)
                with e:
                    st.image(results["Every Color Except White"]["mask"], caption="Non-White Mask", use_container_width=True)
                    st.image(
                        _bgr_to_rgb(cv2.bitwise_and(frame_bgr, frame_bgr, mask=results["Every Color Except White"]["mask"])),
                        caption="Every Color Except White",
                        use_container_width=True,
                    )
                with f:
                    st.empty()
            else:
                key = list(results.keys())[0]
                s = results[key]
                a, b, c = st.columns(3)
                with a:
                    st.image(_bgr_to_rgb(frame_bgr), caption="Original", use_container_width=True)
                with b:
                    st.image(s["mask"], caption=f"{key} Mask", use_container_width=True)
                with c:
                    res_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=s["mask"])
                    st.image(_bgr_to_rgb(res_bgr), caption=f"{key} Detection", use_container_width=True)
                    if key == "Custom" and custom_low is not None and custom_high is not None:
                        st.info(f"HSV Range: Low = {custom_low.tolist()}, High = {custom_high.tolist()}")

            downloads_for_results(frame_bgr, results)

        if source in {"Upload image", "Webcam Snapshot"}:
            frame = _read_uploaded_image_bgr(uploaded_img) if source == "Upload image" else _read_camera_input_bgr(camera_snapshot)
            results = run_one_frame(frame, 0)
            display_results(frame, "Image", results)

            # CSV (one row per key)
            for key, s in results.items():
                rows.append(
                    {
                        "frame_index": 0,
                        "mode": key,
                        "coverage": s["coverage"],
                        "num_contours": s["num_contours"],
                        "largest_contour_area": s["largest_contour_area"],
                    }
                )
            df = pd.DataFrame(rows)
            table_ph.dataframe(df, use_container_width=True, height=240)
        else:
            vs = _open_source(source, uploaded, webcam_index)
            try:
                for idx, frame in iter_frames(vs.cap, stride=int(frame_stride), max_frames=int(max_frames)):
                    results = run_one_frame(frame, idx)

                    # Show a single preview image in the left pane
                    if mode == "All Colors":
                        low, high = preset_hsv("Red")
                        s = mask_and_stats(frame, low, high)
                        preview_bgr = overlay_red_mask(frame, s["mask"])
                    else:
                        key = list(results.keys())[0]
                        preview_bgr = overlay_red_mask(frame, results[key]["mask"])
                    frame_ph.image(_bgr_to_rgb(preview_bgr), caption=f"Frame {idx}", use_container_width=True)

                    # Stats rows
                    for key, s in results.items():
                        rows.append(
                            {
                                "frame_index": idx,
                                "mode": key,
                                "coverage": s["coverage"],
                                "num_contours": s["num_contours"],
                                "largest_contour_area": s["largest_contour_area"],
                            }
                        )

                    df = pd.DataFrame(rows)
                    table_ph.dataframe(df.tail(50), use_container_width=True, height=420)
                    time.sleep(1.0 / float(preview_fps))
            finally:
                vs.release()

        if rows:
            df = pd.DataFrame(rows)
            st.download_button(
                "Download Colour CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="colour_mask_stats.csv",
                mime="text/csv",
            )


with tab_yolo:
    st.subheader("YOLOv8 (Ultralytics)")
    left, right = st.columns([1.2, 1])

    with right:
        task = st.selectbox("Task", ["detect", "segment", "pose"], index=0)
        weights = st.text_input("Weights path/name", value=default_weights(task))
        run_od = st.button("Run Object Detection", type="primary")

        st.caption(
            "Tip: the first run will auto-download weights if you use names like "
            "`yolov8n.pt`, `yolov8n-seg.pt`, `yolov8n-pose.pt`."
        )

    if run_od:
        rows: list[dict] = []
        frame_ph = left.empty()
        met_ph = right.empty()
        table_ph = right.empty()
        if source == "Upload image":
            frame = _read_uploaded_image_bgr(uploaded_img)
            idx = 0
            out = run_yolo(frame, task=task, weights_path=weights)
            frame_ph.image(_bgr_to_rgb(out.annotated_bgr), caption="Image result", use_container_width=True)
            rows.append({"frame_index": idx, "num_objects": out.num_objects})
            df = pd.DataFrame(rows)
            with met_ph.container():
                st.metric("Objects", out.num_objects)
            table_ph.dataframe(df, use_container_width=True, height=200)
        else:
            vs = _open_source(source, uploaded, webcam_index)
            try:
                for idx, frame in iter_frames(vs.cap, stride=int(frame_stride), max_frames=int(max_frames)):
                    out = run_yolo(frame, task=task, weights_path=weights)
                    frame_ph.image(_bgr_to_rgb(out.annotated_bgr), caption=f"Frame {idx}", use_container_width=True)

                    rows.append({"frame_index": idx, "num_objects": out.num_objects})
                    df = pd.DataFrame(rows)

                    with met_ph.container():
                        st.metric("Objects (last)", out.num_objects)
                        st.metric("Processed frames", len(rows))

                    table_ph.dataframe(df.tail(25), use_container_width=True, height=360)
                    time.sleep(1.0 / float(preview_fps))
            finally:
                vs.release()

        if rows:
            df = pd.DataFrame(rows)
            st.download_button(
                "Download Object CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="object_detection_stats.csv",
                mime="text/csv",
            )


with tab_combined:
    st.subheader("Combined: Colour Mask + YOLO")
    st.write("Runs YOLO first, then overlays the colour mask on the YOLO annotated frame.")

    left, right = st.columns([1.2, 1])

    with right:
        preset = st.selectbox("Colour preset", ["Red", "Blue", "Green", "Every color except white"], index=0, key="comb_preset")
        low, high = preset_hsv(preset)
        task = st.selectbox("YOLO task", ["detect", "segment", "pose"], index=0, key="comb_task")
        weights = st.text_input("YOLO weights", value=default_weights(task), key="comb_weights")
        run_both = st.button("Run Combined", type="primary")

    if run_both:
        rows: list[dict] = []
        frame_ph = left.empty()
        table_ph = right.empty()
        if source == "Upload image":
            frame = _read_uploaded_image_bgr(uploaded_img)
            idx = 0
            y = run_yolo(frame, task=task, weights_path=weights)
            c = mask_and_stats(frame, low, high)
            combined_bgr = overlay_red_mask(y.annotated_bgr, c["mask"])
            frame_ph.image(_bgr_to_rgb(combined_bgr), caption="Image result", use_container_width=True)
            rows.append(
                {
                    "frame_index": idx,
                    "num_objects": y.num_objects,
                    "coverage": c["coverage"],
                    "num_contours": c["num_contours"],
                    "largest_contour_area": c["largest_contour_area"],
                }
            )
            df = pd.DataFrame(rows)
            table_ph.dataframe(df, use_container_width=True, height=240)
        else:
            vs = _open_source(source, uploaded, webcam_index)
            try:
                for idx, frame in iter_frames(vs.cap, stride=int(frame_stride), max_frames=int(max_frames)):
                    y = run_yolo(frame, task=task, weights_path=weights)
                    c = mask_and_stats(frame, low, high)
                    combined_bgr = overlay_red_mask(y.annotated_bgr, c["mask"])
                    frame_ph.image(_bgr_to_rgb(combined_bgr), caption=f"Frame {idx}", use_container_width=True)

                    rows.append(
                        {
                            "frame_index": idx,
                            "num_objects": y.num_objects,
                            "coverage": c["coverage"],
                            "num_contours": c["num_contours"],
                            "largest_contour_area": c["largest_contour_area"],
                        }
                    )
                    df = pd.DataFrame(rows)
                    table_ph.dataframe(df.tail(25), use_container_width=True, height=420)
                    time.sleep(1.0 / float(preview_fps))
            finally:
                vs.release()

        if rows:
            df = pd.DataFrame(rows)
            st.download_button(
                "Download Combined CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="combined_stats.csv",
                mime="text/csv",
            )

