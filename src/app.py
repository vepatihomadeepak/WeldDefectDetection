"""Streamlit interface for inspecting weld images with a trained YOLO model."""

from __future__ import annotations

import io
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

# Thread and environment safety to prevent Windows DLL / threadpool crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

import numpy as np
import streamlit as st
import torch
torch.set_num_threads(1)

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    PROJECT_ROOT / "runs" / "detect" / "weld_defect" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "segment" / "weld_defect_seg" / "weights" / "best.pt",
    PROJECT_ROOT / "models" / "best.pt",
)
COLORS = ((239, 68, 68), (34, 197, 94), (59, 130, 246), (245, 158, 11), (168, 85, 247))
GOOD_LABELS = {"good_weld", "good weld", "acceptable_weld", "acceptable weld", "normal_weld", "normal weld"}
DEBUG_CONFIDENCE_FLOOR = 0.10
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Weld Seam Inspector", layout="wide")

# Hide deploy button, toolbar, and hamburger menu
st.markdown("""
<style>
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load a model once per absolute model path."""
    from ultralytics import YOLO
    return YOLO(model_path)


def find_model() -> Path | None:
    """Find a model without relying on Streamlit's working directory."""
    return next((path for path in MODEL_PATHS if path.is_file()), None)


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def is_issue(name: str) -> bool:
    """A recognised good-weld class is a pass; every other class is an issue."""
    return name.lower().replace("-", "_") not in GOOD_LABELS


def annotate(image: np.ndarray, result, alpha: float, report_threshold: float) -> np.ndarray:
    """Draw masks/boxes in RGB, avoiding Ultralytics' BGR plotting output."""
    output = image.copy()
    boxes = result.boxes
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    if boxes is None:
        return output
    for index, box in enumerate(boxes):
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        color = COLORS[class_id % len(COLORS)] if confidence >= report_threshold else (148, 163, 184)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().round().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(output.shape[1] - 1, x2), min(output.shape[0] - 1, y2)
        if masks is not None and index < len(masks):
            mask = cv2.resize(masks[index].astype(np.uint8), (output.shape[1], output.shape[0]), interpolation=cv2.INTER_NEAREST)
            selected = mask.astype(bool)
            overlay = np.zeros_like(output)
            overlay[selected] = color
            output[selected] = cv2.addWeighted(output[selected], 1 - alpha, overlay[selected], alpha, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, color, 2)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name(result.names, class_id).replace('_', ' ')} {confidence:.0%}"
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(height + 6, y1)
        cv2.rectangle(output, (x1, label_y - height - 6), (x1 + width + 6, label_y), color, -1)
        cv2.putText(output, label, (x1 + 3, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def analyze_image(model, image: Image.Image, confidence: float, iou: float, alpha: float, debug_mode: bool):
    """Run one low-floor inference, then separate reportable and debug candidates."""
    source = np.asarray(image.convert("RGB"))
    inference_floor = min(confidence, DEBUG_CONFIDENCE_FLOOR) if debug_mode else confidence
    result = model.predict(source=source, conf=inference_floor, iou=iou, verbose=False)[0]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    detections = []
    if result.boxes is not None:
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            name = class_name(result.names, class_id)
            detections.append({
                "issue": is_issue(name), "class_name": name,
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].cpu().numpy().round().astype(int).tolist(),
                "has_mask": result.masks is not None and index < len(result.masks.data),
            })
    reported = [item for item in detections if item["confidence"] >= confidence]
    metadata = {
        "model": str(getattr(model, "ckpt_path", "unknown model")),
        "task": str(getattr(model, "task", "unknown")),
        "image_dimensions": f"{image.width}x{image.height}",
        "inference_floor": inference_floor,
        "reporting_threshold": confidence,
        "post_nms_candidates": len(detections),
        "reportable_detections": len(reported),
        "detections": detections,
    }
    logger.info("Weld inference: %s", metadata)
    return annotate(source, result, alpha, confidence), detections, reported, metadata


def render_assessment(candidates: list[dict], reported: list[dict], threshold: float) -> None:
    """Keep no inference, low-confidence inference, and issue findings distinct."""
    issues = [item for item in reported if item["issue"]]
    healthy = [item for item in reported if not item["issue"]]
    if issues:
        grouped = defaultdict(list)
        for item in issues:
            grouped[item["class_name"]].append(item)
        st.error(f"Issues detected: {len(issues)} finding(s) across {len(grouped)} issue type(s).")
        for name, findings in grouped.items():
            best = max(findings, key=lambda item: item["confidence"])
            st.write(f"- **{name.replace('_', ' ').title()}** - {len(findings)} finding(s); highest confidence {best['confidence']:.1%}.")
    elif healthy:
        st.info(f"No reportable defect labels were detected. The model identified {len(healthy)} good-weld region(s); this is not a weld-quality pass.")
    elif candidates:
        st.warning(f"The model produced {len(candidates)} post-NMS candidate(s), but all were below the reporting threshold of {threshold:.0%}. They are shown in gray for review; no defect has been confirmed.")
    else:
        st.warning("No objects met the confidence threshold. This is not a quality pass; inspect manually or lower the threshold.")


def do_analysis(model, image, confidence, iou, alpha, debug_mode, upload_name):
    """Run analysis and store results in session state."""
    try:
        annotated, candidates, reported, metadata = analyze_image(
            model, image, confidence, iou, alpha, debug_mode
        )
        buffer = io.BytesIO()
        Image.fromarray(annotated).save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        st.session_state.analysis_result = {
            "annotated": annotated,
            "png_bytes": png_bytes,
            "candidates": candidates,
            "reported": reported,
            "metadata": metadata,
            "confidence": confidence,
            "timestamp": time.time(),
            "upload_name": upload_name,
        }
        st.session_state.analysis_count += 1
    except Exception as error:
        st.session_state.analysis_error = str(error)


def main() -> None:
    # Initialize session state
    for key, default in [("analysis_result", None), ("analysis_count", 0),
                         ("last_upload_name", None), ("analysis_error", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.title("AI Weld Seam Inspector")
    st.caption("Upload an image to identify model-detected weld issues. Results need inspector review.")

    with st.sidebar:
        st.header("Settings")
        detected = find_model()
        custom_path = st.text_input("Model path (optional)", value=str(detected) if detected else "")
        model_path = Path(custom_path).expanduser() if custom_path else None
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.15, 0.05)
        iou = st.slider("IoU threshold", 0.05, 0.95, 0.60, 0.05)
        alpha = st.slider("Mask opacity", 0.1, 0.9, 0.50, 0.1)
        debug_mode = st.checkbox("Debug: show candidates down to 10%", value=True)
        st.caption("Debug mode runs inference at 10% (or your lower setting) and shows below-threshold boxes in gray. NMS still runs inside YOLO.")

    if model_path is None or not model_path.is_file():
        st.error("No trained model found. Put best.pt in models/ or enter a valid path.")
        return
    try:
        model = load_model(str(model_path.resolve()))
    except Exception as error:
        st.error(f"The model could not be loaded: {error}")
        return
    loaded_classes = [class_name(model.names, index) for index in range(len(model.names))]
    st.caption(f"Loaded model: `{model_path.name}` | task: `{getattr(model, 'task', 'unknown')}` | classes: {', '.join(loaded_classes)}")
    if not {"bad_weld", "good_weld", "defect"}.intersection({name.lower() for name in loaded_classes}):
        st.warning("This model does not expose the project's weld classes. Its results are not evidence of weld-defect detection; select a weld-trained checkpoint.")

    upload = st.file_uploader("Upload a weld image", type=("jpg", "jpeg", "png", "webp"))

    # Clear stored results if the user uploads a different image
    if upload is not None and upload.name != st.session_state.last_upload_name:
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.session_state.last_upload_name = upload.name

    if upload is None:
        st.session_state.analysis_result = None
        st.session_state.last_upload_name = None
        st.info("Choose a JPG, JPEG, PNG, or WEBP weld image to begin.")
        return

    try:
        image = Image.open(upload).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"The uploaded file is not a readable image: {error}")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(image, width="stretch")
        st.caption(f"{image.width} x {image.height} px")

    with right:
        st.subheader("Analysis")

        # Button triggers analysis - results persist in session_state
        if st.button(
            "Re-analyze weld" if st.session_state.analysis_result else "Analyze weld",
            type="primary",
            width="stretch",
            key="analyze_button",
        ):
            with st.spinner("Running model inference..."):
                do_analysis(model, image, confidence, iou, alpha, debug_mode, upload.name)

        # Show error if analysis failed
        if st.session_state.analysis_error:
            st.error(f"Analysis failed: {st.session_state.analysis_error}")
            st.session_state.analysis_error = None

        # Show persisted results (survives reruns)
        if st.session_state.analysis_result is not None:
            result = st.session_state.analysis_result
            st.image(result["annotated"], width="stretch")
            render_assessment(result["candidates"], result["reported"], result["confidence"])

    # Detailed results below (always visible when results exist)
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        st.divider()

        # Summary metrics
        issues = [item for item in result["reported"] if item["issue"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total detections", result["metadata"]["post_nms_candidates"])
        c2.metric("Above threshold", result["metadata"]["reportable_detections"])
        c3.metric("Issues found", len(issues))

        st.caption(
            f"Image: {result['metadata']['image_dimensions']} | "
            f"inference floor: {result['metadata']['inference_floor']:.0%} | "
            f"post-NMS candidates: {result['metadata']['post_nms_candidates']} | "
            f"remaining at {result['confidence']:.0%}: {result['metadata']['reportable_detections']}"
        )

        # Detailed findings table rendered via safe Markdown (prevents pyarrow / arrow.dll crash)
        if result["candidates"]:
            st.subheader("Inference details")
            table_rows = [
                "| Status | Assessment | Class | Confidence | Box (original pixels) | Mask |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
            for item in result["candidates"]:
                status = "Reportable" if item["confidence"] >= result["confidence"] else "Below threshold"
                assessment = "Issue" if item["issue"] else "Good weld"
                class_label = item["class_name"].replace("_", " ").title()
                conf_str = f"{item['confidence']:.1%}"
                bbox_str = f"`{item['bbox']}`"
                mask_str = "Yes" if item["has_mask"] else "No"
                table_rows.append(f"| {status} | {assessment} | {class_label} | {conf_str} | {bbox_str} | {mask_str} |")
            st.markdown("\n".join(table_rows))

        # Debug metadata
        with st.expander("Debug metadata"):
            st.json({key: value for key, value in result["metadata"].items() if key != "detections"})
            st.caption("Coordinates are YOLO's xyxy boxes mapped to the original uploaded image.")

        # Download button
        st.download_button(
            "Download annotated image",
            result["png_bytes"],
            f"annotated_{Path(upload.name).stem}.png",
            "image/png",
            width="stretch",
            key="download_button",
        )

        st.info("Upload another image or adjust settings and click Re-analyze to run again.")


if __name__ == "__main__":
    main()
