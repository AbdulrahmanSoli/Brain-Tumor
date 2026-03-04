import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

IMG_SIZE = 960  # locked


# -----------------------------
# Preprocessing
# -----------------------------
def _to_gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float32)

def _percentile_normalize(gray: np.ndarray) -> np.ndarray:
    mask = gray > 0
    data = gray[mask] if mask.any() else gray.reshape(-1)

    lo, hi = np.percentile(data, (1, 99))
    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo + 1e-6)
    return (gray * 255.0).astype(np.uint8)

def _brain_bbox(gray_u8: np.ndarray):
    t = max(5, int(np.percentile(gray_u8, 10)))
    mask = gray_u8 > t

    if mask.sum() < 100:
        mask = gray_u8 > 0
        if mask.sum() < 100:
            return None

    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _expand_bbox(x0, y0, x1, y1, w, h, margin=0.08):
    bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return x0, y0, x1, y1

def _square_pad(pil_img: Image.Image, fill=0) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (fill, fill, fill))
    out.paste(pil_img, ((s - w) // 2, (s - h) // 2))
    return out

def preprocess_mri(pil_img: Image.Image, margin: float = 0.08) -> Image.Image:
    gray = _to_gray_np(pil_img)
    gray_u8 = _percentile_normalize(gray)

    bbox = _brain_bbox(gray_u8)
    if bbox is not None:
        h, w = gray_u8.shape
        x0, y0, x1, y1 = _expand_bbox(*bbox, w=w, h=h, margin=margin)
        gray_u8 = gray_u8[y0:y1 + 1, x0:x1 + 1]

    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    pil_rgb = Image.fromarray(rgb, mode="RGB")

    pil_rgb = _square_pad(pil_rgb, fill=0)
    pil_rgb = pil_rgb.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    return pil_rgb


# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


# -----------------------------
# Postprocessing + draw
# -----------------------------
def predict_tumor(pil_img: Image.Image, conf=0.25, iou=0.45, min_area=80):
    proc = preprocess_mri(pil_img)

    r = model.predict(proc, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return {"has_tumor": False, "xyxy": np.empty((0, 4)), "conf": np.array([]), "proc_img": proc}

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    wh = (xyxy[:, 2:4] - xyxy[:, 0:2])
    area = wh[:, 0] * wh[:, 1]
    keep = area >= min_area

    xyxy, confs = xyxy[keep], confs[keep]
    has_tumor = len(xyxy) > 0
    return {"has_tumor": has_tumor, "xyxy": xyxy, "conf": confs, "proc_img": proc}

def draw_boxes_pil(pil_img: Image.Image, xyxy: np.ndarray, confs: np.ndarray) -> Image.Image:
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)

    for (x0, y0, x1, y1), c in zip(xyxy, confs):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        draw.text((x0, max(0, y0 - 16)), f"{c:.2f}", fill=(0, 255, 0))

    return out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Brain MRI Tumor Detection", layout="wide")
st.title("Brain MRI Tumor Detection (YOLO)")

with st.sidebar:
    st.header("Controls")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.01)
    min_area = st.slider("Min box area filter", 0, 5000, 80, 10)
    st.caption(f"Input is always resized to {IMG_SIZE}×{IMG_SIZE}.")

uploaded = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    out = predict_tumor(pil_img, conf=conf, iou=iou, min_area=min_area)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with col2:
        st.subheader("Preprocessed (960×960)")
        st.image(out["proc_img"], use_container_width=True)

    with col3:
        st.subheader("Result")
        if out["has_tumor"]:
            max_conf = float(out["conf"].max()) if len(out["conf"]) else 0.0
            st.success(f"Tumor detected (max conf: {max_conf:.2f})")
            boxed = draw_boxes_pil(out["proc_img"], out["xyxy"], out["conf"])
            st.image(boxed, use_container_width=True)
        else:
            st.warning("No tumor detected")
else:
    st.info("Upload an image to run detection.")
