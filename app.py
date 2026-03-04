import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import matplotlib.cm as cm

from ultralytics import YOLO

IMG_SIZE = 1280  # locked

# Class colours
CLASS_COLORS = {
    "glioma":      (255, 99,  99),
    "meningioma":  (99,  220, 255),
    "pituitary":   (255, 210, 80),
}
DEFAULT_COLOR = (180, 180, 180)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* ── Global ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: #080c10 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e2a38 !important;
    }
    [data-testid="stSidebar"] * { color: #8fa3b8 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #c9d8e8 !important; }

    /* ── Main text ── */
    h1, h2, h3, p, span, div, label { color: #c9d8e8 !important; }

    /* ── App title ── */
    h1 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.6rem !important;
        letter-spacing: 0.08em !important;
        color: #e8f4ff !important;
        border-bottom: 1px solid #1e2a38;
        padding-bottom: 0.5rem;
    }

    /* ── Subheaders ── */
    h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: #4a7fa5 !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #0d1117 !important;
        border: 1px dashed #1e2a38 !important;
        border-radius: 4px !important;
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #0d1117 !important;
        border: 1px solid #1e2a38 !important;
        border-radius: 4px !important;
        padding: 14px 18px !important;
    }
    [data-testid="metric-container"] label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #4a7fa5 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.4rem !important;
        color: #e8f4ff !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #0d1117 !important;
        border: 1px solid #1e2a38 !important;
        border-radius: 4px !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        background: #0d1117 !important;
        border: 1px solid #1e2a38 !important;
        border-radius: 4px !important;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] > div > div { background: #1e2a38 !important; }

    /* ── Selectbox ── */
    [data-baseweb="select"] > div {
        background: #0d1117 !important;
        border-color: #1e2a38 !important;
        color: #c9d8e8 !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div { border-top-color: #2e7fb8 !important; }

    /* ── Divider ── */
    hr { border-color: #1e2a38 !important; }

    /* ── Status cards ── */
    .card-detected {
        background: linear-gradient(135deg, #1a0a0a 0%, #2a0f0f 100%);
        border: 1px solid #6b1a1a;
        border-left: 3px solid #e05252;
        padding: 18px 20px;
        border-radius: 4px;
        margin: 8px 0;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .card-detected h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        color: #e05252 !important;
        margin: 0 0 10px 0;
    }
    .card-detected p {
        font-size: 0.88rem !important;
        color: #c4a0a0 !important;
        margin: 4px 0 !important;
    }
    .card-detected strong { color: #f0c8c8 !important; }

    .card-clear {
        background: linear-gradient(135deg, #071510 0%, #0a1e18 100%);
        border: 1px solid #1a4a38;
        border-left: 3px solid #2ecc8f;
        padding: 18px 20px;
        border-radius: 4px;
        margin: 8px 0;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .card-clear h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        color: #2ecc8f !important;
        margin: 0 0 10px 0;
    }
    .card-clear p {
        font-size: 0.88rem !important;
        color: #8ab8a8 !important;
        margin: 4px 0 !important;
    }

    /* ── Class badge ── */
    .badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 2px;
        margin: 3px 3px 3px 0;
        font-weight: 600;
    }
    .badge-glioma     { background: #3d1515; color: #ff6363; border: 1px solid #6b2020; }
    .badge-meningioma { background: #0d2a33; color: #63dcff; border: 1px solid #1a5a6b; }
    .badge-pituitary  { background: #332b05; color: #ffd250; border: 1px solid #6b5a10; }
    .badge-unknown    { background: #1a1a1a; color: #aaaaaa; border: 1px solid #333; }

    /* ── Section label ── */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #4a7fa5;
        border-bottom: 1px solid #1e2a38;
        padding-bottom: 6px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def _to_gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float32)

def _percentile_normalize(gray: np.ndarray):
    mask = gray > 0
    data = gray[mask] if mask.any() else gray.reshape(-1)
    lo, hi = np.percentile(data, (1, 99))
    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo + 1e-6)
    return (gray * 255.0).astype(np.uint8), lo, hi

def _square_pad(pil_img: Image.Image, fill=0) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (fill, fill, fill))
    out.paste(pil_img, ((s - w) // 2, (s - h) // 2))
    return out

def preprocess_mri(pil_img: Image.Image):
    gray = _to_gray_np(pil_img)
    gray_u8, lo, hi = _percentile_normalize(gray)
    preprocessing_info = {
        "original_shape": pil_img.size,
        "percentile_lo": lo,
        "percentile_hi": hi,
    }
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    pil_rgb = Image.fromarray(rgb, mode="RGB")
    pil_rgb = _square_pad(pil_rgb, fill=0)
    pil_rgb = pil_rgb.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    return pil_rgb, preprocessing_info


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

def get_model_safe():
    try:
        return load_model()
    except Exception as e:
        st.error("Failed to load model: {}".format(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def predict_tumor(pil_img, conf=0.20, iou=0.45, min_area=600):
    try:
        proc, preprocessing_info = preprocess_mri(pil_img)
        model = get_model_safe()
        if model is None:
            return None

        start_time = time.time()
        r = model.predict(proc, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]
        inference_time = time.time() - start_time

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return {
                "has_tumor": False,
                "xyxy": np.empty((0, 4)),
                "conf": np.array([]),
                "classes": [],
                "proc_img": proc,
                "preprocessing_info": preprocessing_info,
                "inference_time": inference_time,
            }

        xyxy   = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names  = r.names  # {0: 'glioma', ...}

        wh   = xyxy[:, 2:4] - xyxy[:, 0:2]
        area = wh[:, 0] * wh[:, 1]
        keep = area >= min_area

        xyxy    = xyxy[keep]
        confs   = confs[keep]
        cls_ids = cls_ids[keep]
        classes = [names.get(i, "unknown") for i in cls_ids]

        return {
            "has_tumor": len(xyxy) > 0,
            "xyxy": xyxy,
            "conf": confs,
            "classes": classes,
            "proc_img": proc,
            "preprocessing_info": preprocessing_info,
            "inference_time": inference_time,
        }
    except Exception as e:
        st.error("Error during prediction: {}".format(e))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_boxes_pil(pil_img, xyxy, confs, classes):
    out  = pil_img.copy()
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1), c, cls in zip(xyxy, confs, classes):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = "{} {:.0%}".format(cls, c)
        tw, th = 8 * len(label), 14
        draw.rectangle([x0, max(0, y0 - th - 4), x0 + tw + 6, max(0, y0)],
                       fill=color)
        draw.text((x0 + 3, max(0, y0 - th - 2)), label, fill=(10, 10, 10))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def generate_heatmap(pil_img, xyxy, confs, colormap="inferno", alpha=0.55, sigma_scale=0.35):
    h, w = IMG_SIZE, IMG_SIZE
    heatmap = np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    for (x0, y0, x1, y1), conf_val in zip(xyxy, confs):
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        sx = max(sigma_scale * (x1 - x0) / 2.0, 8.0)
        sy = max(sigma_scale * (y1 - y0) / 2.0, 8.0)
        gauss = np.exp(-(((xx - cx)**2)/(2*sx**2) + ((yy - cy)**2)/(2*sy**2)))
        heatmap += float(conf_val) * gauss
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    cmap = cm.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = Image.blend(pil_img.convert("RGB"),
                          Image.fromarray(colored, "RGB"), alpha=alpha)
    bar_w = 18
    bar = np.linspace(1.0, 0.0, h).reshape(h, 1)
    bar_rgb = (cmap(bar)[:, :, :3] * 255).astype(np.uint8)
    bar_img = Image.fromarray(np.tile(bar_rgb, (1, bar_w, 1)).astype(np.uint8), "RGB")
    canvas = Image.new("RGB", (w + bar_w + 4, h), (8, 12, 16))
    canvas.paste(blended, (0, 0))
    canvas.paste(bar_img, (w + 4, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((w + 5, 4),      "Hi", fill=(200, 200, 200))
    draw.text((w + 5, h - 16), "Lo", fill=(200, 200, 200))
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain MRI — Tumor Detection", layout="wide",
                   page_icon="🧠")
load_css()

st.title("🧠 Brain MRI — Tumor Detection")
st.markdown(
    "<p style='color:#4a7fa5;font-size:0.85rem;margin-top:-8px;font-family:IBM Plex Mono,monospace;"
    "letter-spacing:0.05em;'>YOLO26n · 1280×1280 · glioma · meningioma · pituitary</p>",
    unsafe_allow_html=True
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-label'>Detection</div>", unsafe_allow_html=True)
    conf = st.slider("Confidence Threshold", 0.10, 0.60, 0.20, 0.01,
        help="Lower = more detections. Higher = fewer false positives.")
    iou = st.slider("NMS IoU", 0.30, 0.70, 0.45, 0.01,
        help="Controls overlap suppression.")
    min_area = st.slider("Min Box Area (px²)", 0, 15000, 600, 50,
        help="Filter tiny boxes. 0 = disabled.")

    st.markdown("<div class='section-label' style='margin-top:20px;'>Heatmap</div>",
                unsafe_allow_html=True)
    heatmap_colormap = st.selectbox("Colormap",
        ["inferno", "jet", "hot", "plasma", "magma", "YlOrRd"], index=0)
    heatmap_alpha = st.slider("Opacity", 0.1, 0.9, 0.55, 0.05)
    heatmap_sigma = st.slider("Blob Spread", 0.1, 1.0, 0.35, 0.05)

    st.markdown(
        "<p style='font-size:0.7rem;color:#3a5a72;margin-top:16px;'>"
        "Input: {}×{} px</p>".format(IMG_SIZE, IMG_SIZE),
        unsafe_allow_html=True
    )

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a brain MRI scan", type=["png", "jpg", "jpeg"])

if uploaded:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        if pil_img.size[0] < 64 or pil_img.size[1] < 64:
            st.error("Image too small — minimum 64×64 px.")
        else:
            with st.spinner("Running inference…"):
                out = predict_tumor(pil_img, conf=conf, iou=iou, min_area=min_area)

            if out is None:
                st.error("Prediction failed. Please try again.")
            else:
                # ── Row 1 ─────────────────────────────────────────────────
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("<div class='section-label'>Original</div>",
                                unsafe_allow_html=True)
                    st.image(pil_img, use_container_width=True)
                    with st.expander("Image info"):
                        st.write("Size: {}".format(
                            out['preprocessing_info']['original_shape']))

                with col2:
                    st.markdown("<div class='section-label'>Preprocessed</div>",
                                unsafe_allow_html=True)
                    st.image(out["proc_img"], use_container_width=True)

                with col3:
                    st.markdown("<div class='section-label'>Detection</div>",
                                unsafe_allow_html=True)
                    if out["has_tumor"]:
                        max_conf   = float(out["conf"].max())
                        mean_conf  = float(out["conf"].mean())
                        num_tumors = len(out["conf"])
                        unique_cls = list(dict.fromkeys(out["classes"]))
                        badges = "".join(
                            "<span class='badge badge-{}'>{}</span>".format(
                                c, c.capitalize())
                            for c in unique_cls
                        )
                        st.markdown("""
                            <div class='card-detected'>
                                <h3>⚠ Tumor Detected</h3>
                                <p>{badges}</p>
                                <p><strong>Count:</strong> {n}</p>
                                <p><strong>Max Confidence:</strong> {mx:.1%}</p>
                                <p><strong>Mean Confidence:</strong> {mn:.1%}</p>
                                <p style='font-size:0.72rem;color:#a06060;margin-top:10px;
                                   border-top:1px solid #6b1a1a;padding-top:8px;'>
                                ⚠ Tumor type classification is experimental and
                                may not be reliable. Do not use for diagnosis.</p>
                            </div>
                        """.format(badges=badges, n=num_tumors, mx=max_conf, mn=mean_conf),
                            unsafe_allow_html=True)
                        boxed = draw_boxes_pil(out["proc_img"], out["xyxy"],
                                               out["conf"], out["classes"])
                        st.image(boxed, use_container_width=True)
                    else:
                        st.markdown("""
                            <div class='card-clear'>
                                <h3>✓ No Tumor Detected</h3>
                                <p>No suspicious regions found above the
                                   current confidence threshold.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(out["proc_img"], use_container_width=True)

                # ── Heatmap ───────────────────────────────────────────────
                if out["has_tumor"]:
                    st.markdown("---")
                    st.markdown("<div class='section-label'>Confidence Heatmap</div>",
                                unsafe_allow_html=True)
                    heat_img = generate_heatmap(
                        out["proc_img"], out["xyxy"], out["conf"],
                        colormap=heatmap_colormap,
                        alpha=heatmap_alpha,
                        sigma_scale=heatmap_sigma,
                    )
                    st.image(heat_img, use_container_width=True,
                             caption="Brighter = higher detection confidence")

                # ── Detection table ───────────────────────────────────────
                if out["has_tumor"]:
                    st.markdown("---")
                    st.markdown("<div class='section-label'>Detection Details</div>",
                                unsafe_allow_html=True)
                    tumor_data = []
                    for i, (bbox, c, cls) in enumerate(
                            zip(out["xyxy"], out["conf"], out["classes"])):
                        x0, y0, x1, y1 = bbox
                        tumor_data.append({
                            "ID":         i + 1,
                            "Type":       cls,
                            "Confidence": "{:.1%}".format(c),
                            "X":          int(x0),
                            "Y":          int(y0),
                            "W":          int(x1 - x0),
                            "H":          int(y1 - y0),
                        })
                    st.dataframe(tumor_data, use_container_width=True)

                # ── Metrics ───────────────────────────────────────────────
                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Inference Time",
                              "{:.2f} s".format(out['inference_time']))
                with m2:
                    st.metric("Detections", len(out["conf"]))
                with m3:
                    if out["has_tumor"]:
                        st.metric("Peak Confidence",
                                  "{:.1%}".format(float(out["conf"].max())))
                    else:
                        st.metric("Peak Confidence", "—")

    except Exception as e:
        st.error("Error processing image: {}".format(e))
        st.info("Please try a different image.")

else:
    st.info("Upload a brain MRI image (PNG or JPG) to begin.")
    with st.expander("How to use"):
        st.markdown("""
        1. Upload a brain MRI scan in the box above
        2. The model detects glioma, meningioma, and pituitary tumors
        3. Adjust thresholds in the sidebar if needed
        4. Review bounding boxes, heatmap, and detection table

        **Threshold guide:**
        - Confidence ↓ → more detections, more false positives
        - Confidence ↑ → fewer detections, more false negatives
        - Min Area filters out tiny spurious boxes

        > This tool is for research purposes only and is not a medical device.
        """)
