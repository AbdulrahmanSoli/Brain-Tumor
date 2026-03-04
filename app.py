import sys
import types

# Ultralytics imports cv2 at module level, which needs libGL.so.1 — unavailable
# on Streamlit Cloud. We only use PIL for image I/O so we mock cv2 out entirely.
_cv2_mock = types.ModuleType("cv2")
_cv2_mock.__version__ = "4.8.0"
sys.modules["cv2"] = _cv2_mock

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import matplotlib.cm as cm

from ultralytics import YOLO

IMG_SIZE = 960  # locked


# Custom CSS for better styling
def load_css():
    st.markdown("""
        <style>
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 10px 0;
            }
            .tumor-detected {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;
            }
            .no-tumor {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin: 10px 0;
            }
            .parameter-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
        </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Preprocessing
# -----------------------------
def _to_gray_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float32)

def _percentile_normalize(gray: np.ndarray):
    mask = gray > 0
    data = gray[mask] if mask.any() else gray.reshape(-1)

    lo, hi = np.percentile(data, (1, 99))
    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo + 1e-6)
    return (gray * 255.0).astype(np.uint8), lo, hi

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

def preprocess_mri(pil_img: Image.Image, margin: float = 0.08):
    gray = _to_gray_np(pil_img)
    gray_u8, lo, hi = _percentile_normalize(gray)

    bbox = _brain_bbox(gray_u8)
    preprocessing_info = {
        "original_shape": pil_img.size,
        "percentile_lo": lo,
        "percentile_hi": hi,
        "brain_bbox": bbox
    }

    if bbox is not None:
        h, w = gray_u8.shape
        x0, y0, x1, y1 = _expand_bbox(*bbox, w=w, h=h, margin=margin)
        gray_u8 = gray_u8[y0:y1 + 1, x0:x1 + 1]

    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    pil_rgb = Image.fromarray(rgb, mode="RGB")

    pil_rgb = _square_pad(pil_rgb, fill=0)
    pil_rgb = pil_rgb.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    return pil_rgb, preprocessing_info


# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

def get_model_safe():
    try:
        return load_model()
    except Exception as e:
        st.error("Failed to load model: {}".format(e))
        return None


# -----------------------------
# Postprocessing + draw
# -----------------------------
def predict_tumor(pil_img: Image.Image, conf=0.45, iou=0.45, min_area=1800):
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
                "proc_img": proc,
                "preprocessing_info": preprocessing_info,
                "inference_time": inference_time
            }

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        wh = (xyxy[:, 2:4] - xyxy[:, 0:2])
        area = wh[:, 0] * wh[:, 1]
        keep = area >= min_area

        xyxy, confs = xyxy[keep], confs[keep]
        has_tumor = len(xyxy) > 0
        
        return {
            "has_tumor": has_tumor,
            "xyxy": xyxy,
            "conf": confs,
            "proc_img": proc,
            "preprocessing_info": preprocessing_info,
            "inference_time": inference_time
        }
    except Exception as e:
        st.error("Error during prediction: {}".format(e))
        return None

def draw_boxes_pil(pil_img: Image.Image, xyxy: np.ndarray, confs: np.ndarray) -> Image.Image:
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)

    for (x0, y0, x1, y1), c in zip(xyxy, confs):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
        draw.text((x0, max(0, y0 - 16)), "{:.2f}".format(c), fill=(0, 255, 0))

    return out


# -----------------------------
# Heatmap
# -----------------------------
def generate_heatmap(pil_img, xyxy, confs, colormap="jet", alpha=0.55, sigma_scale=0.35):
    """
    Overlay a confidence-weighted Gaussian heatmap on the preprocessed MRI.
    Each detection box contributes a Gaussian blob weighted by its confidence score.
    """
    h, w = IMG_SIZE, IMG_SIZE
    heatmap = np.zeros((h, w), dtype=np.float32)

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    for (x0, y0, x1, y1), conf_val in zip(xyxy, confs):
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        sigma_x = max(sigma_scale * (x1 - x0) / 2.0, 8.0)
        sigma_y = max(sigma_scale * (y1 - y0) / 2.0, 8.0)
        gauss = np.exp(
            -(((xx - cx) ** 2) / (2 * sigma_x ** 2) +
              ((yy - cy) ** 2) / (2 * sigma_y ** 2))
        )
        heatmap += float(conf_val) * gauss

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    cmap = cm.get_cmap(colormap)
    colored_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(colored_rgb, "RGB")

    blended = Image.blend(pil_img.convert("RGB"), heat_pil, alpha=alpha)

    # Colorbar
    bar_w = 22
    bar = np.linspace(1.0, 0.0, h).reshape(h, 1)
    bar_rgb = (cmap(bar)[:, :, :3] * 255).astype(np.uint8)
    bar_img = Image.fromarray(np.tile(bar_rgb, (1, bar_w, 1)).astype(np.uint8), "RGB")

    canvas = Image.new("RGB", (w + bar_w + 6, h), (20, 20, 20))
    canvas.paste(blended, (0, 0))
    canvas.paste(bar_img, (w + 6, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((w + 7, 4),      "High", fill=(255, 255, 255))
    draw.text((w + 7, h - 16), "Low",  fill=(255, 255, 255))

    return canvas


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Brain MRI Tumor Detection", layout="wide")
load_css()

# Header
st.title("Brain MRI Tumor Detection")
st.markdown("Using YOLOv8 for real-time tumor detection in brain MRI scans")

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")
    
    st.markdown("### Model Parameters")
    conf = st.slider(
        "Confidence Threshold",
        0.20, 0.90, 0.45, 0.01,
        help="Minimum confidence score for detection. Lower = more detections."
    )
    iou = st.slider(
        "NMS IoU",
        0.30, 0.70, 0.45, 0.01,
        help="Intersection over Union threshold for Non-Maximum Suppression."
    )
    min_area = st.slider(
        "Minimum Box Area",
        300, 15000, 1800, 50,
        help="Filter out detections smaller than this pixel area (in 960x960 image)."
    )

    st.markdown("### Heatmap Settings")
    heatmap_colormap = st.selectbox(
        "Colormap",
        ["jet", "hot", "inferno", "plasma", "magma", "YlOrRd"],
        index=0,
        help="Color scheme for the heatmap overlay."
    )
    heatmap_alpha = st.slider(
        "Heatmap Opacity",
        0.1, 0.9, 0.55, 0.05,
        help="How strongly the heatmap blends over the MRI."
    )
    heatmap_sigma = st.slider(
        "Blob Spread",
        0.1, 1.0, 0.35, 0.05,
        help="How wide each detection's heatmap blob spreads."
    )

    st.markdown("**Input Resolution:** {}x{}".format(IMG_SIZE, IMG_SIZE))
    st.markdown("All images are resized to this square resolution for consistent processing.")


# Main content
uploaded = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
        
        if pil_img.size[0] < 64 or pil_img.size[1] < 64:
            st.error("Image is too small. Please upload an image with dimensions >= 64x64 pixels.")
        else:
            with st.spinner("Processing image..."):
                out = predict_tumor(pil_img, conf=conf, iou=iou, min_area=min_area)
            
            if out is None:
                st.error("Prediction failed. Please try again.")
            else:
                # ── Row 1: Original | Preprocessed | Detection ───────────────
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.subheader("Original Image")
                    st.image(pil_img, use_container_width=True)
                    with st.expander("Image Info"):
                        st.write("**Size:** {}".format(out['preprocessing_info']['original_shape']))

                with col2:
                    st.subheader("Preprocessed (960x960)")
                    st.image(out["proc_img"], use_container_width=True)

                with col3:
                    st.subheader("Detection Result")
                    if out["has_tumor"]:
                        max_conf = float(out["conf"].max())
                        mean_conf = float(out["conf"].mean())
                        num_tumors = len(out["conf"])
                        
                        st.markdown("""
                            <div class="tumor-detected">
                                <h3>Tumor Detected</h3>
                                <p><strong>Count:</strong> {}</p>
                                <p><strong>Max Confidence:</strong> {:.2%}</p>
                                <p><strong>Mean Confidence:</strong> {:.2%}</p>
                            </div>
                        """.format(num_tumors, max_conf, mean_conf), unsafe_allow_html=True)
                        
                        boxed = draw_boxes_pil(out["proc_img"], out["xyxy"], out["conf"])
                        st.image(boxed, use_container_width=True)
                        
                        st.markdown("### Detection Details")
                        tumor_data = []
                        for i, (bbox, c) in enumerate(zip(out["xyxy"], out["conf"])):
                            x0, y0, x1, y1 = bbox
                            bw = x1 - x0
                            bh = y1 - y0
                            tumor_data.append({
                                "Tumor ID": i + 1,
                                "Confidence": "{:.2%}".format(c),
                                "X": "{}".format(int(x0)),
                                "Y": "{}".format(int(y0)),
                                "Width": "{}".format(int(bw)),
                                "Height": "{}".format(int(bh))
                            })
                        st.dataframe(tumor_data, use_container_width=True)
                    else:
                        st.markdown("""
                            <div class="no-tumor">
                                <h3>No Tumor Detected</h3>
                                <p>The model did not find any suspicious regions.</p>
                            </div>
                        """, unsafe_allow_html=True)

                # ── Row 2: Heatmap (only shown when tumor detected) ───────────
                if out["has_tumor"]:
                    st.markdown("---")
                    st.subheader("Confidence Heatmap")
                    heat_img = generate_heatmap(
                        out["proc_img"],
                        out["xyxy"],
                        out["conf"],
                        colormap=heatmap_colormap,
                        alpha=heatmap_alpha,
                        sigma_scale=heatmap_sigma,
                    )
                    st.image(heat_img, use_container_width=True,
                             caption="Confidence heatmap — red/yellow = high confidence, blue = low")

                    with st.expander("How the heatmap works"):
                        st.markdown("""
                        Each detected bounding box contributes a **Gaussian blob** centred on
                        the box, weighted by its **confidence score**. Blobs are summed and
                        normalised, then the chosen colormap is applied.

                        - **Hot colours** (red/yellow) → high confidence regions
                        - **Cool colours** (blue/purple) → low or no activity
                        - Use the sidebar sliders to adjust opacity and blob spread
                        """)

                # ── Row 3: Metrics ────────────────────────────────────────────
                st.markdown("---")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Processing Time", "{:.2f}s".format(out['inference_time']))
                
                with metric_col2:
                    st.metric("Detections", len(out["conf"]))
                
                with metric_col3:
                    if out["has_tumor"]:
                        max_conf = float(out["conf"].max())
                        st.metric("Highest Confidence", "{:.2%}".format(max_conf))
                    else:
                        st.metric("Highest Confidence", "N/A")

    except Exception as e:
        st.error("Error processing image: {}".format(e))
        st.info("Please try uploading a different image.")

else:
    st.info("Upload a brain MRI image to get started with tumor detection.")
    
    with st.expander("How to use this app"):
        st.markdown("""
        1. **Upload an MRI image** in PNG or JPG format
        2. **Adjust detection settings** in the sidebar if needed
        3. **View results** showing the original, preprocessed, and annotated images
        4. **Review the heatmap** to see confidence intensity across the scan

        **Detection tips:**
        - Lower confidence threshold = more detections (more false positives)
        - Higher confidence threshold = fewer detections (more false negatives)
        - Adjust minimum area to filter out very small detected regions

        **Heatmap tips:**
        - jet / hot give classic medical-imaging style coloring
        - Increase Blob Spread for a smoother, wider activation region
        - Reduce Heatmap Opacity to see the underlying MRI more clearly
        """)
