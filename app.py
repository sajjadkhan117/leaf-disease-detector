"""
Plant Disease Detection - Live Web App
Classical DIP Pipeline - No ML training needed!
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Severity thresholds ───────────────────────────────────────────────
SEVERITY = {
    "Healthy":  {"range": (0,   5),  "color": "#27ae60", "icon": "🌿", "badge": "HEALTHY"},
    "Mild":     {"range": (5,  20),  "color": "#f39c12", "icon": "⚠️", "badge": "MILD"},
    "Moderate": {"range": (20, 50),  "color": "#e67e22", "icon": "🔶", "badge": "MODERATE"},
    "Severe":   {"range": (50, 100), "color": "#e74c3c", "icon": "🚨", "badge": "SEVERE"},
}

TREATMENT = {
    "Healthy":  "Your plant looks healthy! Keep watering regularly and ensure proper sunlight.",
    "Mild":     "Early signs detected. Remove affected leaves. Improve air circulation around the plant.",
    "Moderate": "Significant infection. Apply appropriate fungicide. Remove all visibly infected leaves immediately.",
    "Severe":   "URGENT! Heavy infection detected. Isolate plant immediately. Apply systemic fungicide or consider removing plant to protect others.",
}

# ── Core DIP Pipeline ─────────────────────────────────────────────────

def remove_background(img):
    """Lab 02 - Remove background, isolate leaf."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 15, 20]),
                            np.array([100, 255, 255]))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.dilate(mask, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        clean = np.zeros_like(mask)
        cv2.drawContours(clean,
                         [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        mask = clean
    leaf = cv2.bitwise_and(img, img, mask=mask)
    return leaf, mask


def enhance_image(img):
    """Lab 05 - CLAHE illumination normalisation."""
    lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def detect_disease(img, leaf_mask):
    """Lab 02+03 - HSV colour masking for disease regions."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellowing (H=15-40), Browning (H=5-18), Blight (H=0-12)
    m1 = cv2.inRange(hsv, np.array([15, 50, 50]),
                          np.array([40, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([5,  50, 30]),
                          np.array([18, 220, 200]))
    m3 = cv2.inRange(hsv, np.array([0,  60, 30]),
                          np.array([12, 255, 180]))

    combined = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    combined = cv2.bitwise_and(combined, leaf_mask)

    # Remove very bright/dark pixels (shadows)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    combined  = cv2.bitwise_and(combined, bright)

    return combined


def clean_mask(mask):
    """Lab 11 - Morphology: opening + closing + top-hat."""
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko)
    cl = cv2.morphologyEx(op,   cv2.MORPH_CLOSE, kc)
    th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, ko)
    _, tht = cv2.threshold(th, 30, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_or(cl, tht)


def detect_edges(img, mask):
    """Lab 07 - Canny edge detection on disease regions."""
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    masked  = cv2.bitwise_and(blurred, blurred,
                               mask=mask if mask.any() else None)
    return cv2.Canny(masked, 30, 100)


def compute_pdi(disease_mask, leaf_mask):
    """Lab 12 - Compute PDI + count lesions."""
    leaf_area    = float(np.sum(leaf_mask > 0)) or 1.0
    disease_area = float(np.sum(disease_mask > 0))
    pdi          = disease_area / leaf_area * 100.0

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        disease_mask, connectivity=8)
    lesions = sum(1 for i in range(1, n)
                  if stats[i, cv2.CC_STAT_AREA] >= 50)
    return round(pdi, 2), lesions


def classify_severity(pdi):
    for name, info in SEVERITY.items():
        lo, hi = info["range"]
        if lo <= pdi < hi:
            return name
    return "Severe"


def build_overlay(original, disease_mask, severity_class):
    """Create red-highlight overlay on disease regions."""
    rgb     = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[disease_mask > 0] = [220, 50, 50]
    blended = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)

    # Draw contours
    color   = SEVERITY[severity_class]["color"]
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    cnts, _ = cv2.findContours(disease_mask,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, cnts, -1, (r, g, b), 2)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


# ── Full pipeline ─────────────────────────────────────────────────────

def run_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (512, 512))

    # Stage 1 - background removal
    leaf, leaf_mask = remove_background(img)

    # Stage 2 - enhancement
    enhanced = enhance_image(leaf)

    # Stage 3 - disease mask
    raw_mask = detect_disease(enhanced, leaf_mask)

    # Stage 4 - edges
    edges = detect_edges(enhanced, raw_mask)

    # Stage 5 - morphology
    clean = clean_mask(raw_mask)

    # Stage 6 - metrics
    pdi, lesions = compute_pdi(clean, leaf_mask)
    severity     = classify_severity(pdi)

    # Overlay
    overlay_img = build_overlay(enhanced, clean, severity)

    # Edge overlay on original
    edge_col = img.copy()
    edge_col[edges > 0] = [0, 0, 255]

    return {
        "original":    img_to_b64(img),
        "enhanced":    img_to_b64(enhanced),
        "overlay":     img_to_b64(overlay_img),
        "edge":        img_to_b64(edge_col),
        "mask":        img_to_b64(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)),
        "pdi":         pdi,
        "lesions":     lesions,
        "severity":    severity,
        "color":       SEVERITY[severity]["color"],
        "icon":        SEVERITY[severity]["icon"],
        "badge":       SEVERITY[severity]["badge"],
        "treatment":   TREATMENT[severity],
        "leaf_area":   int(np.sum(leaf_mask > 0)),
        "disease_area":int(np.sum(clean > 0)),
    }


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty file"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return jsonify({"error": "Unsupported file type"}), 400

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(UPLOAD_FOLDER, f"{ts}{ext}")
    file.save(filepath)

    result = run_pipeline(filepath)
    if result is None:
        return jsonify({"error": "Could not process image"}), 500

    result["success"] = True
    return jsonify(result)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🌿 PLANT DISEASE DETECTION - LIVE")
    print("=" * 50)
    print("  Open: http://localhost:5000")
    print("  Upload a real leaf photo!")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
