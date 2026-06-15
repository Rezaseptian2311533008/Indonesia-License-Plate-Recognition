import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import easyocr
import re
import pandas as pd
from datetime import datetime
import io
import base64
from collections import Counter

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indonesia License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; color: #a0c4ff; margin-top: 0.5rem; }

    .metric-card {
        background: #1e2a3a;
        border: 1px solid #2d4a6a;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.75rem; color: #7fa8c9; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #4fc3f7; margin: 0.2rem 0; }
    .metric-card .sub   { font-size: 0.8rem; color: #90a4ae; }

    .plate-result {
        background: linear-gradient(135deg, #0d2137, #1a3a5c);
        border: 2px solid #4fc3f7;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .plate-number {
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: 6px;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 20px #4fc3f7;
    }
    .plate-region { font-size: 1rem; color: #4fc3f7; margin-top: 0.4rem; }

    .valid-badge {
        display: inline-block;
        background: #1b5e20;
        color: #69f0ae;
        border: 1px solid #69f0ae;
        border-radius: 20px;
        padding: 0.2rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .invalid-badge {
        display: inline-block;
        background: #7f0000;
        color: #ff8a80;
        border: 1px solid #ff8a80;
        border-radius: 20px;
        padding: 0.2rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #2d4a6a;
        margin: 1.5rem 0;
    }
    .info-box {
        background: #0d2137;
        border-left: 3px solid #4fc3f7;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #cfd8dc;
        margin: 0.5rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2d4a6a !important;
        border-radius: 8px !important;
    }

    /* History cards */
    .history-card {
        background: #132030;
        border: 1px solid #2d4a6a;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .history-plate {
        font-family: 'Courier New', monospace;
        font-size: 1.3rem;
        font-weight: 800;
        color: #4fc3f7;
        letter-spacing: 3px;
        min-width: 140px;
    }
    .history-meta {
        font-size: 0.78rem;
        color: #78909c;
        line-height: 1.6;
    }
    .history-time {
        font-size: 0.72rem;
        color: #546e7a;
        margin-left: auto;
        white-space: nowrap;
    }
    .source-badge-cam {
        display: inline-block;
        background: #1a237e;
        color: #82b1ff;
        border: 1px solid #82b1ff;
        border-radius: 12px;
        padding: 0.1rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .source-badge-upload {
        display: inline-block;
        background: #1b3a1b;
        color: #69f0ae;
        border: 1px solid #69f0ae;
        border-radius: 12px;
        padding: 0.1rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .stat-box {
        background: #0d2137;
        border: 1px solid #2d4a6a;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        text-align: center;
    }
    .stat-box .sv { font-size: 1.6rem; font-weight: 800; color: #4fc3f7; }
    .stat-box .sl { font-size: 0.72rem; color: #78909c; text-transform: uppercase; letter-spacing: 1px; }

    .fix-badge {
        display: inline-block;
        background: #1a2f1a;
        color: #ffcc02;
        border: 1px solid #ffcc02;
        border-radius: 12px;
        padding: 0.1rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS (Komprehensif Seluruh Wilayah Indonesia)
# ============================================================

PLATE_REGIONS = {
    "A":  "Banten", "B":  "DKI Jakarta / Bekasi / Depok / Tangerang", "D":  "Bandung",
    "E":  "Cirebon", "F":  "Bogor", "G":  "Pekalongan", "H":  "Semarang",
    "K":  "Pati", "L":  "Surabaya", "M":  "Madura", "N":  "Malang",
    "P":  "Besuki", "R":  "Banyumas", "S":  "Bojonegoro", "T":  "Karawang",
    "W":  "Sidoarjo", "Z":  "Garut", "AA": "Magelang", "AB": "Yogyakarta",
    "AD": "Surakarta", "AE": "Madiun", "AG": "Kediri", "BA": "Sumatera Barat",
    "BB": "Tapanuli", "BD": "Bengkulu", "BE": "Lampung", "BG": "Palembang",
    "BH": "Jambi", "BK": "Sumatera Utara", "BL": "Aceh", "BM": "Riau",
    "BN": "Bangka Belitung", "BP": "Kepulauan Riau", "DA": "Kalimantan Selatan",
    "KB": "Kalimantan Barat", "KH": "Kalimantan Tengah", "KT": "Kalimantan Timur",
    "KU": "Kalimantan Utara", "DK": "Bali", "DR": "Lombok / Nusa Tenggara Barat",
    "EA": "Sumbawa / Nusa Tenggara Barat", "EB": "Flores / Nusa Tenggara Timur",
    "ED": "Sumba / Nusa Tenggara Timur", "DH": "Timor / Nusa Tenggara Timur",
    "DC": "Sulawesi Barat", "DD": "Sulawesi Selatan", "DN": "Sulawesi Tengah",
    "DT": "Sulawesi Tenggara", "DL": "Sitaro / Sangihe / Talaud", "DM": "Gorontalo",
    "DE": "Maluku", "DG": "Maluku Utara", "PA": "Papua", "PB": "Papua Barat"
}

CHAR_FIX_DIGIT = {
    "O": "0", "D": "0", "Q": "0", 
    "I": "1", "L": "1", "J": "1", 
    "Z": "2", "E": "3", "A": "4", "H": "4", 
    "S": "5", "G": "6", "T": "7", "B": "8", "R": "8"
}
CHAR_FIX_ALPHA = {
    "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"
}

# ============================================================
# SESSION STATE — Detection History
# ============================================================

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []


def add_to_history(records: list, source: str):
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for r in records:
        st.session_state.detection_history.insert(0, {**r, "Sumber": source, "Waktu": ts})


def clear_history():
    st.session_state.detection_history = []


# ============================================================
# LOAD MODELS (cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO("best.pt")


@st.cache_resource(show_spinner=False)
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def _detect_plate_text_row(gray: np.ndarray) -> tuple:
    edges  = cv2.Canny(gray, 40, 120)
    h_proj = np.sum(edges, axis=1).astype(np.float32)

    kernel = np.ones(max(3, len(h_proj) // 20)) / max(3, len(h_proj) // 20)
    h_proj = np.convolve(h_proj, kernel, mode="same")

    threshold = h_proj.max() * 0.20
    active    = np.where(h_proj > threshold)[0]

    if len(active) == 0:
        return 0.0, 0.70

    y0 = max(0, active[0]  - 4)
    y1 = min(len(h_proj) - 1, active[-1] + 4)

    h  = gray.shape[0]
    return y0 / h, min(y1 / h, 0.78)


def _smart_crop_plate(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    y0_r, y1_r = _detect_plate_text_row(gray)

    y0 = max(0,     int(h * y0_r))
    y1 = min(h - 1, int(h * y1_r))

    if (y1 - y0) < int(h * 0.22):
        y0 = 0
        y1 = int(h * 0.70)

    return img_bgr[y0:y1, :]


def _make_preprocessing_variants(img_bgr: np.ndarray, scale: float = 3.0) -> list:
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    up        = cv2.resize(gray_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    variants = []
    variants.append(up.copy())

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe1.apply(up.copy()))

    block_size = int(up.shape[1] * 0.15)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(11, block_size)
    v3 = cv2.adaptiveThreshold(up.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
    variants.append(v3)

    smooth = cv2.bilateralFilter(up.copy(), 9, 75, 75)
    clahe2 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    variants.append(clahe2.apply(smooth))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants.append(cv2.filter2D(up.copy(), -1, kernel))

    return variants


def preprocess_small_plate(img: np.ndarray) -> np.ndarray:
    cropped = _smart_crop_plate(img)
    up = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    block_size = int(up.shape[1] * 0.15)
    if block_size % 2 == 0: block_size += 1
    return cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, max(11, block_size), 2)


# ============================================================
# OCR FUNCTIONS (UPGRADED: Cross-Variant Patching Engine)
# ============================================================

def _fix_prefix_to_region(prefix: str) -> tuple:
    if prefix in PLATE_REGIONS: return prefix, False
    from itertools import product as iproduct
    positions = [[c] + ([CHAR_FIX_ALPHA[c]] if c in CHAR_FIX_ALPHA else []) for c in prefix]
    for combo in iproduct(*positions):
        candidate = "".join(combo)
        if candidate in PLATE_REGIONS: return candidate, True
    return prefix, False


def run_ocr(reader, img: np.ndarray, allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
    dets = reader.readtext(img, allowlist=allowlist, paragraph=False)
    if not dets:
        dets = reader.readtext(img, decoder="beamsearch")
    return dets


def correct_plate_chars(text: str) -> tuple:
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not text or len(text) < 2: return text, False, -1000

    best_score = -1000
    best_plate = text
    best_fixed = False

    for start in range(len(text)):
        for end in range(start + 3, min(len(text) + 1, start + 10)):
            sub = text[start:end]
            sub_len = len(sub)
            
            for i in [1, 2]: 
                if i >= sub_len: continue
                for j in range(i + 1, min(i + 5, sub_len + 1)): 
                    raw_p, raw_n, raw_s = sub[:i], sub[i:j], sub[j:]
                    if len(raw_s) > 3: continue

                    p_fixed, p_was_fixed = _fix_prefix_to_region("".join(CHAR_FIX_ALPHA.get(c, c) for c in raw_p))
                    n_fixed = "".join(CHAR_FIX_DIGIT.get(c, c) for c in raw_n)
                    s_fixed = "".join(CHAR_FIX_ALPHA.get(c, c) for c in raw_s)

                    score = 0
                    
                    # 1. Scoring Prefix
                    if p_fixed in PLATE_REGIONS:
                        score += 40
                        if not p_was_fixed and raw_p == p_fixed: score += 15
                        if len(p_fixed) == 2: score += 5
                    else:
                        score -= 40

                    # 2. Scoring Blok Angka Tengah
                    if n_fixed.isdigit() and 1 <= len(n_fixed) <= 4:
                        score += 50
                        score += len(n_fixed) * 15 
                        
                        for c in raw_n:
                            if c.isalpha():
                                if c in CHAR_FIX_DIGIT: score -= 4  
                                else: score -= 25 
                    else:
                        score -= 100

                    # 3. Scoring Suffix Belakang
                    if (s_fixed.isalpha() or s_fixed == "") and len(s_fixed) <= 3:
                        score += 30
                        score += len(s_fixed) * 10
                        for c in raw_s:
                            if c.isdigit(): score -= 20 
                    else:
                        score -= 40

                    score += sub_len * 2

                    if score > best_score:
                        best_score = score
                        best_plate = p_fixed + n_fixed + s_fixed
                        best_fixed = (sub != text) or (best_plate != sub)

    return best_plate, best_fixed, best_score


def run_adaptive_ocr(reader, plate_crop: np.ndarray) -> tuple:
    cropped = _smart_crop_plate(plate_crop)
    ch, cw  = cropped.shape[:2]
    mode = "MULTI-PASS OCR (Cross-Variant Patching v6.3)"
    scale = max(3.0, min(8.0, 800.0 / max(cw, 1)))
    variants = _make_preprocessing_variants(cropped, scale=scale)
    
    candidates, all_ocr = [], []

    for vimg in variants:
        dets = run_ocr(reader, vimg)
        if not dets: continue
        v_height = vimg.shape[0]
        
        valid_dets = [d for d in dets if ((d[0][0][1] + d[0][2][1]) / 2) <= (v_height * 0.78)]
        dets_sorted = sorted(valid_dets, key=lambda x: x[0][0][0])
        
        valid_fragments = []
        variant_conf_sum = 0
        for det in dets_sorted:
            if det[2] >= 0.10:
                clean_text = re.sub(r"[^A-Z0-9]", "", det[1].upper())
                if clean_text:
                    valid_fragments.append(clean_text)
                    variant_conf_sum += det[2]
                    all_ocr.append({"text": det[1], "confidence": det[2]})
        
        if valid_fragments:
            raw_plate = "".join(valid_fragments)
            corrected_text, is_fixed, struct_score = correct_plate_chars(raw_plate)
            if corrected_text: 
                avg_variant_conf = variant_conf_sum / len(valid_fragments)
                candidates.append({
                    "plate": corrected_text,
                    "fixed": is_fixed,
                    "struct_score": struct_score,
                    "conf": avg_variant_conf
                })

    if candidates:
        # --- CRITICAL PATCH: Cross-Variant Number Alignment Engine ---
        pure_numbers = set()
        for ocr_item in all_ocr:
            txt = re.sub(r"[^A-Z0-9]", "", ocr_item["text"].upper())
            if txt.isdigit() and 2 <= len(txt) <= 4:
                pure_numbers.add(txt)
        
        patched_candidates = []
        for cand in candidates:
            plate = cand["plate"]
            m = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", plate)
            if m:
                prefix, num, suffix = m.group(1), m.group(2), m.group(3)
                for p_num in pure_numbers:
                    if p_num != num and len(p_num) == len(num):
                        # Filter Hamming Distance Posisional Alignment (mencegah intervensi angka masa berlaku)
                        match_count = sum(1 for c1, c2 in zip(p_num, num) if c1 == c2)
                        if match_count >= max(1, len(num) - 2):
                            patched_plate = prefix + p_num + suffix
                            _, _, new_score = correct_plate_chars(patched_plate)
                            patched_candidates.append({
                                "plate": patched_plate,
                                "fixed": True,
                                "struct_score": new_score,
                                "conf": cand["conf"]
                            })
        candidates.extend(patched_candidates)
        # --- END OF PATCH ---

        best_cand = max(candidates, key=lambda x: x["struct_score"] + (x["conf"] * 25))
        plate_text = best_cand["plate"]
        was_corrected = best_cand["fixed"]
    else:
        plate_text, was_corrected = "", False

    return all_ocr, mode, plate_text, was_corrected


# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def process_and_display(image_np: np.ndarray, source_label: str,
                        model, reader, conf_thresh: float,
                        show_debug: bool):
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("📷 Gambar Input")
    st.image(image_np, use_container_width=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    with st.spinner("🔍 Mendeteksi plat nomor..."):
        results = model.predict(source=image_np, conf=conf_thresh, save=False, verbose=False)

    boxes      = results[0].boxes
    result_img = results[0].plot()

    st.subheader("🎯 Hasil Deteksi YOLO")
    st.image(result_img, channels="BGR", use_container_width=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if len(boxes) == 0:
        st.warning("⚠️ Plat nomor tidak berhasil terdeteksi. Turunkan Confidence Threshold.")
        return []

    st.subheader(f"🔍 Detail Plat Terdeteksi  ({len(boxes)} plat)")
    plate_records = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yolo_conf       = float(box.conf[0])

        h_img, w_img = image_bgr.shape[:2]
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.05)
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w_img, x2 + pad_x)
        y2_pad = min(h_img, y2 + pad_y)

        plate_crop = image_bgr[y1_pad:y2_pad, x1_pad:x2_pad]

        with st.expander(f"Plat #{idx + 1}  —  Confidence YOLO: {yolo_conf:.0%}", expanded=True):
            col_img, col_info = st.columns([1, 2])

            with col_img:
                st.markdown("**Crop Plat**")
                st.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                if show_debug:
                    st.markdown("**Preview Preprocessing (Adaptive Gaussian)**")
                    st.image(preprocess_small_plate(plate_crop), use_container_width=True, clamp=True)

            with col_info:
                with st.spinner("Membaca teks..."):
                    ocr_res, ocr_mode, plate_text, was_corrected = run_adaptive_ocr(reader, plate_crop)

                plate_m = re.match(r"^([A-Z]{1,2})([0-9]{1,4})([A-Z]{0,3})$", plate_text)
                display_text = f"{plate_m.group(1)} {plate_m.group(2)} {plate_m.group(3)}" if plate_m else plate_text

                region    = get_plate_region(plate_text)
                is_valid  = bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", plate_text))
                avg_conf  = sum(r["confidence"] for r in ocr_res) / len(ocr_res) if ocr_res else 0

                valid_badge = '<span class="valid-badge">✔ Format Valid</span>' if is_valid else '<span class="invalid-badge">✘ Format Tidak Valid</span>'
                fix_badge = '<span class="fix-badge">🔧 Karakter Dikoreksi</span>' if was_corrected else ""

                st.markdown(f"""
                <div class="plate-result">
                    <div class="plate-number">{display_text if display_text else "—"}</div>
                    <div class="plate-region">📍 {region}</div>
                    <div style="margin-top:0.5rem">{valid_badge} {fix_badge}</div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1: st.markdown(f'<div class="metric-card"><div class="label">YOLO Conf.</div><div class="value">{yolo_conf:.0%}</div></div>', unsafe_allow_html=True)
                with m2: st.markdown(f'<div class="metric-card"><div class="label">OCR Conf.</div><div class="value">{avg_conf:.0%}</div></div>', unsafe_allow_html=True)
                with m3: st.markdown(f'<div class="metric-card"><div class="label">Mode OCR</div><div class="value" style="font-size:0.95rem;padding-top:0.4rem">{ocr_mode}</div></div>', unsafe_allow_html=True)

                if ocr_res:
                    with st.expander("📄 Fragmen OCR mentah"):
                        for r in ocr_res:
                            st.markdown(f"- `{r['text']}` — conf: {r['confidence']:.0%}")

        plate_records.append({
            "Plat #":      idx + 1,
            "Nomor Plat":  plate_text,
            "Dikoreksi":   "✔" if was_corrected else "—",
            "Daerah Asal": region,
            "Valid":       "✔" if is_valid else "✘",
            "YOLO Conf.":  f"{yolo_conf:.0%}",
            "OCR Conf.":   f"{avg_conf:.0%}",
            "Mode OCR":    ocr_mode,
            "Bbox":        f"({x1},{y1})–({x2},{y2})",
        })

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("📊 Ringkasan Semua Plat")
    df = pd.DataFrame(plate_records)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Hasil (CSV)",
        data=csv,
        file_name=f"plate_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    return plate_records


def get_plate_region(plate: str) -> str:
    plate = plate.upper()
    for code in sorted(PLATE_REGIONS.keys(), key=len, reverse=True):
        if plate.startswith(code):
            return PLATE_REGIONS[code]
    return "Tidak Diketahui"


def validate_plate(text: str) -> bool:
    return bool(re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$", text))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    conf_thresh = st.slider("Confidence Threshold (YOLO)", 0.10, 0.90, 0.30, 0.05)
    show_debug = st.checkbox("Tampilkan Debug Preprocessing", value=False)

    st.markdown("---")
    hist   = st.session_state.detection_history
    st.markdown("### 📈 Statistik Session")
    sc1, sc2 = st.columns(2)
    sc1.markdown(f'<div class="stat-box"><div class="sv">{len(hist)}</div><div class="sl">Total</div></div>', unsafe_allow_html=True)
    sc2.markdown(f'<div class="stat-box"><div class="sv">{sum(1 for h in hist if h.get("Valid") == "✔")}</div><div class="sl">Valid</div></div>', unsafe_allow_html=True)

    if st.button("🗑️ Hapus Semua History", use_container_width=True):
        clear_history()
        st.rerun()

# ============================================================
# APP WINDOW MAIN FLOW
# ============================================================

st.markdown('<div class="main-header"><h1>🚗 Indonesia Smart License Plate Recognition</h1><p>YOLOv8 + Sliding Window Cross-Variant Alignment Engine</p></div>', unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    model = load_yolo()
    st.success("✅ YOLO siap")
with col_m2:
    reader = load_ocr()
    st.success("✅ OCR siap")
with col_m3:
    st.info(f"🕒 {datetime.now().strftime('%d %b %Y · %H:%M')}")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

tab_upload, tab_camera, tab_history = st.tabs(["📁 Upload Gambar", "📷 Kamera Langsung", "🕘 Detection History"])

with tab_upload:
    uploaded_file = st.file_uploader("Upload Gambar Kendaraan", type=["jpg", "jpeg", "png"], key="file_uploader")
    if uploaded_file:
        image_np  = np.array(Image.open(uploaded_file).convert("RGB"))
        records = process_and_display(image_np, "📁 Upload", model, reader, conf_thresh, show_debug)
        if records: add_to_history(records, "📁 Upload")

with tab_camera:
    camera_image = st.camera_input("Ambil Foto Plat Nomor", key="camera_input")
    if camera_image:
        image_np  = np.array(Image.open(camera_image).convert("RGB"))
        records = process_and_display(image_np, "📷 Kamera", model, reader, conf_thresh, show_debug)
        if records: add_to_history(records, "📷 Kamera")

with tab_history:
    if not hist:
        st.info("🕘 Belum ada riwayat deteksi.")
    else:
        st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)