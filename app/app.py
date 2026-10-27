import sys
import os
import time
from PIL import Image
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_image

MODEL_NAME = "MobileNetV2 (Feature Extraction)"
INPUT_SIZE = "224 × 224 × 3"
DATASET_NAME = "Chest X-Ray Pneumonia (5,863 images)"
ALLOWED_TYPES = ["jpeg", "jpg", "png", "bmp"]
BORDER_MARGIN = 0.10


def input_warnings(img):
    arr = np.asarray(img.resize((128, 128)), dtype=np.float32)
    colorfulness = float(
        np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2]))
    )
    warns = []
    if colorfulness > 12:
        warns.append("COLOR_IMAGE: NOT A TYPICAL X-RAY")
    if min(img.size) < 150:
        warns.append("LOW_RESOLUTION: RESULT UNRELIABLE")
    return warns


def inject_html(block: str) -> None:
    cleaned = "\n".join(line.strip() for line in block.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


st.set_page_config(
    page_title="CHESTSCAN AI | NEUROTRACE",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_html("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=Space+Mono:wght@400;700&family=Syncopate:wght@700&display=swap');

:root {
    --bg-color: #dcdcd9;
    --text-black: #0c0c0c;
    --accent-red: #ff2a2a;
    --border-dark: #222222;
    --white: #f5f5f5;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-color) !important;
    color: var(--text-black);
}

.stApp {
    background-image:
        linear-gradient(rgba(0,0,0,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,0,0,0.04) 1px, transparent 1px) !important;
    background-size: 80px 80px !important;
    background-position: center top !important;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 9999;
    opacity: 0.25;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
}

[data-testid="stSidebar"], header[data-testid="stHeader"], footer { display: none !important; }
.block-container { padding: 2rem 4rem; max-width: 1600px; position: relative; }

.block-container::before { content: '+'; position: fixed; top: 20px; left: 30px; font-size: 24px; color: var(--accent-red); font-family: 'Space Mono'; font-weight: bold; }
.block-container::after { content: '+'; position: fixed; top: 20px; right: 30px; font-size: 24px; color: var(--accent-red); font-family: 'Space Mono'; font-weight: bold; }

.top-nav { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid var(--border-dark); padding-bottom: 10px; }
.logo-pill { font-family: 'Syncopate', sans-serif; font-weight: 700; font-size: 20px; letter-spacing: 2px; display: flex; align-items: center; gap: 10px; }
.status-pill { font-family: 'Space Mono', monospace; font-weight: 700; font-size: 12px; display: flex; align-items: center; gap: 10px; border: 1px solid var(--border-dark); padding: 6px 16px; border-radius: 4px; }
.blink-dot { width: 10px; height: 10px; background-color: var(--accent-red); border-radius: 50%; animation: blink 1.5s infinite; }
@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0; } 100% { opacity: 1; } }

.hero-section { position: relative; margin-bottom: 40px; }
.giant-title { font-size: clamp(60px, 11vw, 150px); font-weight: 900; line-height: 0.8; letter-spacing: -0.06em; color: var(--text-black); margin: 0; text-transform: uppercase; }
.barcode { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px; position: absolute; right: 0; bottom: 10px; text-align: right; }
.barcode-lines { display: inline-block; width: 120px; height: 30px; background: repeating-linear-gradient(90deg, var(--text-black) 0, var(--text-black) 2px, transparent 2px, transparent 4px, var(--text-black) 4px, var(--text-black) 5px, transparent 5px, transparent 8px); }

.tech-box { border: 1px solid var(--border-dark); background: transparent; padding: 20px; position: relative; margin-bottom: 20px; }
.box-header { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; text-transform: uppercase; background: var(--text-black); color: var(--white); padding: 4px 12px; display: inline-block; position: absolute; top: -14px; left: 15px; letter-spacing: 1px; }

.st-key-input_box {
    position: relative;
    border: 1px solid var(--border-dark);
    padding: 26px 20px 20px !important;
    margin-top: 18px;
    margin-bottom: 20px;
}
.st-key-input_box::before {
    content: 'INPUT_MODULE';
    position: absolute; top: -14px; left: 15px;
    font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; letter-spacing: 1px;
    background: var(--text-black); color: var(--white); padding: 4px 12px;
}

.st-key-img_box {
    position: relative;
    border: 2px solid var(--border-dark);
    padding: 5px !important;
    background: var(--bg-color);
    overflow: hidden;
    margin-top: 5px;
}
.st-key-img_box::before {
    content: '';
    position: absolute; left: 0; width: 100%; height: 4px; top: 0;
    background: var(--accent-red);
    box-shadow: 0 0 15px var(--accent-red), 0 0 30px var(--accent-red);
    z-index: 5; pointer-events: none;
    animation: scan 2.5s infinite linear;
}
@keyframes scan { 0% { top: 0; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 98%; opacity: 0; } }
[data-testid="stImage"] img { filter: grayscale(100%) contrast(1.1); display: block; width: 100%; height: auto; }

.stElementContainer:has([data-testid="stImage"]) { position: relative; }
[data-testid="stElementToolbar"] {
    top: auto !important;
    bottom: 12px !important;
    right: 12px !important;
    z-index: 20 !important;
}
[data-testid="stElementToolbar"] button,
[data-testid="StyledFullScreenButton"] {
    background-color: rgba(12, 12, 12, 0.75) !important;
    border-radius: 0 !important;
    color: var(--white) !important;
}
[data-testid="stElementToolbar"] button *,
[data-testid="StyledFullScreenButton"] * { color: var(--white) !important; fill: var(--white) !important; }
[data-testid="StyledFullScreenButton"] { top: auto !important; bottom: 12px !important; right: 12px !important; z-index: 20 !important; }
[data-testid="stElementToolbar"] button:hover,
[data-testid="StyledFullScreenButton"]:hover { background-color: var(--accent-red) !important; }

.result-huge { font-size: 85px; font-weight: 900; line-height: 1; margin: 10px 0; letter-spacing: -2px; }
.result-label { font-family: 'Space Mono', monospace; font-size: 16px; font-weight: 700; letter-spacing: 3px; }
.alert-red { color: var(--accent-red); }
.ok-black { color: var(--text-black); }

[data-testid="stFileUploader"] { margin-top: 5px; }
[data-testid="stFileUploader"] section { background: transparent !important; border: 1px dashed var(--border-dark) !important; padding: 15px !important; margin-bottom: 0px !important; }
[data-testid="stFileUploader"] section * { color: var(--text-black) !important; opacity: 1 !important; visibility: visible !important; }
[data-testid="stFileUploader"] small { color: #222222 !important; font-weight: 600 !important; }
[data-testid="stFileUploader"] span { font-weight: 700 !important; }
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] section button *,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    color: var(--white) !important;
    -webkit-text-fill-color: var(--white) !important;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploaderDropzone"] button {
    background: var(--text-black) !important;
    font-family: 'Space Mono', monospace; font-size: 11px !important; border-radius: 0 !important;
    border: 1px solid var(--text-black) !important;
}
[data-testid="stFileUploader"] section button:hover,
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--accent-red) !important; border-color: var(--accent-red) !important; }

[data-testid="stSlider"] { padding-top: 5px; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--accent-red); font-weight: bold; margin-bottom: 5px; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent-red); }

.stButton:first-of-type > button { width: 100% !important; display: block; background: var(--accent-red); color: var(--white); border: 2px solid var(--accent-red); border-radius: 0; padding: 18px 20px; font-weight: 900; font-size: 18px; letter-spacing: 2px; text-transform: uppercase; transition: all 0.2s ease; margin-bottom: 20px; }
.stButton:first-of-type > button p { color: inherit; }
.stButton:first-of-type > button:hover { background: transparent; color: var(--accent-red); }

.terminal-log { font-family: 'Space Mono', monospace; font-size: 11px; color: var(--text-black); border-left: 2px solid var(--accent-red); padding-left: 15px; line-height: 1.8; margin-top: 15px; opacity: 0.9; }

.borderline { border: 2px dashed var(--accent-red) !important; }
.warn-line { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 700; color: var(--accent-red); margin-top: 8px; }

.class-bars { margin-top: 22px; }
.cb-row { margin-bottom: 14px; }
.cb-head { display:flex; justify-content:space-between; font-family:'Space Mono', monospace; font-size:12px; font-weight:700; margin-bottom:5px; }
.cb-track { position:relative; height:16px; border:1px solid var(--border-dark); background: rgba(0,0,0,0.06); }
.cb-fill { height:100%; }
.cb-fill.dark { background: var(--text-black); }
.cb-fill.red { background: var(--accent-red); }
.cb-thr { position:absolute; top:-5px; bottom:-5px; width:0; border-left:2px dashed var(--accent-red); }
.cb-thr em { position:absolute; top:-15px; left:4px; font-family:'Space Mono', monospace; font-size:9px; font-style:normal; font-weight:700; color:var(--accent-red); white-space:nowrap; }
.cb-pred { background: var(--text-black); color: var(--white); padding: 1px 8px; margin-left: 8px; font-size: 10px; }

[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"] small {
    color: var(--white) !important;
    -webkit-text-fill-color: var(--white) !important;
}
</style>
""")

inject_html("""
<div class="top-nav">
    <div class="logo-pill"><i class="fa-solid fa-lungs"></i> CHESTSCAN</div>
    <div class="status-pill">SYS. ACTIVE <div class="blink-dot"></div></div>
</div>
<div class="hero-section">
    <h1 class="giant-title">CHESTSCAN</h1>
    <div class="barcode">
        <div class="barcode-lines"></div><br>
        UID: 994-CX-8820
    </div>
    <div class="terminal-log" style="max-width: 500px; margin-top: 20px; border-left-color: var(--border-dark);">
        > INITIATING PROTOCOL_01<br>
        > AI-DRIVEN RADIOLOGY ANALYSIS ENGINE<br>
        > DETECTING PULMONARY OPACITIES WITH HIGH PRECISION.
    </div>
</div>
""")

col_in, col_out = st.columns([1.1, 1], gap="large")

with col_in:
    image = None

    with st.container(key="input_box"):
        uploaded_file = st.file_uploader(
            "Importer une radiographie",
            type=ALLOWED_TYPES,
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image.load()
                image = image.convert("RGB")
            except Exception as exc:
                image = None
                st.error(f"SYSTEM ERROR: {exc}")

        if image is not None:
            inject_html(f"""
<div style="font-family:'Space Mono'; font-size:10px; color:var(--text-black); padding-top:6px; opacity:0.8; margin-bottom: 5px;">
RES: {image.width}x{image.height}
</div>
""")
            with st.container(key="img_box"):
                st.image(image, use_container_width=True)

    inject_html(
        "<div style=\"font-family:'Space Mono'; font-size:11px; font-weight:bold; margin-bottom:5px;\">DETECTION_THRESHOLD</div>"
    )
    threshold = st.slider(
        "DETECTION_THRESHOLD",
        min_value=0.50,
        max_value=0.80,
        value=0.65,
        step=0.01,
        label_visibility="collapsed",
    )

with col_out:
    run_analysis = st.button("EXECUTE SCAN", disabled=(image is None))

    file_key = (
        (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
    )

    if run_analysis and image is not None:
        with st.spinner("EXTRACTING FEATURES..."):
            try:
                t0 = time.perf_counter()
                res = predict_image(image, threshold=threshold)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                st.session_state["scan"] = {
                    "key": file_key,
                    "prob": float(res["probability"]),
                    "ms": res.get("inference_ms", elapsed_ms),
                }
            except Exception as e:
                st.session_state.pop("scan", None)
                st.error(f"CRITICAL FAILURE: {e}")

    scan = st.session_state.get("scan")
    ms_txt = "—"
    if scan is not None and image is not None and scan["key"] == file_key:
        prob = scan["prob"]
        pred = "PNEUMONIA" if prob >= threshold else "NORMAL"
        is_pneu = pred == "PNEUMONIA"
        borderline = abs(prob - threshold) <= BORDER_MARGIN
        if scan["ms"] is not None:
            ms_txt = f"{scan['ms']:.0f} ms"

        text_color = "alert-red" if is_pneu else "ok-black"
        ico = "fa-triangle-exclamation" if is_pneu else "fa-check-double"
        border_color = "var(--accent-red)" if is_pneu else "var(--text-black)"

        if borderline:
            label = f"UNCERTAIN ZONE // LEANS {pred}"
            ico = "fa-circle-question"
            box_cls = "tech-box borderline"
            action = "> ACTION: BORDERLINE CASE, REVIEW MANUALLY<br>"
        else:
            label = pred
            box_cls = "tech-box"
            action = ""

        warn_html = "".join(
            f'<div class="warn-line">! {w}</div>' for w in input_warnings(image)
        )

        p_pneu = prob
        p_norm = 1.0 - prob
        conf = p_pneu if is_pneu else p_norm

        tag = '<span class="cb-pred">PREDICTED</span>'
        inject_html(f"""
            <div class="{box_cls}" style="border: 2px solid {border_color}; padding: 30px;">
                <div class="box-header" style="background:{border_color};">OUTPUT_RESULT</div>
                <div class="result-label {text_color}"><i class="fa-solid {ico}"></i> {label}</div>
                <div class="result-huge {text_color}">{conf * 100:.1f}%</div>
                <div class="result-label {text_color}" style="font-size:12px; letter-spacing:2px;">{pred} CONFIDENCE</div>
                <div class="class-bars">
                    <div class="cb-row">
                        <div class="cb-head"><span>NORMAL{tag if not is_pneu else ''}</span><span>{p_norm * 100:.1f}%</span></div>
                        <div class="cb-track"><div class="cb-fill dark" style="width:{p_norm * 100:.1f}%"></div></div>
                    </div>
                    <div class="cb-row">
                        <div class="cb-head"><span>PNEUMONIA{tag if is_pneu else ''}</span><span>{p_pneu * 100:.1f}%</span></div>
                        <div class="cb-track">
                            <div class="cb-fill red" style="width:{p_pneu * 100:.1f}%"></div>
                            <div class="cb-thr" style="left:{threshold * 100:.0f}%"><em>THR {threshold:.2f}</em></div>
                        </div>
                    </div>
                </div>
                <div class="terminal-log" style="border-left-color: {border_color}; margin-top: 24px;">
                    > NORMAL_SCORE: {p_norm:.4f}<br>
                    > PNEUMONIA_SCORE: {p_pneu:.4f}<br>
                    > THRESHOLD_APPLIED: {threshold:.2f} (PNEUMONIA IF SCORE >= THR)<br>
                    > DISTANCE_TO_THRESHOLD: {abs(prob - threshold):.3f}<br>
                    > INFERENCE_TIME: {ms_txt}<br>
                    {action}
                    > TARGET_CLASS: {pred}
                </div>
                {warn_html}
            </div>
            """)

    inject_html(f"""
        <div class="tech-box" style="margin-top: 20px; opacity: 0.9;">
            <div class="box-header">SYS_TELEMETRY</div>
            <div class="terminal-log" style="border:none; padding:0; margin:0; line-height:2;">
                <b>ARCHITECTURE :</b> {MODEL_NAME}<br>
                <b>INPUT_TENSOR :</b> {INPUT_SIZE}<br>
                <b>LOSS_FUNCTION:</b> Binary Cross-Entropy<br>
                <b>BASE_WEIGHTS :</b> ImageNet<br>
                <b>TRAINING_SET :</b> {DATASET_NAME}<br>
                <b>LAST_LATENCY :</b> {ms_txt}<br>
                <br>
                [ CPU MEMORY ] : STABLE<br>
                [ GPU STATUS ] : STANDBY
            </div>
        </div>
        """)

inject_html("""
    <div style="margin-top: 60px; padding-top: 15px; border-top: 1px solid var(--border-dark); display: flex; justify-content: space-between; font-family: 'Space Mono', monospace; font-size: 10px;">
        <span>// CHESTSCAN AI PROTOTYPE v2.0</span>
        <span>STRICTLY NOT FOR CLINICAL DIAGNOSIS</span>
    </div>
    """)
