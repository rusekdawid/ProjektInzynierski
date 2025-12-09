import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import random
import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config as cfg
from ai_model import SRResNet
import auto_detector
from predict import process_tiled

st.set_page_config(page_title="AI Image Restoration", page_icon="🤖", layout="wide")
st.markdown("""<style>.stButton>button { height: 3em; font-weight: bold; font-size: 1.1em; }</style>""", unsafe_allow_html=True)

# --- FUNKCJE ---
def load_model_safe(task_name):
    if 'model_cache' not in st.session_state: st.session_state['model_cache'] = {}
    if st.session_state.get('force_reload', False):
        if task_name in st.session_state['model_cache']: del st.session_state['model_cache'][task_name]
    
    if task_name in st.session_state['model_cache']: return st.session_state['model_cache'][task_name]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRResNet().to(device)
    model_path = cfg.MODELS_DIR / f'model_{task_name}.pth'
    if not model_path.exists(): return None
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.session_state['model_cache'][task_name] = model
        return model
    except: return None

def get_random_image():
    files = list(cfg.RAW_DIR.rglob('*.[jp][pn]*[g]'))
    if not files: return None
    return random.choice(files)

def apply_degradation(img, method, intensity):
    res = img.copy()
    if method == "Szum (Noise)":
        noise = np.random.normal(0, intensity, img.shape)
        res = np.clip(img + noise, 0, 255).astype(np.uint8)
    elif method == "Rozmycie (Blur)":
        k = int(intensity)
        if k % 2 == 0: k += 1
        res = cv2.GaussianBlur(img, (k, k), 0)
    elif method == "Niska Rozdzielczość (Low Res)":
        h, w = img.shape[:2]
        scale = int(intensity)
        # POWRÓT DO STANDARDU: Zmniejszamy i powiększamy CUBIC.
        # To daje obraz "miękki", który model potrafi wyostrzyć i podbić PSNR.
        small = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
        res = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return res

def calc_metrics(orig, degraded):
    try:
        h = min(orig.shape[0], degraded.shape[0])
        w = min(orig.shape[1], degraded.shape[1])
        o = orig[:h, :w]
        d = degraded[:h, :w]
        p = psnr(o, d, data_range=255)
        s = ssim(o, d, channel_axis=2, data_range=255)
        return p, s
    except: return 0.0, 0.0

# --- STAN ---
if 'original_image' not in st.session_state: st.session_state['original_image'] = None
if 'current_image' not in st.session_state: st.session_state['current_image'] = None
if 'processed_image' not in st.session_state: st.session_state['processed_image'] = None
if 'last_file_id' not in st.session_state: st.session_state['last_file_id'] = None
if 'force_reload' not in st.session_state: st.session_state['force_reload'] = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Panel Sterowania")
    source = st.radio("Tryb:", ["Wgraj plik", "Losuj z bazy"], label_visibility="collapsed")
    
    new_loaded = False
    if source == "Wgraj plik":
        uploaded = st.file_uploader("Wybierz zdjęcie", type=['jpg', 'png', 'jpeg'])
        if uploaded:
            if st.session_state['last_file_id'] != uploaded.file_id:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                st.session_state['original_image'] = img
                st.session_state['current_image'] = img.copy()
                st.session_state['processed_image'] = None
                st.session_state['last_file_id'] = uploaded.file_id
                new_loaded = True
    else:
        if st.button("🎲 Wylosuj nowe zdjęcie", use_container_width=True):
            f = get_random_image()
            if f:
                img = cv2.imread(str(f))
                st.session_state['original_image'] = img
                st.session_state['current_image'] = img.copy()
                st.session_state['processed_image'] = None
                st.session_state['last_file_id'] = f"rnd_{random.randint(0, 1e9)}"
                new_loaded = True
                st.success(f"Plik: {f.name}")
    
    if new_loaded: st.toast("Załadowano!")
    st.divider()
    if st.button("🔄 Przeładuj Modele"):
        st.session_state['force_reload'] = True
        st.toast("Odświeżono!")

# --- GŁÓWNE OKNO ---
st.title("✨ Automatyczny System Poprawy Jakości (AI)")

if st.session_state['original_image'] is None:
    st.info("👈 Wczytaj zdjęcie.")
    st.stop()

tab_destroy, tab_fix = st.tabs(["💥 1. Symulator Zniszczeń", "🚀 2. Automatyczna Naprawa"])

# ZAKŁADKA 1
with tab_destroy:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Parametry")
        method = st.selectbox("Rodzaj", ["Szum (Noise)", "Rozmycie (Blur)", "Niska Rozdzielczość (Low Res)"])
        
        intensity = 0
        if method == "Szum (Noise)": 
            intensity = st.slider("Poziom", 0, 100, 40)
        elif method == "Rozmycie (Blur)": 
            intensity = st.slider("Siła", 1, 31, 15, step=2)
        elif method == "Niska Rozdzielczość (Low Res)": 
            # Skala 4 jest optymalna dla Twojego modelu
            intensity = st.slider("Skala", 2, 8, 4)
        
        if st.button("⚡ ZASTOSUJ", type="primary", use_container_width=True):
            orig = st.session_state['original_image']
            st.session_state['current_image'] = apply_degradation(orig, method, intensity)
            st.session_state['processed_image'] = None
            st.toast("Zniszczono!")
            st.rerun()
        
        if st.button("↩️ RESET", use_container_width=True):
            st.session_state['current_image'] = st.session_state['original_image'].copy()
            st.session_state['processed_image'] = None
            st.rerun()

    with c2:
        st.image(cv2.cvtColor(st.session_state['current_image'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Podgląd")

# ZAKŁADKA 2
with tab_fix:
    img_in = st.session_state['current_image']
    cv2.imwrite("temp_detect.png", img_in)
    detected, info = auto_detector.detect_distortion("temp_detect.png")
    
    with st.container():
        c_a, c_b, c_c = st.columns([1, 2, 1])
        with c_a:
            st.metric("Wykryto", detected.upper())
        with c_b:
            st.info(f"{info}")
        with c_c:
            st.write("")
            if st.button("✨ URUCHOM AI", type="primary", use_container_width=True):
                if st.session_state['force_reload']: st.session_state['force_reload'] = False
                model = load_model_safe(detected)
                if model:
                    with st.spinner("Przetwarzanie..."):
                        try:
                            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            inp = img_in.copy()
                            
                            # Teraz po prostu wpuszczamy obraz do sieci.
                            # Jest on już "miękki" (Bicubic), więc model go zrozumie.
                            res = process_tiled(model, inp, dev, tile_size=256, overlap=16)
                            
                            st.session_state['processed_image'] = res
                            st.success("Gotowe!")
                        except Exception as e: st.error(f"Błąd: {e}")
                else: st.error(f"Brak modelu: {detected}")

    st.divider()
    if st.session_state['processed_image'] is not None:
        res = st.session_state['processed_image']
        orig = st.session_state['original_image']
        
        p_in, s_in = calc_metrics(orig, img_in)
        p_out, s_out = calc_metrics(orig, res)
        d_p = p_out - p_in
        d_s = s_out - s_in
        
        cc1, cc2 = st.columns(2)
        with cc1: 
            st.image(cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Wejście")
            m1, m2 = st.columns(2)
            m1.metric("PSNR", f"{p_in:.2f} dB")
            m2.metric("SSIM", f"{s_in:.4f}")
        with cc2: 
            st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Wyjście (AI)")
            m1, m2 = st.columns(2)
            m1.metric("PSNR", f"{p_out:.2f} dB", delta=f"{d_p:+.2f} dB")
            m2.metric("SSIM", f"{s_out:.4f}", delta=f"{d_s:+.4f}")
            
        st.divider()
        st.subheader("🔍 Zoom 1:1")
        h, w = res.shape[:2]
        cy, cx = h//2, w//2
        S = 120
        y1, y2 = max(0, cy-S), min(h, cy+S)
        x1, x2 = max(0, cx-S), min(w, cx+S)
        
        # Używamy INTER_CUBIC do wyświetlania zoomu, żeby było ładnie i gładko
        crop_in = cv2.resize(img_in[y1:y2, x1:x2], (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crop_out = cv2.resize(res[y1:y2, x1:x2], (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        z1, z2 = st.columns(2)
        with z1: st.image(cv2.cvtColor(crop_in, cv2.COLOR_BGR2RGB), caption="Zoom: Wejście")
        with z2: st.image(cv2.cvtColor(crop_out, cv2.COLOR_BGR2RGB), caption="Zoom: Wyjście")