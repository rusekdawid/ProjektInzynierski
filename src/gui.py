import sys
import os

# --- FIX ŚCIEŻEK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# -------------------

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
# --- ZMIANA: Importujemy rygorystyczną metrykę ---
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    from smart_system import SmartSystem
except ImportError:
    st.error("CRITICAL ERROR: Brak pliku smart_system.py!")
    st.stop()

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="AI Image Lab", layout="wide", page_icon="🔬")

# --- CSS ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .stButton>button {
        width: 100%; border-radius: 8px; font-weight: bold; font-size: 18px; 
        background-color: #4F8BF9; color: white; height: 50px;
    }
    .metric-value {font-size: 1.2rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- FUNKCJE POMOCNICZE ---
def calculate_psnr(img1, img2):
    """
    Oblicza PSNR używając standardu scikit-image (tak jak w evaluate.py).
    """
    # Wyrównanie wymiarów (na wszelki wypadek)
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    # Używamy data_range=255, bo operujemy na uint8
    return psnr(img1, img2, data_range=255)

def apply_degradation(img, mode, val):
    if mode == "Brak": return img
    
    if mode == "Szum":
        if val == 0: return img
        row, col, ch = img.shape
        # UWAGA: Generujemy szum losowo. 
        # Każde odświeżenie strony da troszkę inny rozkład szumu,
        # więc PSNR może się wahać o +/- 0.1 dB względem testów statycznych.
        gauss = np.random.normal(0, val, (row, col, ch))
        return np.clip(img + gauss, 0, 255).astype(np.uint8)
        
    elif mode == "Blur":
        k = val if val % 2 != 0 else val + 1
        return cv2.GaussianBlur(img, (k, k), 0)
        
    elif mode == "LowRes":
        if val <= 1: return img
        h, w = img.shape[:2]
        small = cv2.resize(img, (w//val, h//val), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return img

# --- ŁADOWANIE SYSTEMU ---
@st.cache_resource
def get_system():
    return SmartSystem()

system = get_system()

# ==========================================
# GŁÓWNY INTERFEJS
# ==========================================

st.markdown('<p class="main-header"> Laboratorium Naprawy Obrazu</p>', unsafe_allow_html=True)

# --- PANEL BOCZNY ---
with st.sidebar:
    st.header("1. Wczytaj")
    uploaded_file = st.file_uploader("Wybierz plik", type=['jpg', 'png', 'jpeg'])
    
    st.markdown("---")
    st.header("2. Degradacja")
    deg_mode = st.selectbox("Uszkodzenie:", ["Brak", "Szum", "Blur", "LowRes"])
    
    val = 0
    if deg_mode == "Szum": val = st.slider("Poziom", 0, 40, 20)
    elif deg_mode == "Blur": val = st.slider("Poziom", 3, 15, 7, step=2)
    elif deg_mode == "LowRes": val = st.slider("Skala", 2, 8, 4)

# --- LOGIKA ---
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    
    # Degradacja
    processed_img = apply_degradation(original_img, deg_mode, val)
    
    # Oblicz PSNR dla zepsutego (Metoda Rygorystyczna)
    psnr_bad = calculate_psnr(original_img, processed_img)

    # --- UKŁAD KOLUMN ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Obraz Zdegradowany")
        st.image(processed_img, use_container_width=True)
        # Metryka
        st.info(f" Jakość (PSNR): **{psnr_bad:.2f} dB**")
        
        st.write("---")
        st.write("**Panel Diagnostyczny:**")
        img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        detected = system.detect_problem(img_bgr)
        
        if detected == 'clean': st.success("System widzi: **CZYSTY**")
        elif detected == 'noise': st.error("System widzi: **SZUM**")
        elif detected == 'blur': st.warning("System widzi: **ROZMYCIE**")
        elif detected == 'low_res': st.warning("System widzi: **PIKSELOZĘ**")

    with col2:
        st.subheader(" Wynik Naprawy")
        result_placeholder = st.empty()
        metric_placeholder = st.empty()
        
        
        
        if st.button(" URUCHOM AI"):
            with st.spinner("Przetwarzanie..."):
                start_t = time.time()
                
                mode_map = 'Auto'
                
                
                res_img, msg, used_mode = system.process_image(processed_img, mode_map)
                
                # Wyświetlenie
                result_placeholder.image(res_img, use_container_width=True)
                
                # Oblicz PSNR dla naprawionego (Metoda Rygorystyczna)
                psnr_good = calculate_psnr(original_img, res_img)
                gain = psnr_good - psnr_bad
                
                # Metryka z Deltą (Zielona strzałka)
                metric_placeholder.metric(label=" Jakość (PSNR) po naprawie", 
                                          value=f"{psnr_good:.2f} dB", 
                                          delta=f"{gain:+.2f} dB")
        else:
            result_placeholder.info("Kliknij przycisk, aby naprawić.")

    # --- ZOOM ---
    st.markdown("---")
    st.subheader(" Inspekcja Detali (Zoom)")
    
    if 'res_img' in locals():
        h, w, _ = original_img.shape
        center_y, center_x = h // 2, w // 2
        crop_size = 100
        y1 = max(0, center_y - crop_size)
        y2 = min(h, center_y + crop_size)
        x1 = max(0, center_x - crop_size)
        x2 = min(w, center_x + crop_size)
        
        crop_bad = processed_img[y1:y2, x1:x2]
        crop_good = res_img[y1:y2, x1:x2]
        
        zoom_factor = 4
        crop_bad_zoom = cv2.resize(crop_bad, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
        crop_good_zoom = cv2.resize(crop_good, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
        
        z1, z2 = st.columns(2)
        with z1: st.image(crop_bad_zoom, caption="Zbliżenie: Przed", use_container_width=True)
        with z2: st.image(crop_good_zoom, caption="Zbliżenie: Po", use_container_width=True)
            
    else:
        st.caption("Napraw zdjęcie, aby zobaczyć porównanie detali.")

else:
    st.info(" Wgraj zdjęcie z panelu po lewej.")