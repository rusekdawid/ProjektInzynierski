import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Prosty Niszczyciel Zdjęć", page_icon="💥", layout="wide")

# --- FUNKCJE NISZCZĄCE (Czysty OpenCV) ---

def apply_noise(img, intensity):
    """
    Dodaje szum. Intensity (0-100) to odchylenie standardowe.
    """
    if intensity == 0: return img
    noise = np.random.normal(0, intensity, img.shape)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img

def apply_blur(img, intensity):
    """
    Dodaje rozmycie. Intensity (1-30) to wielkość plamki.
    """
    k = int(intensity)
    # Kernel musi być nieparzysty (np. 3, 5, 7...)
    if k % 2 == 0: k += 1
    if k < 1: k = 1
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_low_res(img, scale):
    """
    Symuluje pikselozę. Scale (2-16) to krotność pomniejszenia.
    """
    if scale <= 1: return img
    h, w = img.shape[:2]
    
    # 1. Zmniejszamy (tracimy dane)
    small = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
    
    # 2. Powiększamy NEAREST (żebyś widział kwadraty na ekranie)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

# --- INTERFEJS ---

st.title("Prosty Symulator Zniszczeń")
st.markdown("Narzędzie do generowania uszkodzonych obrazów w celu testowania algorytmów naprawczych.")

# 1. Wczytywanie
uploaded_file = st.file_uploader("Wgraj zdjęcie (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Konwersja pliku na obraz OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1) # BGR
    
    # Konwersja na RGB do wyświetlania
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 2. Panel Sterowania
    with st.sidebar:
        st.header("Ustawienia")
        method = st.radio("Wybierz metodę:", ["Szum (Noise)", "Rozmycie (Blur)", "Pikseloza (Low Res)"])
        
        intensity = 0
        processed_img = original_img.copy()

        if method == "Szum (Noise)":
            intensity = st.slider("Poziom szumu", 0, 100, 30)
            processed_img = apply_noise(original_img, intensity)
            
        elif method == "Rozmycie (Blur)":
            intensity = st.slider("Siła rozmycia", 1, 31, 15)
            processed_img = apply_blur(original_img, intensity)
            
        elif method == "Pikseloza (Low Res)":
            intensity = st.slider("Skala pikseli", 2, 16, 6)
            processed_img = apply_low_res(original_img, intensity)

    # 3. Wyświetlanie (Dwie kolumny)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Oryginał")
        st.image(original_rgb, use_container_width=True)
        st.caption(f"Rozmiar: {original_img.shape[1]}x{original_img.shape[0]}")

    with col2:
        st.subheader("Po zniszczeniu")
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.image(processed_rgb, use_container_width=True)
        st.caption(f"Efekt: {method} | Siła: {intensity}")

    # 4. Pobieranie
    st.divider()
    res_pil = Image.fromarray(processed_rgb)
    buf = io.BytesIO()
    res_pil.save(buf, format="PNG")
    
    st.download_button(
        label="Pobierz zniszczone zdjęcie",
        data=buf.getvalue(),
        file_name=f"zniszczone_{method}.png",
        mime="image/png"
    )

else:
    st.info("Wgraj zdjęcie, aby rozpocząć.")