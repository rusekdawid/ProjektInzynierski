import cv2
import shutil
import numpy as np
from tqdm import tqdm
import config as cfg
from pathlib import Path

def generate_all_datasets():
    print(" GENEROWANIE DANYCH (SZTYWNE PARAMETRY Z CONFIG.PY)...")
    print(f"Noise Level: {cfg.NOISE_LEVEL}")
    print(f"Blur Kernel: {cfg.BLUR_KERNEL}")
    print(f"Scale Factor: {cfg.SCALE_FACTOR}")
    
    # 1. Pobieranie plików
    files = list(cfg.RAW_DIR.rglob('*.[jJ][pP][gG]')) + \
            list(cfg.RAW_DIR.rglob('*.[pP][nN][gG]')) + \
            list(cfg.RAW_DIR.rglob('*.[jJ][pP][eE][gG]'))
    files = files[:cfg.NUM_IMAGES]
    
    if not files:
        print("Błąd: Brak zdjęć w folderze data/raw!")
        return

    # 2. Czyszczenie folderów
    for t in ['noise', 'blur', 'low_res']:
        d = cfg.PROCESSED_DIR / t
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 3. Przetwarzanie
    for f in tqdm(files, desc="Generowanie"):
        img = cv2.imread(str(f))
        if img is None: continue
        
        # --- 1. NOISE (Sztywno z configu) ---
        noise = np.random.normal(0, cfg.NOISE_LEVEL, img.shape)
        res_noise = np.clip(img + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(cfg.PROCESSED_DIR / 'noise' / f.name), res_noise)
        
        # --- 2. BLUR (Sztywno z configu) ---
        # Upewniamy się, że kernel jest nieparzysty (wymóg OpenCV)
        k = int(cfg.BLUR_KERNEL)
        if k % 2 == 0: k += 1 
        
        res_blur = cv2.GaussianBlur(img, (k, k), 0)
        cv2.imwrite(str(cfg.PROCESSED_DIR / 'blur' / f.name), res_blur)
        
        # --- 3. LOW RES (Sztywno z configu + Małe pliki) ---
        h, w = img.shape[:2]
        scale = int(cfg.SCALE_FACTOR)
        
        # Obliczamy nowe wymiary
        new_w = int(w / scale)
        new_h = int(h / scale)
        
        # Zmniejszamy (Linear - standard)
        small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # ZAPISUJEMY MAŁE ZDJĘCIE
        cv2.imwrite(str(cfg.PROCESSED_DIR / 'low_res' / f.name), small)

    print("\n✅ Gotowe! Dane wygenerowane ściśle według config.py.")

if __name__ == "__main__":
    generate_all_datasets()