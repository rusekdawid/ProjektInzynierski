import cv2
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
import config as cfg

def add_noise(img):
    row, col, ch = img.shape
    gauss = np.random.normal(0, cfg.NOISE_LEVEL, (row, col, ch))
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_blur(img):
    k = cfg.BLUR_KERNEL
    return cv2.GaussianBlur(img, (k, k), 0)

def add_low_res(img):
    h, w = img.shape[:2]
    # Zmniejszamy (utrata informacji)
    small = cv2.resize(img, (w//cfg.SCALE_FACTOR, h//cfg.SCALE_FACTOR), interpolation=cv2.INTER_LINEAR)
    # Powiększamy metodą "najbliższy sąsiad" (efekt pikselozy)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def generate_all_datasets():
    # Lista zadań do wykonania
    tasks = ['noise', 'blur', 'low_res']
    
    # 1. Pobieramy listę zdjęć źródłowych (raz dla wszystkich)
    raw_files = list(cfg.RAW_DIR.rglob('*.[jJ][pP][gG]')) + list(cfg.RAW_DIR.rglob('*.[pP][nN][gG]'))
    raw_files = raw_files[:cfg.NUM_IMAGES]
    
    print(f"🚀 ROZPOCZYNAM GENEROWANIE DANYCH (Liczba zdjęć: {len(raw_files)})")
    
    for task in tasks:
        print(f"\n--- Przetwarzanie: {task.upper()} ---")
        
        # Przygotowanie folderu (czyścimy stary, robimy nowy)
        target_dir = cfg.PROCESSED_DIR / task
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Pętla po zdjęciach
        for f in tqdm(raw_files, desc=f"Generowanie {task}"):
            img = cv2.imread(str(f))
            if img is None: continue
            
            # Wybór metody psucia
            if task == 'noise':
                result = add_noise(img)
            elif task == 'blur':
                result = add_blur(img)
            elif task == 'low_res':
                result = add_low_res(img)
            
            # Zapis
            cv2.imwrite(str(target_dir / f.name), result)

    print("\n✅ WSZYSTKIE DANE WYGENEROWANE POMYŚLNIE!")
    print(f"Lokalizacja: {cfg.PROCESSED_DIR}")

if __name__ == "__main__":
    generate_all_datasets()