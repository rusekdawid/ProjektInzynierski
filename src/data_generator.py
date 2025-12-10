import cv2
import numpy as np
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import config as cfg

def clean_and_create_dirs():
    """
    Agresywne czyszczenie: Usuwa cały folder 'processed' i tworzy go od nowa.
    """
    print(f"🧹 Czyszczenie folderu: {cfg.PROCESSED_DIR} ...")
    
    if cfg.PROCESSED_DIR.exists():
        try:
            shutil.rmtree(cfg.PROCESSED_DIR)
        except Exception as e:
            print(f"   ❌ Błąd usuwania: {e}")
            return False

    tasks = ['noise', 'blur', 'low_res']
    for task in tasks:
        path = cfg.PROCESSED_DIR / task
        path.mkdir(parents=True, exist_ok=True)
        
    return True

def add_noise(img):
    noise = np.random.normal(0, cfg.NOISE_LEVEL, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_blur(img):
    return cv2.GaussianBlur(img, (cfg.BLUR_KERNEL, cfg.BLUR_KERNEL), 0)

def make_low_res(img):
    """
    Low-res o TYM SAMYM rozmiarze co oryginał:
    zmniejszamy, potem powiększamy z powrotem.
    """
    h, w = img.shape[:2]
    scale = cfg.SCALE_FACTOR

    # zmniejszenie
    small = cv2.resize(
        img,
        (max(1, w // scale), max(1, h // scale)),
        interpolation=cv2.INTER_AREA
    )

    # powrót do dokładnie tego samego rozmiaru
    low_res = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST   # lub INTER_LINEAR
    )

    return low_res

def generate_all_datasets():
    if not clean_and_create_dirs():
        return

    # 1. Zbieranie plików
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    files = []
    for ext in extensions:
        files.extend(list(cfg.RAW_DIR.rglob(ext)))
    
    files = sorted(list(set(files)))

    if not files:
        print(f"⚠️  Brak zdjęć w {cfg.RAW_DIR}!")
        return

    # 2. LIMITOWANIE LICZBY ZDJĘĆ (NUM_IMAGES)
    total_files = len(files)
    limit = cfg.NUM_IMAGES

    random.shuffle(files)

    if total_files > limit:
        print(f"ℹ️  Znaleziono {total_files} zdjęć, ale limit w configu to {limit}.")
        print(f"   ✂️  Przycinam listę do {limit} sztuk.")
        files = files[:limit]
    else:
        print(f"ℹ️  Znaleziono {total_files} zdjęć (Limit w configu: {limit}).")
        print("   Przetwarzam wszystkie dostępne.")

    # 3. Przetwarzanie
    print(f"📸 Rozpoczynam generowanie dla {len(files)} plików...")

    for f in tqdm(files, desc="Generowanie"):
        img = cv2.imread(str(f))
        if img is None: continue
        
        cv2.imwrite(str(cfg.PROCESSED_DIR / 'noise' / f.name), add_noise(img))
        cv2.imwrite(str(cfg.PROCESSED_DIR / 'blur' / f.name), add_blur(img))
        cv2.imwrite(str(cfg.PROCESSED_DIR / 'low_res' / f.name), make_low_res(img))

    print("\n✅ Zakończono!")
    print(f"   Wygenerowano po {len(files)} zdjęć w każdym folderze (noise/blur/low_res).")

if __name__ == "__main__":
    generate_all_datasets()
