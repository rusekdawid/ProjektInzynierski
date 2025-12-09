import cv2, shutil
import numpy as np
from tqdm import tqdm
import config as cfg

def generate_all_datasets():
    print("🏭 GENEROWANIE DANYCH...")
    # Pobieramy listę plików
    files = (list(cfg.RAW_DIR.rglob('*.[jJ][pP][gG]')) + list(cfg.RAW_DIR.rglob('*.[pP][nN][gG]')))
    files = files[:cfg.NUM_IMAGES] # Bierzemy tylko pierwsze 300
    
    if not files:
        print("❌ Błąd: Brak zdjęć w folderze data/raw!")
        return

    for t in ['noise', 'blur', 'low_res']:
        d = cfg.PROCESSED_DIR / t
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
        for f in tqdm(files, desc=t):
            img = cv2.imread(str(f))
            if img is None: continue
            
            # Noise
            if t=='noise': 
                noise = np.random.normal(0, cfg.NOISE_LEVEL, img.shape)
                res = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Blur
            elif t=='blur': 
                res = cv2.GaussianBlur(img, (cfg.BLUR_KERNEL, cfg.BLUR_KERNEL), 0)
            
            # Low Res (Zapisujemy małe zdjęcie!)
            elif t=='low_res': 
                h, w = img.shape[:2]
                # Upewniamy się, że wymiary są parzyste dla łatwiejszego skalowania
                h, w = (h // 2) * 2, (w // 2) * 2
                img = img[:h, :w]
                small = cv2.resize(img, (w//cfg.SCALE_FACTOR, h//cfg.SCALE_FACTOR), interpolation=cv2.INTER_LINEAR)
                res = small # Zapisujemy małe
                
            cv2.imwrite(str(d / f.name), res)