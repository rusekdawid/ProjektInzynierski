import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import config as cfg
from pathlib import Path
from tqdm import tqdm

def load_raw_files_map():
    print("⏳ Indeksowanie oryginałów...")
    raw_map = {}
    for f in cfg.RAW_DIR.rglob('*'):
        if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            raw_map[f.name] = f
    return raw_map

def get_metrics(img1, img2):
    """Pomocnicza funkcja licząca PSNR i SSIM"""
    if img1 is None or img2 is None: return 0, 0
    
    # Wyrównanie wymiarów
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    try:
        p = psnr(img1, img2, data_range=255)
        s = ssim(img1, img2, channel_axis=2, win_size=3, data_range=255)
        return p, s
    except:
        return 0, 0

def evaluate_scenario(task_name, ai_folder, classic_folder, level_info, raw_map):
    # Ścieżki
    dir_degraded = cfg.PROCESSED_DIR / task_name
    dir_classic = cfg.RESULTS_DIR / 'classic' / classic_folder
    dir_ai = cfg.RESULTS_DIR / 'ai' / ai_folder
    
    files = list(dir_degraded.glob('*'))
    if not files:
        print(f"⚠️ Brak plików w {dir_degraded}")
        return

    # Listy na wyniki
    deg_psnr, deg_ssim = [], []
    cla_psnr, cla_ssim = [], []
    ai_psnr, ai_ssim = [], []

    print(f"\n🔍 Analiza zadania: {task_name.upper()} (Poziom: {level_info})...")

    for f in tqdm(files, leave=False):
        if f.name not in raw_map: continue
        
        # Wczytujemy 4 obrazy: Oryginał, Zepsuty, Klasyka, AI
        img_orig = cv2.imread(str(raw_map[f.name]))
        img_deg = cv2.imread(str(f))
        img_cla = cv2.imread(str(dir_classic / f.name))
        img_ai = cv2.imread(str(dir_ai / f.name))

        # 1. ZDEGRADOWANY vs ORYGINAŁ
        p, s = get_metrics(img_orig, img_deg)
        deg_psnr.append(p); deg_ssim.append(s)

        # 2. KLASYKA vs ORYGINAŁ
        p, s = get_metrics(img_orig, img_cla)
        cla_psnr.append(p); cla_ssim.append(s)

        # 3. AI vs ORYGINAŁ
        p, s = get_metrics(img_orig, img_ai)
        ai_psnr.append(p); ai_ssim.append(s)

    # Średnie
    avg_deg_p, avg_deg_s = np.mean(deg_psnr), np.mean(deg_ssim)
    avg_cla_p, avg_cla_s = np.mean(cla_psnr), np.mean(cla_ssim)
    avg_ai_p, avg_ai_s = np.mean(ai_psnr), np.mean(ai_ssim)

    # --- WYDRUK GOTOWY DO TABELI ---
    print("\n" + "="*80)
    print(f"WYNIKI DLA TABELI ({task_name.upper()} - {level_info})")
    print("="*80)
    print(f"{'Rodzaj':<15} | {'Poziom':<10} | {'PSNR(Zły)':<10} | {'SSIM(Zły)':<10} | {'PSNR(Klas)':<10} | {'SSIM(Klas)':<10} | {'PSNR(AI)':<10} | {'SSIM(AI)':<10}")
    print("-" * 110)
    print(f"{task_name:<15} | {str(level_info):<10} | {avg_deg_p:.2f} dB    | {avg_deg_s:.4f}     | {avg_cla_p:.2f} dB    | {avg_cla_s:.4f}     | {avg_ai_p:.2f} dB    | {avg_ai_s:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    raw_map = load_raw_files_map()

    # KONFIGURACJA ZADAŃ
    # (Nazwa folderu processed, folder AI, folder Classic, Opis poziomu z Configa)
    
    # 1. SZUM
    evaluate_scenario('noise', 'noise', 'denoised', cfg.NOISE_LEVEL, raw_map)
    
    # 2. BLUR
    evaluate_scenario('blur', 'blur', 'sharpened', cfg.BLUR_KERNEL, raw_map)
    
    # 3. LOW RES
    evaluate_scenario('low_res', 'low_res', 'low_res', f"x{cfg.SCALE_FACTOR}", raw_map)