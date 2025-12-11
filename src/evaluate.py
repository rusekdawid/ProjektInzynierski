import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import config as cfg

def load_raw_files_map():
    """Tworzy mapę nazw plików do ich pełnych ścieżek w folderze raw."""
    raw_map = {}
    # Szukamy we wszystkich podfolderach raw
    for f in cfg.RAW_DIR.rglob('*'):
        if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            raw_map[f.name] = f
    return raw_map

def calculate_metrics(method_type, task_name, raw_map):
    """
    Liczy PSNR i SSIM dla danej metody i zadania.
    method_type: 'classic' lub 'ai'
    task_name: 'noise', 'blur', 'low_res'
    """
    # Mapowanie nazw folderów (klasyka ma specyficzne nazwy w run_classic.py)
    folder_map = {
        'noise': 'denoised' if method_type == 'classic' else 'noise',
        'blur': 'sharpened' if method_type == 'classic' else 'blur',
        'low_res': 'low_res'
    }
    
    folder_name = folder_map.get(task_name, task_name)
    path_results = cfg.RESULTS_DIR / method_type / folder_name
    
    if not path_results.exists():
        return None, None

    files = list(path_results.glob('*'))
    if not files:
        return None, None

    psnr_list = []
    ssim_list = []

    print(f"   [{method_type.upper()}] Liczenie dla: {task_name} ({len(files)} plików)...")

    for f in tqdm(files, leave=False):
        if f.name not in raw_map:
            continue
        
        # Wczytanie obrazów
        img_res = cv2.imread(str(f))
        img_orig = cv2.imread(str(raw_map[f.name]))
        
        if img_res is None or img_orig is None:
            continue

        # --- Ujednolicenie wymiarów ---
        # AI (przez kafelkowanie) lub klasyka mogą dać minimalnie inny rozmiar.
        # Przycinamy do mniejszego wspólnego mianownika.
        h = min(img_res.shape[0], img_orig.shape[0])
        w = min(img_res.shape[1], img_orig.shape[1])
        
        img_res = img_res[:h, :w]
        img_orig = img_orig[:h, :w]
        
        # Obliczenie metryk
        try:
            # PSNR
            p = psnr(img_orig, img_res, data_range=255)
            # SSIM (wymaga określenia kanałów dla obrazów kolorowych)
            s = ssim(img_orig, img_res, channel_axis=2, win_size=3, data_range=255)
            
            psnr_list.append(p)
            ssim_list.append(s)
        except Exception as e:
            pass

    if not psnr_list:
        return 0, 0

    return np.mean(psnr_list), np.mean(ssim_list)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("RAPORT KOŃCOWY (PSNR / SSIM)")
    print("="*50)
    
    raw_map = load_raw_files_map()
    tasks = ['noise', 'blur', 'low_res']
    results = {}

    print(f"{'ZADANIE':<10} | {'METODA':<10} | {'PSNR (dB)':<10} | {'SSIM':<10}")
    print("-" * 50)

    for task in tasks:
        # 1. Klasyka
        p_c, s_c = calculate_metrics('classic', task, raw_map)
        
        # 2. AI
        p_ai, s_ai = calculate_metrics('ai', task, raw_map)
        
        # Wyświetlanie wyników w tabeli
        if p_c is not None:
            print(f"{task:<10} | {'Classic':<10} | {p_c:.2f}       | {s_c:.4f}")
        else:
            print(f"{task:<10} | {'Classic':<10} | {'BRAK':<10} | {'BRAK':<10}")
            
        if p_ai is not None:
            print(f"{task:<10} | {'AI (Ty)':<10} | {p_ai:.2f}       | {s_ai:.4f}")
        else:
            print(f"{task:<10} | {'AI (Ty)':<10} | {'BRAK':<10} | {'BRAK':<10}")
            
        print("-" * 50)
        
        # Zapis do słownika dla JSON
        results[task] = {
            'classic': {'psnr': round(p_c, 2) if p_c else 0, 'ssim': round(s_c, 4) if s_c else 0},
            'ai': {'psnr': round(p_ai, 2) if p_ai else 0, 'ssim': round(s_ai, 4) if s_ai else 0}
        }

    # Zapis do pliku
    json_path = cfg.RESULTS_DIR / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nZapisano szczegółowy raport do: {json_path}")