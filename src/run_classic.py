import cv2
import shutil
from tqdm import tqdm
import config as cfg

def run_classic_all():
    print("\nMETODY KLASYCZNE (Generowanie i czyszczenie)...")
    
    # Definicja zadań: (Folder wejściowy, Folder wyjściowy, Funkcja przetwarzająca)
    tasks = {
        'noise': ('denoised', lambda i: cv2.GaussianBlur(i, (5,5), 0)),
        'blur': ('sharpened', lambda i: cv2.addWeighted(i, 1.5, cv2.GaussianBlur(i, (9,9), 10), -0.5, 0, i)),
        'low_res': ('low_res', lambda i: cv2.resize(i, (i.shape[1]*cfg.SCALE_FACTOR, i.shape[0]*cfg.SCALE_FACTOR), interpolation=cv2.INTER_LANCZOS4))
    }
    
    for inp_n, (out_n, func) in tasks.items():
        # Ścieżka do wyników
        out_dir = cfg.RESULTS_DIR / 'classic' / out_n
        
        # --- CZYSZCZENIE ---
        if out_dir.exists():
            shutil.rmtree(out_dir) # Usuwa stary folder z całą zawartością
        out_dir.mkdir(parents=True, exist_ok=True) # Tworzy nowy, pusty
        
        # Pobieranie plików wejściowych
        files = list((cfg.PROCESSED_DIR / inp_n).glob('*'))
        
        if not files:
            print(f"Pusto w {inp_n}, pomijam.")
            continue

        for f in tqdm(files, desc=f"Classic {inp_n}"):
            img = cv2.imread(str(f))
            if img is None: continue
            
            # Przetwarzanie i zapis
            res = func(img)
            cv2.imwrite(str(out_dir / f.name), res)