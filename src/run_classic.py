import cv2
from pathlib import Path
from tqdm import tqdm
# Importujemy Twoje narzędzia
from classic_methods import ClassicEnhancer

def run_classic_pipeline():
    # 1. Ustalanie ścieżek
    base_dir = Path('data')
    input_dir = base_dir / 'processed'       # Tu są zepsute
    output_dir = base_dir / 'results' / 'classic' # Tu zapiszemy naprawione
    
    # Tworzenie folderów wynikowych
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'denoised').mkdir(exist_ok=True)
    (output_dir / 'sharpened').mkdir(exist_ok=True)
    (output_dir / 'upscaled').mkdir(exist_ok=True)

    enhancer = ClassicEnhancer()

    # --- 1. ODSZUMIANIE (Noise) ---
    noise_path = input_dir / 'noise'
    files = list(noise_path.glob('*'))
    if files:
        print(f"Odszumianie {len(files)} zdjęć...")
        for file_path in tqdm(files):
            img = cv2.imread(str(file_path))
            if img is None: continue
            
            # Używamy metody NLM (najlepsza klasyczna)
            result = enhancer.denoising(img, method='nlm')
            cv2.imwrite(str(output_dir / 'denoised' / file_path.name), result)

    # --- 2. WYOSTRZANIE (Blur) ---
    blur_path = input_dir / 'blur'
    files = list(blur_path.glob('*'))
    if files:
        print(f"Wyostrzanie {len(files)} zdjęć...")
        for file_path in tqdm(files):
            img = cv2.imread(str(file_path))
            if img is None: continue
            
            result = enhancer.deblurring(img)
            cv2.imwrite(str(output_dir / 'sharpened' / file_path.name), result)

    # --- 3. NAPRAWA PIKSELOZY (Low Res) ---
    low_res_path = input_dir / 'low_res'
    files = list(low_res_path.glob('*'))
    if files:
        print(f"Wygładzanie pikselozy {len(files)} zdjęć...")
        for file_path in tqdm(files):
            img = cv2.imread(str(file_path))
            if img is None: continue
            
            # Ponieważ Twoje zdjęcia są już DUŻE (tylko pikselowate),
            # używamy scale_factor=1, żeby ich nie powiększać,
            # ale używamy metody Lanczos, która spróbuje wygładzić "kwadraty".
            result = enhancer.super_resolution(img, scale_factor=1, method='lanczos')
            cv2.imwrite(str(output_dir / 'upscaled' / file_path.name), result)

    print("\nGotowe! Wyniki są w folderze data/results/classic")

if __name__ == "__main__":
    run_classic_pipeline()