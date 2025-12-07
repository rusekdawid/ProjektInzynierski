import cv2
import json
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def print_experiment_settings(processed_dir):
    """Odczytuje i wyświetla parametry użyte do psucia zdjęć."""
    settings_path = Path(processed_dir) / 'parameters.json'
    print("\n" + "="*40)
    print(" RAPORT EKSPERYMENTU")
    print("="*40)
    
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            print(f"[-] Siła Szumu (Sigma):      {settings.get('noise_severity')}")
            print(f"[-] Wielkość Rozmycia:       {settings.get('blur_kernel')}")
            print(f"[-] Skala Pikselozy:         {settings.get('scale_factor')}")
    else:
        print("[!] UWAGA: Nie znaleziono pliku parameters.json!")
        print("    Nie wiadomo, jak bardzo zepsute są zdjęcia.")
    print("="*40 + "\n")

def calculate_metrics(img_path_restored, img_path_original):
    img_restored = cv2.imread(str(img_path_restored))
    img_original = cv2.imread(str(img_path_original))

    if img_restored is None or img_original is None:
        return None, None

    h_min = min(img_restored.shape[0], img_original.shape[0])
    w_min = min(img_restored.shape[1], img_original.shape[1])
    
    img_restored = img_restored[:h_min, :w_min]
    img_original = img_original[:h_min, :w_min]

    val_psnr = psnr(img_original, img_restored)
    # win_size dostosowany do małych obrazów, channel_axis dla koloru
    val_ssim = ssim(img_original, img_restored, channel_axis=2, win_size=3)

    return val_psnr, val_ssim

def evaluate_folder(task_name, results_folder, originals_folder):
    results_path = Path(results_folder)
    files = list(results_path.glob('*'))
    
    if not files:
        print(f"--- {task_name.upper()}: Brak plików ---")
        return

    psnr_values = []
    ssim_values = []

    print(f"--- OCENA: {task_name.upper()} ---")
    
    for file_path in files:
        original_candidates = list(Path(originals_folder).rglob(file_path.name))
        if not original_candidates: continue
        original_path = original_candidates[0]

        val_psnr, val_ssim = calculate_metrics(file_path, original_path)
        
        if val_psnr is not None:
            psnr_values.append(val_psnr)
            ssim_values.append(val_ssim)

    if psnr_values:
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print(f"-> ŚREDNI PSNR: {avg_psnr:.2f} dB")
        print(f"-> ŚREDNI SSIM: {avg_ssim:.4f}\n")
    else:
        print("Brak wyników.\n")

def run_evaluation():
    base_dir = Path('data')
    originals_dir = base_dir / 'raw'
    processed_dir = base_dir / 'processed' # Tu szukamy pliku json
    results_dir = base_dir / 'results' / 'ai'

    # 1. Najpierw wyświetl parametry
    print_experiment_settings(processed_dir)

    # 2. Potem oceń wyniki
    evaluate_folder("Usuwanie szumu", results_dir / 'denoised', originals_dir)
    evaluate_folder("Redukcja rozmycia", results_dir / 'sharpened', originals_dir)
    evaluate_folder("Poprawa rozdzielczości", results_dir / 'upscaled', originals_dir)

if __name__ == "__main__":
    run_evaluation()