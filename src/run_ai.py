import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ai_model import SimpleUNet

# --- KONFIGURACJA ---
MODEL_PATH = 'models/model_noise.pth'  # Ścieżka do wytrenowanego modelu
IMG_SIZE = 256                         # Musi być taki sam jak w treningu!
# --------------------

def run_ai_inference():
    # 1. Przygotowanie ścieżek
    base_dir = Path('data')
    input_dir = base_dir / 'processed' / 'noise'  # Skupiamy się na szumie
    output_dir = base_dir / 'results' / 'ai' / 'denoised'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Ładowanie modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Uruchamiam AI na urządzeniu: {device}")
    
    model = SimpleUNet().to(device)
    
    # Wczytujemy wagi z pliku .pth
    if not Path(MODEL_PATH).exists():
        print(f"BŁĄD: Nie znaleziono modelu {MODEL_PATH}! Uruchom najpierw train.py.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Ważne! Przełącza sieć w tryb "pracy" (wyłącza uczenie)
    print("Model załadowany pomyślnie.")

    # 3. Przetwarzanie zdjęć
    files = list(input_dir.glob('*'))
    print(f"Przetwarzanie {len(files)} zdjęć...")

    with torch.no_grad(): # Ważne! Wyłączamy obliczanie gradientów (oszczędza pamięć)
        for file_path in tqdm(files):
            # A. Wczytaj obraz
            original_img = cv2.imread(str(file_path))
            if original_img is None: continue

            # Zapamiętujemy oryginalny rozmiar, żeby potem do niego wrócić
            h, w = original_img.shape[:2]

            # B. Pre-processing (Tak samo jak w dataset.py!)
            img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Zamiana na tensor: (H, W, C) -> (C, H, W) -> Dodanie Batch (1, C, H, W)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # C. AI naprawia zdjęcie (Inference)
            output_tensor = model(img_tensor)

            # D. Post-processing (Powrót do obrazka)
            # Usuwamy Batch, zmieniamy kolejność wymiarów, zdejmujemy z GPU
            output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Zabezpieczenie wartości (clip 0-1) i skalowanie do 255
            output_img = np.clip(output_img, 0, 1) * 255.0
            output_img = output_img.astype(np.uint8)
            
            # Konwersja RGB -> BGR (żeby zapisać przez OpenCV)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            # Opcjonalnie: Skalowanie z powrotem do oryginalnego rozmiaru zdjęcia
            output_img = cv2.resize(output_img, (w, h))

            # E. Zapisz wynik
            save_path = output_dir / file_path.name
            cv2.imwrite(str(save_path), output_img)

    print(f"Gotowe! Wyniki zapisano w: {output_dir}")

if __name__ == "__main__":
    run_ai_inference()