import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ai_model import SimpleUNet

# --- KONFIGURACJA ---
MODEL_PATH = 'models/model_noise.pth'
# UWAGA: Nie ma tu IMG_SIZE, bo bierzemy oryginał!
# --------------------

def run_ai_full_resolution():
    base_dir = Path('data')
    input_dir = base_dir / 'processed' / 'noise'
    output_dir = base_dir / 'results' / 'ai' / 'denoised'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}")
    
    # 1. Ładujemy model
    model = SimpleUNet().to(device)
    if not Path(MODEL_PATH).exists():
        print("Brak modelu!")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    files = list(input_dir.glob('*'))
    print(f"Przetwarzanie {len(files)} zdjęć w pełnej rozdzielczości...")

    with torch.no_grad():
        for file_path in tqdm(files):
            # A. Wczytaj oryginał
            original_img = cv2.imread(str(file_path))
            if original_img is None: continue
            
            # B. Przygotowanie wymiarów (Padding/Crop)
            # Sieć U-Net ma 2 poziomy poolingu, więc wymiary muszą być podzielne przez 4 (dla bezpieczeństwa przez 16)
            h, w = original_img.shape[:2]
            
            # Obliczamy nowe wymiary (najbliższa wielokrotność 16 w dół)
            new_h = (h // 16) * 16
            new_w = (w // 16) * 16
            
            # Przycinamy lekko obraz, żeby pasował do sieci (tracimy max 15 pikseli przy krawędzi)
            img_cropped = original_img[:new_h, :new_w]
            
            # C. Konwersja do Tensora
            img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(device) # Dodajemy batch dimension
            
            # D. AI w akcji (Na pełnym obrazie!)
            try:
                output_tensor = model(input_tensor)
            except RuntimeError as e:
                print(f"\nBłąd pamięci GPU dla pliku {file_path.name}: {e}")
                print("Obraz jest za duży na Twoją kartę graficzną.")
                continue

            # E. Zapis wyniku
            output_img = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_img = np.clip(output_img, 0, 1) * 255.0
            output_img = output_img.astype(np.uint8)
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            
            save_path = output_dir / file_path.name
            cv2.imwrite(str(save_path), output_img)

    print("Gotowe! Sprawdź folder results/ai/denoised.")

if __name__ == "__main__":
    run_ai_full_resolution()