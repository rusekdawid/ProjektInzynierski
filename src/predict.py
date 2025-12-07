import torch
import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from ai_model import SimpleUNet
import config as cfg

# --- PANEL STEROWANIA ---
# Wybierz, co chcesz naprawiać: 'noise', 'blur' lub 'low_res'
TASK_TYPE = 'low_res'
# ------------------------

def process_image_in_tiles(model, img, device, tile_size=cfg.TILE_SIZE):
    """
    Funkcja dzieli duże zdjęcie na mniejsze kwadraty (kafelki),
    naprawia każdy z nich osobno i skleja w całość.
    Zapobiega błędom braku pamięci (OOM).
    """
    h, w, c = img.shape
    result_img = np.zeros_like(img)
    
    # Obliczamy ile kafelków potrzebujemy
    tiles_y = math.ceil(h / tile_size)
    tiles_x = math.ceil(w / tile_size)
    
    with torch.no_grad():
        for y in range(tiles_y):
            for x in range(tiles_x):
                # Współrzędne wycinania
                start_y = y * tile_size
                start_x = x * tile_size
                end_y = min(start_y + tile_size, h)
                end_x = min(start_x + tile_size, w)
                
                # Wycinamy kafelek
                tile = img[start_y:end_y, start_x:end_x]
                
                # Padding (Dopełnienie krawędzi do wielokrotności 16)
                # U-Net tego wymaga, żeby wymiary się zgadzały przy łączeniu warstw
                th, tw = tile.shape[:2]
                pad_h = (16 - (th % 16)) % 16
                pad_w = (16 - (tw % 16)) % 16
                
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

                # Przygotowanie dla AI
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).float() / 255.0
                tensor = tensor.unsqueeze(0).to(device)
                
                # Magia AI
                output_tensor = model(tensor)
                
                # Powrót do obrazka
                output_tile = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output_tile = np.clip(output_tile, 0, 1) * 255.0
                output_tile = output_tile.astype(np.uint8)
                output_tile = cv2.cvtColor(output_tile, cv2.COLOR_RGB2BGR)
                
                # Usuwamy Padding (wycinamy środek)
                output_tile = output_tile[:th, :tw]
                
                # Wklejamy w miejsce docelowe
                result_img[start_y:end_y, start_x:end_x] = output_tile

    return result_img

def run_prediction():
    # Automatyczne ścieżki na podstawie TASK_TYPE
    model_name = f'model_{TASK_TYPE}.pth'
    model_path = cfg.MODELS_DIR / model_name
    
    input_dir = cfg.PROCESSED_DIR / TASK_TYPE
    output_dir = cfg.RESULTS_DIR / 'ai' / TASK_TYPE
    
    # Tworzymy folder wynikowy
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔮 URUCHAMIAM NAPRAWIANIE: {TASK_TYPE.upper()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie: {device}")
    print(f"Szukam modelu w: {model_path}")
    
    # Sprawdzenie czy model istnieje
    if not model_path.exists():
        print(f"❌ BŁĄD: Nie znaleziono pliku {model_name}!")
        print(f"   Upewnij się, że uruchomiłeś train.py z TASK_TYPE='{TASK_TYPE}'")
        return

    # Ładowanie modelu
    try:
        model = SimpleUNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model załadowany pomyślnie.")
    except Exception as e:
        print(f"❌ BŁĄD ładowania modelu: {e}")
        return
    
    # Pobranie listy plików
    files = list(input_dir.glob('*'))
    if not files:
        print(f"❌ BŁĄD: Pusty folder wejściowy: {input_dir}")
        print("   Uruchom data_generator.py!")
        return

    print(f"Przetwarzanie {len(files)} zdjęć...")

    # Główna pętla po plikach
    for file_path in tqdm(files):
        original_img = cv2.imread(str(file_path))
        if original_img is None: continue
        
        try:
            # Uruchamiamy funkcję kafelkową
            final_img = process_image_in_tiles(model, original_img, device)
            
            save_path = output_dir / file_path.name
            cv2.imwrite(str(save_path), final_img)
            
        except Exception as e:
            print(f"Błąd przy pliku {file_path.name}: {e}")

    print(f"\n✅ ZAKOŃCZONO! Wyniki w: {output_dir}")

if __name__ == "__main__":
    run_prediction()