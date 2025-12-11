import torch
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ai_model import SRResNet
import config as cfg

def process_tiled(model, img, device, tile_size=256, overlap=16):
    """
    Przetwarza obraz kafelkami z zakładką (overlap), aby usunąć linie łączenia.
    Obsługuje duże obrazy bez wywalania błędu pamięci (OOM).
    """
    h, w, c = img.shape
    
    # 1. Padding do wielokrotności tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    # Dodajemy padding do oryginalnego rozmiaru (odbicie lustrzane krawędzi)
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    # 2. Dodajemy DODATKOWY padding na overlap (zakładkę) dookoła całego obrazu
    img_padded = cv2.copyMakeBorder(img_padded, overlap, overlap, overlap, overlap, cv2.BORDER_REFLECT)
    
    h_pad, w_pad, _ = img_padded.shape
    
    # Pusty obraz wynikowy (rozmiar jak po pierwszym paddingu, bez overlap)
    result_h = h + pad_h
    result_w = w + pad_w
    result = np.zeros((result_h, result_w, c), dtype=np.uint8)
    
    # 3. Iteracja po kafelkach
    # Używamy torch.no_grad() tutaj, żeby nie trzymać grafu obliczeń w pamięci
    with torch.no_grad():
        for y in range(0, result_h, tile_size):
            for x in range(0, result_w, tile_size):
                
                # Wycinamy kafelek wejściowy (większy o 2*overlap)
                tile_in = img_padded[y : y + tile_size + 2*overlap, 
                                     x : x + tile_size + 2*overlap]
                
                # --- PRZETWARZANIE AI ---
                inp = cv2.cvtColor(tile_in, cv2.COLOR_BGR2RGB)
                inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
                inp_tensor = inp_tensor.unsqueeze(0).to(device)
                
                out_tensor = model(inp_tensor)
                    
                out = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = np.clip(out * 255, 0, 255).astype(np.uint8)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                # ------------------------
                
                # 4. Wycinanie środka (usuwanie brzegów z błędami/overlapem)
                tile_out = out[overlap : -overlap, overlap : -overlap]
                
                # Wklejamy czysty środek do wyniku
                # Zabezpieczenie wymiarów (na wypadek krawędzi)
                th, tw = tile_out.shape[:2]
                h_end = min(y + th, result_h)
                w_end = min(x + tw, result_w)
                
                result[y:h_end, x:w_end] = tile_out[:h_end-y, :w_end-x]

    # 5. Przycięcie do oryginalnego rozmiaru (usuwamy padding z kroku 1)
    return result[:h, :w]

def run_prediction(task_name):
    # Wykrywanie urządzenia (zgodnie z configiem lub auto)
    if hasattr(cfg, 'DEVICE'):
        device = cfg.DEVICE
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nGENEROWANIE WYNIKÓW AI (SRResNet): {task_name.upper()}")
    print(f"   Urządzenie: {device}")
    
    # Ładowanie modelu
    model = SRResNet().to(device)
    path = cfg.MODELS_DIR / f'model_{task_name}.pth'
    
    if not path.exists():
        print(f"Nie znaleziono modelu: {path}")
        print("Najpierw uruchom trening (Opcja 3).")
        return

    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Błąd ładowania wag: {e}")
        return
        
    # Przygotowanie folderów
    out_dir = cfg.RESULTS_DIR / 'ai' / task_name
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pobranie listy plików
    input_dir = cfg.PROCESSED_DIR / task_name
    files = list(input_dir.glob('*'))
    
    if not files:
        print(f"Brak plików w {input_dir}. Uruchom najpierw generator danych (Opcja 1).")
        return

    # Pętla przetwarzania
    for f in tqdm(files, desc="Przetwarzanie"):
        img = cv2.imread(str(f))
        if img is None: continue
        
        # --- FIX: Obsługa obrazów czarno-białych ---
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # --- LOGIKA LOW_RES (Zgodna z train.py) ---
        # Model SRResNet (VDSR) oczekuje na wejściu obrazu już powiększonego,
        # żeby móc dodać do niego detale (skip connection).
        if task_name == 'low_res':
            h, w = img.shape[:2]
            img = cv2.resize(img, (w * cfg.SCALE_FACTOR, h * cfg.SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)

        # --- INFERENCE Z ZABEZPIECZENIEM PAMIĘCI ---
        try:
            # Domyślne ustawienia: kafelki 256px, zakładka 16px
            res = process_tiled(model, img, device, tile_size=256, overlap=16)
            cv2.imwrite(str(out_dir / f.name), res)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM na GPU dla {f.name}, przełączam na CPU...")
                torch.cuda.empty_cache()
                
                # Przenosimy model na CPU tymczasowo
                model_cpu = model.to('cpu')
                res = process_tiled(model_cpu, img, 'cpu', tile_size=128, overlap=16)
                cv2.imwrite(str(out_dir / f.name), res)
                
                # Wracamy na GPU
                model.to(device)
            else:
                print(f"Nieoczekiwany błąd dla {f.name}: {e}")

if __name__ == "__main__":
    # Testowe uruchomienie
    run_prediction('noise')