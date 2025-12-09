import torch
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from ai_model import SRResNet
import config as cfg

def process_tiled(model, img, device, tile_size=256, overlap=16):
    """
    Przetwarza obraz kafelkami z zakładką (overlap), aby usunąć linie łączenia.
    """
    h, w, c = img.shape
    
    # 1. Padding do wielokrotności tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    # Dodajemy padding do oryginalnego rozmiaru
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    # 2. Dodajemy DODATKOWY padding na overlap (zakładkę) dookoła całego obrazu
    img_padded = cv2.copyMakeBorder(img_padded, overlap, overlap, overlap, overlap, cv2.BORDER_REFLECT)
    
    h_pad, w_pad, _ = img_padded.shape
    
    # Pusty obraz wynikowy (rozmiar jak po pierwszym paddingu, bez overlap)
    result_h = h + pad_h
    result_w = w + pad_w
    result = np.zeros((result_h, result_w, c), dtype=np.uint8)
    
    # 3. Iteracja po kafelkach
    for y in range(0, result_h, tile_size):
        for x in range(0, result_w, tile_size):
            
            # Współrzędne wejściowe (z overlapem)
            # Wycinamy kafelek wejściowy (większy o 2*overlap)
            tile_in = img_padded[y : y + tile_size + 2*overlap, 
                                 x : x + tile_size + 2*overlap]
            
            # --- PRZETWARZANIE AI ---
            inp = cv2.cvtColor(tile_in, cv2.COLOR_BGR2RGB)
            inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
            inp_tensor = inp_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                out_tensor = model(inp_tensor)
                
            out = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # ------------------------
            
            # 4. Wycinanie środka (usuwanie brzegów z błędami)
            # Odcinamy 'overlap' z każdej strony, zostawiając czysty środek
            tile_out = out[overlap : -overlap, overlap : -overlap]
            
            # Wklejamy czysty środek do wyniku
            # Zabezpieczenie wymiarów (na wypadek gdybyśmy byli przy krawędzi)
            th, tw = tile_out.shape[:2]
            h_end = min(y + th, result_h)
            w_end = min(x + tw, result_w)
            
            result[y:h_end, x:w_end] = tile_out[:h_end-y, :w_end-x]

    # 5. Przycięcie do oryginalnego rozmiaru
    return result[:h, :w]

def run_prediction(task_name):
    print(f"\n🔮 GENEROWANIE WYNIKÓW AI (SRResNet - Seamless): {task_name.upper()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SRResNet().to(device)
    path = cfg.MODELS_DIR / f'model_{task_name}.pth'
    
    if not path.exists():
        print(f"❌ Nie znaleziono modelu: {path}")
        return

    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except:
        print("⚠️ Błąd ładowania wag.")
        return
        
    model.eval()
    
    out_dir = cfg.RESULTS_DIR / 'ai' / task_name
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list((cfg.PROCESSED_DIR / task_name).glob('*'))
    
    for f in tqdm(files, desc="Przetwarzanie"):
        img = cv2.imread(str(f))
        if img is None: continue
        
        if task_name == 'low_res':
            h, w = img.shape[:2]
            img = cv2.resize(img, (w*cfg.SCALE_FACTOR, h*cfg.SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)

        try:
            # Używamy overlap=16 domyślnie
            res = process_tiled(model, img, device, tile_size=256, overlap=16)
            cv2.imwrite(str(out_dir / f.name), res)
        except RuntimeError:
            print(f"⚠️ OOM na GPU dla {f.name}, próba na CPU...")
            torch.cuda.empty_cache()
            model_cpu = model.to('cpu')
            res = process_tiled(model_cpu, img, 'cpu', tile_size=128, overlap=16)
            cv2.imwrite(str(out_dir / f.name), res)
            model.to(device)

if __name__ == "__main__":
    run_prediction('noise')