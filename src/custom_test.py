import cv2
import torch
import numpy as np
import config as cfg
from ai_model import SimpleUNet
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- KONFIGURACJA ---
INPUT_FOLDER = cfg.BASE_DIR / 'test_manual'
OUTPUT_FOLDER = INPUT_FOLDER / 'results'
TASK = 'noise'  # Możesz zmienić na 'blur' lub 'low_res'

def add_noise(img):
    noise = np.random.normal(0, cfg.NOISE_LEVEL, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def process_tiled(model, img, device, tile_size=512):
    """Kafelkowanie (skopiowane z predict.py dla bezpieczeństwa)"""
    h, w, c = img.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad, _ = img_padded.shape
    result = np.zeros_like(img_padded)
    
    for y in range(0, h_pad, tile_size):
        for x in range(0, w_pad, tile_size):
            y_end = min(y + tile_size, h_pad)
            x_end = min(x + tile_size, w_pad)
            tile = img_padded[y:y_end, x:x_end]
            inp = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            inp_tensor = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
            inp_tensor = inp_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                out_tensor = model(inp_tensor)
            out = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            result[y:y_end, x:x_end] = out
    return result[:h, :w]

def run_custom_test():
    print(f"🧪 TEST MANUALNY: {TASK.upper()}")
    print(f"📂 Folder wejściowy: {INPUT_FOLDER}")
    
    if not INPUT_FOLDER.exists():
        print("❌ Nie znaleziono folderu data/test_manual! Stwórz go i wrzuć zdjęcia.")
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Ładowanie modelu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet().to(device)
    model_path = cfg.MODELS_DIR / f'model_{TASK}.pth'
    
    if not model_path.exists():
        print(f"❌ Brak modelu: {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    files = list(INPUT_FOLDER.glob('*.*'))
    files = [f for f in files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    if not files:
        print("❌ Folder jest pusty.")
        return

    print(f"Znaleziono {len(files)} zdjęć. Przetwarzanie...")

    for f in files:
        # 1. Wczytaj oryginał
        orig = cv2.imread(str(f))
        if orig is None: continue
        
        # 2. Zepsuj go (Symulacja zniekształcenia)
        if TASK == 'noise':
            noisy = add_noise(orig)
        else:
            print("Ten skrypt w tej wersji obsługuje tylko noise (dla uproszczenia).")
            return

        # 3. Napraw przez AI
        restored = process_tiled(model, noisy, device)
        
        # 4. Policz PSNR dla tego jednego zdjęcia
        psnr_val = psnr(orig, restored, data_range=255)
        
        # 5. Zapisz wyniki
        cv2.imwrite(str(OUTPUT_FOLDER / f"1_orig_{f.name}"), orig)
        cv2.imwrite(str(OUTPUT_FOLDER / f"2_noisy_{f.name}"), noisy)
        cv2.imwrite(str(OUTPUT_FOLDER / f"3_ai_{f.name}"), restored)
        
        print(f"   📸 {f.name}: PSNR = {psnr_val:.2f} dB")

    print(f"\n✅ Gotowe! Sprawdź folder: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_custom_test()