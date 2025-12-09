import torch
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from ai_model import SimpleUNet
import config as cfg

def process_tiled(model, img, device, tile_size=512):
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

def run_prediction(task_name):
    print(f"\n🔮 GENEROWANIE WYNIKÓW AI: {task_name.upper()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleUNet().to(device)
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
    
    # --- CZYSZCZENIE ---
    if out_dir.exists():
        shutil.rmtree(out_dir) # Usuwa stare śmieci
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list((cfg.PROCESSED_DIR / task_name).glob('*'))
    
    for f in tqdm(files, desc="Przetwarzanie"):
        img = cv2.imread(str(f))
        if img is None: continue
        
        if task_name == 'low_res':
            h, w = img.shape[:2]
            img = cv2.resize(img, (w*cfg.SCALE_FACTOR, h*cfg.SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)

        try:
            res = process_tiled(model, img, device, tile_size=512)
            cv2.imwrite(str(out_dir / f.name), res)
        except RuntimeError:
            torch.cuda.empty_cache()
            model_cpu = model.to('cpu')
            res = process_tiled(model_cpu, img, 'cpu', tile_size=256)
            cv2.imwrite(str(out_dir / f.name), res)
            model.to(device)