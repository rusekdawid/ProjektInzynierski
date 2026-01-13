import cv2
import numpy as np
import torch
import math
from pathlib import Path
from ai_model import SimpleUNet
from classic_methods import ClassicEnhancer
import config as cfg

class SmartSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.classic = ClassicEnhancer()
        
        self.model_paths = {
            'noise': cfg.MODELS_DIR / 'model_noise.pth',
            'blur': cfg.MODELS_DIR / 'model_blur.pth',
            'low_res': cfg.MODELS_DIR / 'model_low_res.pth'
        }

    def load_model(self, task_type):
        if task_type in self.models: return self.models[task_type]
        path = self.model_paths.get(task_type)
        if not path or not path.exists(): return None
        
        model = SimpleUNet().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        self.models[task_type] = model
        return model

    def detect_problem(self, img):
        """
        INTELIGENTNA DETEKCJA v3
        Rozróżnia silne rozmycie od pikselozy.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Obliczamy ostrość krawędzi (Laplacian)
        # To jest klucz! 
        # Pikseloza (Low Res) ma WYSOKĄ wariancję (bo ma ostre schodki).
        # Mocny Blur ma BARDZO NISKĄ wariancję (bo wszystko jest mydłem).
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Obliczamy "płaskość" (dla pikselozy)
        diff_x = np.abs(gray[:, :-1].astype(np.float32) - gray[:, 1:].astype(np.float32))
        flat_pixels = np.sum(diff_x < 2.0)
        total_pixels = diff_x.size
        flat_ratio = flat_pixels / total_pixels

        # 3. Obliczamy Szum
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_score = np.mean(cv2.absdiff(gray, blurred))

        print(f"[Detekcja] Blur Var: {laplacian_var:.1f}, Flat Ratio: {flat_ratio:.2f}, Noise: {noise_score:.1f}")

        # --- LOGIKA DECYZYJNA (Poprawiona kolejność) ---

        # A. Jeśli obraz jest tragicznie nieostry (Blur > 10), to Laplacian spada drastycznie.
        # Nawet jeśli flat_ratio jest wysokie, to brak krawędzi oznacza Blur.
        if laplacian_var < 20.0:  # Bardzo niski próg dla silnego blura
            print("--> Wykryto SILNE ROZMYCIE (Brak krawędzi)")
            return 'blur'

        # B. Jeśli obraz ma ostre krawędzie (Laplacian > 20), ale jest "płaski" w środku, to Pikseloza.
        if flat_ratio > 0.40:
            print("--> Wykryto PIKSELOZĘ (Schodki)")
            return 'low_res'

        # C. Standardowe sprawdzanie mniejszego blura
        if laplacian_var < 100.0:
            return 'blur'
            
        # D. Szum na końcu
        if noise_score > 3.0:
            return 'noise'
            
        return 'clean'

    def process_tile(self, model, img, tile_size=1536): # <--- ZMIANA NA 1536
        """Naprawa kafelkowa (Zwiększony kafel = brak linii na średnich zdjęciach)"""
        h, w, c = img.shape
        result_img = np.zeros_like(img)
        tiles_y = math.ceil(h / tile_size)
        tiles_x = math.ceil(w / tile_size)
        
        with torch.no_grad():
            for y in range(tiles_y):
                for x in range(tiles_x):
                    sy, sx = y*tile_size, x*tile_size
                    ey, ex = min(sy+tile_size, h), min(sx+tile_size, w)
                    
                    tile = img[sy:ey, sx:ex]
                    
                    # Padding
                    th, tw = tile.shape[:2]
                    ph = (16 - (th % 16)) % 16
                    pw = (16 - (tw % 16)) % 16
                    if ph>0 or pw>0: tile = cv2.copyMakeBorder(tile, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                    
                    inp = torch.from_numpy(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255.0).unsqueeze(0).to(self.device)
                    out = model(inp)
                    out = out.squeeze(0).permute(1,2,0).cpu().numpy()
                    out = np.clip(out, 0, 1) * 255.0
                    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    result_img[sy:ey, sx:ex] = out[:th, :tw]
        return result_img

    def process_image(self, img_rgb, manual_mode='Auto'):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        problem = 'clean'
        if manual_mode == 'Auto':
            problem = self.detect_problem(img_bgr)
        else:
            # Mapowanie GUI
            if 'Szum' in manual_mode: problem = 'noise'
            elif 'Rozmycie' in manual_mode: problem = 'blur'
            elif 'Rozdzielczość' in manual_mode or 'Pikseloza' in manual_mode: problem = 'low_res'

        if problem == 'clean': return img_rgb, "Obraz jest czysty.", problem

        model = self.load_model(problem)
        if model:
            res_bgr = self.process_tile(model, img_bgr)
            msg = f"AI: {problem.upper()}"
        else:
            if problem == 'noise': res_bgr = self.classic.denoise_gaussian(img_bgr)
            elif problem == 'blur': res_bgr = self.classic.deblur_unsharp_mask(img_bgr)
            elif problem == 'low_res': res_bgr = self.classic.upscale_lanczos(img_bgr)
            else: res_bgr = img_bgr
            msg = f"Klasyka: {problem}"

        return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), msg, problem