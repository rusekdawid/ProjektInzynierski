import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, task_type, img_size=256):
        """
        task_type: 'noise', 'blur' lub 'low_res'
        img_size: do jakiego rozmiaru skalować zdjęcia do treningu (domyślnie 256x256)
        """
        self.img_size = img_size
        self.base_dir = Path('data')
        self.input_dir = self.base_dir / 'processed' / task_type
        self.target_dir = self.base_dir / 'raw'
        
        # Pobieramy listę zepsutych plików
        self.files = list(self.input_dir.glob('*'))
        
        if len(self.files) == 0:
            print(f"BŁĄD: Nie znaleziono plików w {self.input_dir}")
            print("Upewnij się, że uruchomiłeś generator!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Ścieżki
        input_path = self.files[idx]
        file_name = input_path.name
        
        target_candidates = list(self.target_dir.rglob(file_name))
        if not target_candidates:
            raise FileNotFoundError(f"Brak oryginału dla {file_name}") 
        target_path = target_candidates[0]

        # 2. Wczytanie obrazów
        input_img = cv2.imread(str(input_path))
        target_img = cv2.imread(str(target_path))

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # --- ZMIANA: ZAMIAST RESIZE -> RANDOM CROP ---
        # Musimy wyciąć ten sam fragment z obu zdjęć!
        
        h, w, _ = input_img.shape
        crop_size = self.img_size # np. 256
        
        # Jeśli obraz jest mniejszy niż 256, musimy go powiększyć (rzadki przypadek)
        if h < crop_size or w < crop_size:
            input_img = cv2.resize(input_img, (crop_size, crop_size))
            target_img = cv2.resize(target_img, (crop_size, crop_size))
        else:
            # Losujemy punkt startowy (lewy górny róg)
            start_x = np.random.randint(0, w - crop_size + 1)
            start_y = np.random.randint(0, h - crop_size + 1)
            
            # Wycinamy kwadraty
            input_img = input_img[start_y:start_y+crop_size, start_x:start_x+crop_size]
            target_img = target_img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        # ---------------------------------------------

        # Normalizacja i Tensor
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor