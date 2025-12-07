import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ai_model import SimpleUNet
import config as cfg

# --- PANEL STEROWANIA TRENINGIEM ---
# Zmieniaj tylko tę jedną wartość!
# Dostępne opcje: 'noise', 'blur', 'low_res'
TASK_TYPE = 'low_res' 
# -----------------------------------

class RandomCropDataset(Dataset):
    def __init__(self):
        # Automatycznie wybieramy folder na podstawie TASK_TYPE
        self.source_folder = cfg.PROCESSED_DIR / TASK_TYPE
        self.files = list(self.source_folder.glob('*'))
        
        if len(self.files) == 0:
            raise RuntimeError(f"BŁĄD: Folder {self.source_folder} jest pusty! Uruchom data_generator.py.")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        input_path = self.files[idx]
        # Szukamy oryginału
        try:
            target_path = list(cfg.RAW_DIR.rglob(input_path.name))[0]
        except IndexError:
             raise RuntimeError(f"Brak oryginału dla {input_path.name}")

        inp = cv2.imread(str(input_path))
        tar = cv2.imread(str(target_path))
        
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
        
        # Random Crop
        h, w, _ = inp.shape
        if h > cfg.IMG_SIZE and w > cfg.IMG_SIZE:
            x = np.random.randint(0, w - cfg.IMG_SIZE)
            y = np.random.randint(0, h - cfg.IMG_SIZE)
            inp = inp[y:y+cfg.IMG_SIZE, x:x+cfg.IMG_SIZE]
            tar = tar[y:y+cfg.IMG_SIZE, x:x+cfg.IMG_SIZE]
        else:
            inp = cv2.resize(inp, (cfg.IMG_SIZE, cfg.IMG_SIZE))
            tar = cv2.resize(tar, (cfg.IMG_SIZE, cfg.IMG_SIZE))
            
        inp_t = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0
        tar_t = torch.from_numpy(tar).permute(2, 0, 1).float() / 255.0
        return inp_t, tar_t

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 ROZPOCZYNAM TRENING: {TASK_TYPE.upper()}")
    print(f"Urządzenie: {device}")
    
    dataset = RandomCropDataset()
    print(f"Liczba zdjęć w folderze '{TASK_TYPE}': {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    model = SimpleUNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    model.train()
    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0
        progress = tqdm(loader, desc=f"Epoka {epoch+1}/{cfg.EPOCHS}")
        for inp, tar in progress:
            inp, tar = inp.to(device), tar.to(device)
            
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, tar)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
    # Zapis z dynamiczną nazwą
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    save_path = cfg.MODELS_DIR / f'model_{TASK_TYPE}.pth' # <-- Automatyczna nazwa!
    torch.save(model.state_dict(), save_path)
    
    print("="*30)
    print(f"✅ SUKCES! Model zapisano jako: {save_path.name}")
    print("="*30)

if __name__ == "__main__":
    train()