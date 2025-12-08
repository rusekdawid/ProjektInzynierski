import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt  # <--- Do wykresów
from pathlib import Path
from tqdm import tqdm
from ai_model import SimpleUNet
import config as cfg
import random

# --- PANEL STEROWANIA ---
TASK_TYPE = 'low_res'   # noise / blur / low_res
VALIDATION_SPLIT = 0.2  # 20% zdjęć odkładamy na bok do testów
# ------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

class AdvancedDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            input_path = self.files[idx]
            target_path = list(cfg.RAW_DIR.rglob(input_path.name))[0]
            
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
        except Exception:
            return torch.zeros(3, cfg.IMG_SIZE, cfg.IMG_SIZE), torch.zeros(3, cfg.IMG_SIZE, cfg.IMG_SIZE)

def plot_learning_curve(train_loss, val_loss):
    """Generuje wykres do pracy inżynierskiej"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Trening (Nauka)', color='blue')
    plt.plot(val_loss, label='Walidacja (Sprawdzian)', color='orange')
    plt.title(f'Krzywa Uczenia: {TASK_TYPE.upper()}')
    plt.xlabel('Epoki')
    plt.ylabel('Błąd (Loss)')
    plt.legend()
    plt.grid(True)
    
    # Zapisz wykres w folderze results
    save_path = cfg.RESULTS_DIR / f'learning_curve_{TASK_TYPE}.png'
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"\n📈 Wykres zapisano w: {save_path}")

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 TRENING Z WALIDACJĄ: {TASK_TYPE.upper()}")
    
    # 1. Pobieranie plików
    source_folder = cfg.PROCESSED_DIR / TASK_TYPE
    all_files = list(source_folder.glob('*'))
    if not all_files:
        print("Brak plików! Uruchom generator.")
        return

    # 2. Mieszanie i dzielenie (Train / Val)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * (1 - VALIDATION_SPLIT))
    
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"📊 Statystyka: {len(train_files)} do nauki | {len(val_files)} do testów")
    
    # Loadery
    train_loader = DataLoader(AdvancedDataset(train_files), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AdvancedDataset(val_files), batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    model = SimpleUNet().to(device)
    criterion = CharbonnierLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Historia błędów
    history_train = []
    history_val = []
    best_val_loss = float('inf')

    for epoch in range(cfg.EPOCHS):
        # --- ETAP NAUKI ---
        model.train()
        epoch_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1}/{cfg.EPOCHS}")
        
        for inp, tar in loop:
            inp, tar = inp.to(device), tar.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, tar)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_train = epoch_train_loss / len(train_loader)
        history_train.append(avg_train)
        
        # --- ETAP SPRAWDZIANU (Walidacja) ---
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inp, tar in val_loader:
                inp, tar = inp.to(device), tar.to(device)
                out = model(inp)
                loss = criterion(out, tar)
                epoch_val_loss += loss.item()
        
        avg_val = epoch_val_loss / len(val_loader)
        history_val.append(avg_val)
        
        # Scheduler (zwalnia naukę jak wyniki stoją)
        scheduler.step(avg_val)
        
        # Zapisujemy "Najlepszy Model" (Checkpoint)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            cfg.MODELS_DIR.mkdir(exist_ok=True)
            save_path = cfg.MODELS_DIR / f'model_{TASK_TYPE}.pth'
            torch.save(model.state_dict(), save_path)
            # print(f"  [Zapisano rekord: {best_val_loss:.5f}]")

    print("✅ Trening zakończony.")
    plot_learning_curve(history_train, history_val)

if __name__ == "__main__":
    train()