import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ai_model import SimpleUNet
import config as cfg
import random
import torch.nn.functional as F
import json 


# --- LOSS FUNCTION ---
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        self.register_buffer('kx', k)
        self.register_buffer('ky', k.transpose(2, 3))
        
    def forward(self, p, t):
        # Konwersja do skali szarości dla krawędzi
        p_g = p.mean(1, keepdim=True)
        t_g = t.mean(1, keepdim=True)
        return torch.mean(torch.abs(F.conv2d(p_g, self.kx, padding=1) - F.conv2d(t_g, self.kx, padding=1)) +
                          torch.abs(F.conv2d(p_g, self.ky, padding=1) - F.conv2d(t_g, self.ky, padding=1)))

# --- DATASET Z AUGMENTACJĄ ---
class SmartDataset(Dataset):
    def __init__(self, task):
        self.task = task
        self.files = list((cfg.PROCESSED_DIR / task).glob('*'))
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        try:
            f = self.files[idx]
            # Znajdź oryginał
            orig_path = list(cfg.RAW_DIR.rglob(f.name))[0]
            
            inp = cv2.imread(str(f))
            tar = cv2.imread(str(orig_path))
            
            # --- FIX DLA LOW_RES ---
            # Jeśli to low_res, musimy powiększyć wejście, żeby pasowało do celu
            if self.task == 'low_res':
                h_t, w_t = tar.shape[:2]
                inp = cv2.resize(inp, (w_t, h_t), interpolation=cv2.INTER_CUBIC)
            
            # Upewnij się, że wymiary się zgadzają (czasem są różnice o 1px)
            h = min(inp.shape[0], tar.shape[0])
            w = min(inp.shape[1], tar.shape[1])
            inp = inp[:h, :w]
            tar = tar[:h, :w]

            # --- RANDOM CROP (Losowy wycinek) ---
            # Wycinamy kwadrat IMG_SIZE x IMG_SIZE
            if h > cfg.IMG_SIZE and w > cfg.IMG_SIZE:
                y = random.randint(0, h - cfg.IMG_SIZE)
                x = random.randint(0, w - cfg.IMG_SIZE)
                inp = inp[y:y+cfg.IMG_SIZE, x:x+cfg.IMG_SIZE]
                tar = tar[y:y+cfg.IMG_SIZE, x:x+cfg.IMG_SIZE]
            else:
                # Jeśli zdjęcie jest za małe, skalujemy je do wymaganego rozmiaru
                inp = cv2.resize(inp, (cfg.IMG_SIZE, cfg.IMG_SIZE))
                tar = cv2.resize(tar, (cfg.IMG_SIZE, cfg.IMG_SIZE))

            # --- AUGMENTACJA (To daje wysoki wynik!) ---
            # 1. Odbicie poziome
            if random.random() > 0.5:
                inp = cv2.flip(inp, 1)
                tar = cv2.flip(tar, 1)
            # 2. Odbicie pionowe
            if random.random() > 0.5:
                inp = cv2.flip(inp, 0)
                tar = cv2.flip(tar, 0)
            # 3. Rotacja
            k = random.randint(0, 3)
            if k > 0:
                inp = np.rot90(inp, k).copy()
                tar = np.rot90(tar, k).copy()

            # Konwersja BGR -> RGB i normalizacja 0-1
            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
            
            return (torch.from_numpy(inp).permute(2,0,1).float()/255.0, 
                    torch.from_numpy(tar).permute(2,0,1).float()/255.0)
                    
        except Exception as e:
            # W razie błędu zwróć czarny obraz (żeby nie wywalić treningu)
            return torch.zeros(3, cfg.IMG_SIZE, cfg.IMG_SIZE), torch.zeros(3, cfg.IMG_SIZE, cfg.IMG_SIZE)

def train_model(task_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 START TRENINGU: {task_name.upper()}")
    
    ds = SmartDataset(task_name)
    train_size = int(0.9 * len(ds))
    t_ds, v_ds = torch.utils.data.random_split(ds, [train_size, len(ds)-train_size])
    
    t_load = DataLoader(t_ds, cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    v_load = DataLoader(v_ds, cfg.BATCH_SIZE, shuffle=False)
    
    model = SimpleUNet().to(device)
    
    # Próba wczytania istniejącego modelu, żeby go dotrenować (Fine-tuning)
    model_path = cfg.MODELS_DIR / f'model_{task_name}.pth'
    if model_path.exists():
        print("   🔄 Wczytuję istniejący model do dotrenowania...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    criterion_pix = nn.L1Loss()
    criterion_edge = EdgeLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    
    # Historia do wykresów
    history = {'train_loss': [], 'val_loss': []}
    
    for ep in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(t_load, desc=f"Epoka {ep+1}/{cfg.EPOCHS}", leave=False)
        
        for inp, tar in loop:
            inp, tar = inp.to(device), tar.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion_pix(out, tar) + 0.1 * criterion_edge(out, tar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inp, tar in v_load:
                inp, tar = inp.to(device), tar.to(device)
                out = model(inp)
                val_loss += criterion_pix(out, tar).item()
        
        avg_train = train_loss / len(t_load)
        avg_val = val_loss / len(v_load)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        
        scheduler.step(avg_val)
        print(f"   -> Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), model_path)
            print("      💾 Zapisano model!")

    # Zapis historii do pliku
    with open(cfg.RESULTS_DIR / f'history_{task_name}.json', 'w') as f:
        json.dump(history, f)
    print(f"✅ Historia zapisana w history_{task_name}.json")