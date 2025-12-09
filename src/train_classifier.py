import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import config as cfg
from classifier_model import DistortionClassifier

# --- KONFIGURACJA ---
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15  # Zwiększyłem lekko liczbę epok

class DistortionDataset(Dataset):
    def __init__(self, num_samples=3000):
        self.files = list(cfg.RAW_DIR.rglob('*.[jp][pn]*[g]'))
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 1. Losujemy plik
        f = random.choice(self.files)
        img = cv2.imread(str(f))
        if img is None: return torch.zeros(3, IMG_SIZE, IMG_SIZE), 0
        
        # Upewniamy się, że obraz nie jest za duży (dla wydajności), ale też nie za mały
        h, w = img.shape[:2]
        if h > 512:
            img = cv2.resize(img, (512, 512)) # Pracujemy na rozsądnym rozmiarze
        
        # 2. LOSUJEMY ZNIEKSZTAŁCENIE (NA DUŻYM OBRAZIE!)
        label = random.randint(0, 2) # 0: Noise, 1: Blur, 2: LowRes
        
        if label == 0: # NOISE
            # Szum jest łatwy do wykrycia, więc dajemy różną siłę
            noise = np.random.normal(0, random.uniform(15, 60), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
        elif label == 1: # BLUR
            # Rozmycie musi być wyraźne
            k = random.choice([5, 7, 9, 11, 13])
            img = cv2.GaussianBlur(img, (k, k), 0)
            
        elif label == 2: # LOW RES (Pikseloza)
            # Symulujemy pikselozę (Nearest Neighbor)
            scale = random.choice([4, 6, 8, 10])
            h_curr, w_curr = img.shape[:2]
            # Zmniejszamy
            small = cv2.resize(img, (w_curr//scale, h_curr//scale), interpolation=cv2.INTER_LINEAR)
            # Powiększamy NEAREST (żeby powstały kwadraty)
            img = cv2.resize(small, (w_curr, h_curr), interpolation=cv2.INTER_NEAREST)

        # 3. Dopiero teraz zmniejszamy do 64x64 dla sieci
        # Dzięki temu sieć zobaczy "zmniejszone kwadraty" lub "zmniejszone rozmycie"
        img_input = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalizacja i tensory
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        
        return img_tensor, label

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🚀 Trenowanie Klasyfikatora Zniekształceń (Poprawiona logika)...")
    
    model = DistortionClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    ds = DistortionDataset(num_samples=4000) # Więcej próbek
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    for ep in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(loader, desc=f"Epoka {ep+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=f"{100 * correct / total:.1f}%")
            
    cfg.MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), cfg.MODELS_DIR / 'classifier.pth')
    print("✅ Klasyfikator zapisany! Teraz powinien działać poprawnie.")

if __name__ == "__main__":
    train()