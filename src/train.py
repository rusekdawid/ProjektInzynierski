import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Importujemy nasze moduły
from ai_model import SimpleUNet
from dataset import ImageDataset

# --- KONFIGURACJA TRENINGU ---
# Możesz eksperymentować z tymi liczbami w pracy
EPOCHS = 100           # Ile razy przerobimy cały zbiór (na start 20 jest ok)
BATCH_SIZE = 8        # Ile zdjęć naraz (zmniejsz do 2, jeśli wywali błąd pamięci)
LEARNING_RATE = 0.001 # Szybkość uczenia (standardowa wartość)
IMG_SIZE = 256        # Rozmiar obrazków do treningu
# -----------------------------

def train_task(task_name):
    print(f"\n" + "="*40)
    print(f" 🚀 START TRENINGU: {task_name.upper()}")
    print("="*40)
    
    # 1. Wybór urządzenia (GPU nvidia lub procesor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Urządzenie obliczeniowe: {device}")

    # 2. Dane
    dataset = ImageDataset(task_type=task_name, img_size=IMG_SIZE)
    if len(dataset) == 0:
        return # Przerywamy, jeśli brak danych
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Liczba zdjęć: {len(dataset)}")
    print(f"Liczba kroków na epokę: {len(dataloader)}")

    # 3. Model
    model = SimpleUNet().to(device)
    
    # 4. Narzędzia uczenia
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Główna pętla
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        # Pasek postępu
        progress = tqdm(dataloader, desc=f"Epoka {epoch+1}/{EPOCHS}", unit="batch")
        
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zerowanie starej wiedzy o błędach
            optimizer.zero_grad()

            # A. Sieć próbuje zgadnąć (Forward)
            outputs = model(inputs)

            # B. Liczymy jak bardzo się pomyliła (Loss)
            loss = criterion(outputs, targets)

            # C. Nauka (Backward)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        # print(f"Średni błąd w epoce {epoch+1}: {avg_loss:.6f}")

    # 6. Zapisywanie wytrenowanego modelu
    save_dir = Path('models')
    save_dir.mkdir(exist_ok=True)
    
    model_path = save_dir / f"model_{task_name}.pth"
    torch.save(model.state_dict(), model_path)
    
    print("\n" + "="*40)
    print(f"✅ TRENING ZAKOŃCZONY!")
    print(f"Model zapisano w: {model_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Tutaj decydujesz, co trenujesz. 
    # Na razie uruchomimy tylko SZUM (noise).
    
    train_task("noise")
    
    # Później odkomentujesz te linie:
    # train_task("blur")
    # train_task("low_res")