from pathlib import Path

# --- ŚCIEŻKI ---
BASE_DIR = Path('data')
RAW_DIR = BASE_DIR / 'raw'
PROCESSED_DIR = BASE_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = Path('models')

# --- PARAMETRY JAKOŚCIOWE ---
NUM_IMAGES = 300         # Optymalna liczba do szybkiego treningu z augmentacją
NOISE_LEVEL = 30         # Nieco mniejszy szum = łatwiej uzyskać wysoki PSNR
BLUR_KERNEL = 9          # Mniejsze rozmycie
SCALE_FACTOR = 4         # Skala x2

# --- TRENING ---
IMG_SIZE = 128           # Mniejsze kafelki = szybsze epoki i mniejsze zużycie VRAM
BATCH_SIZE = 16          # Stabilny batch
EPOCHS = 40              # Wystarczająco przy augmentacji
LEARNING_RATE = 0.0002   # Mniejszy LR dla precyzji (ważne dla PSNR!)