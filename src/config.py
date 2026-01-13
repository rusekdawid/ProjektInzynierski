from pathlib import Path
import torch

#ŚCIEŻKI
BASE_DIR = Path('data')
RAW_DIR = BASE_DIR / 'val_raw'
PROCESSED_DIR = BASE_DIR / 'val_processed'
RESULTS_DIR = BASE_DIR / 'val_results'
MODELS_DIR = Path('models')

# KONFIGURACJA SPRZĘTOWA 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2 if torch.cuda.is_available() else 0  # Przyspiesza ładowanie danych w train.py
PIN_MEMORY = True if torch.cuda.is_available() else False

# PARAMETRY 
NUM_IMAGES = 10 
NOISE_LEVEL = 10         
BLUR_KERNEL = 3          
SCALE_FACTOR = 2        

#TRENING 
IMG_SIZE = 128           
BATCH_SIZE = 16          
EPOCHS = 40              
LEARNING_RATE = 0.0002