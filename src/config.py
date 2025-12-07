from pathlib import Path

# --- ŚCIEŻKI ---
BASE_DIR = Path('data')
RAW_DIR = BASE_DIR / 'raw'
PROCESSED_DIR = BASE_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = Path('models')

# --- PARAMETRY PSUCIA DANYCH ---
NUM_IMAGES = 100         # Na ilu zdjęciach pracujemy (im więcej, tym mądrzejsze AI)
NOISE_LEVEL = 30         # Siła szumu (0-255)
BLUR_KERNEL = 11         # Siła rozmycia (musi być nieparzysta)
SCALE_FACTOR = 4         # Siła pikselozy

# --- PARAMETRY TRENINGU AI ---
IMG_SIZE = 256           # Wielkość wycinka (crop) do nauki
BATCH_SIZE = 8           # Ile wycinków naraz (zmniejsz do 4 jak braknie pamięci)
EPOCHS = 60              # Czas nauki (60 epok przy 100 zdjęciach wystarczy na start)
LEARNING_RATE = 0.001    # Szybkość nauki

# --- PARAMETRY WNIOSKOWANIA (UŻYCIA) ---
TILE_SIZE = 512          # Wielkość kafla przy naprawianiu dużych zdjęć