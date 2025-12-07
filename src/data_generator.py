import cv2
import numpy as np
import json
import shutil  # <--- Do usuwania folderów
from pathlib import Path
from tqdm import tqdm

# --- PANEL STEROWANIA ---
NUM_IMAGES = 300          # <--- TU WPISUJESZ ILE ZDJĘĆ CHCESZ (np. 10, 50, 800)
SETTINGS = {
    "noise_severity": 30,    # Siła szumu (np. 15 słaby, 30 średni, 50 mocny)
    "blur_kernel": 15,       # Rozmycie (np. 5 słabe, 15 średnie, 25 mocne)
    "scale_factor": 4        # Pikseloza (np. 4, 8)
}
# ------------------------

class ImageDegrader:
    def __init__(self, output_dir='data/processed'):
        self.output_dir = Path(output_dir)
        
        # 1. AUTOMATYCZNE CZYSZCZENIE
        # Jeśli folder już istnieje, usuwamy go w całości razem ze starymi plikami
        if self.output_dir.exists():
            print(f"Czyszczenie starego folderu: {self.output_dir}...")
            shutil.rmtree(self.output_dir)
        
        # 2. Tworzenie czystych folderów
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'noise').mkdir(exist_ok=True)
        (self.output_dir / 'blur').mkdir(exist_ok=True)
        (self.output_dir / 'low_res').mkdir(exist_ok=True)

    def save_settings(self):
        settings_path = self.output_dir / 'parameters.json'
        with open(settings_path, 'w') as f:
            json.dump(SETTINGS, f, indent=4)

    def add_noise(self, image, severity):
        row, col, ch = image.shape
        mean = 0
        sigma = severity
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def add_blur(self, image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def downsample(self, image, scale_factor):
        height, width = image.shape[:2]
        new_height, new_width = height // scale_factor, width // scale_factor
        small_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pixelated_img = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated_img

    def process_dataset(self, source_dir):
        source_path = Path(source_dir)
        print(f"Szukam plików w: {source_path.absolute()}...")
        
        files = list(source_path.rglob('*.[jJ][pP][gG]')) + list(source_path.rglob('*.[pP][nN][gG]'))
        
        # --- KONTROLA LICZBY ZDJĘĆ ---
        if len(files) > NUM_IMAGES:
            files = files[:NUM_IMAGES]
        # -----------------------------

        print(f"Wybrano {len(files)} obrazów do przetworzenia.")
        print(f"Parametry: {SETTINGS}")

        self.save_settings()

        for file_path in tqdm(files):
            img = cv2.imread(str(file_path))
            if img is None: continue

            # 1. Szum
            noisy_img = self.add_noise(img, severity=SETTINGS["noise_severity"])
            cv2.imwrite(str(self.output_dir / 'noise' / file_path.name), noisy_img)

            # 2. Rozmycie
            blurred_img = self.add_blur(img, kernel_size=SETTINGS["blur_kernel"])
            cv2.imwrite(str(self.output_dir / 'blur' / file_path.name), blurred_img)

            # 3. Pikseloza
            low_res_img = self.downsample(img, scale_factor=SETTINGS["scale_factor"])
            cv2.imwrite(str(self.output_dir / 'low_res' / file_path.name), low_res_img)

if __name__ == "__main__":
    degrader = ImageDegrader(output_dir='data/processed')
    degrader.process_dataset(source_dir='data/raw')
    print("Zakończono generowanie danych! Stare pliki usunięte.")