Celem jest porównanie klasycznych alogyrtmów przetwarzania obrazu z metodami opartymi na siecian neuronowych i ocenie
uwzględniamy 32 głownye przypadki
usuwanie szumu, redukcja rozymcia oraz zwiększenie rozdzielczości obrazów
tradycyjne techniki oraz głębokiego uczenia
system wykryje znieksztyałcenie oraz oraz zastosuje odpowiednią metode poprawy jakości 
'


python -m venv venv
venv\Scripts\activate


Biblioteki:

numpy macierze
matplotlib wykresy, podgląd wyników
OpenCV-python   najważszniejsza w projekcie
Scikit-image (skimage + metrics) opcjonalna, zawiera gotowe funkcjie do oceny jakości obrazu
pytourch + tourchvision
streamlit   interface


1. NOISE (Szum Gaussa)
Parametr: NOISE_LEVEL (Odchylenie standardowe)
15 (Łatwy / Treningowy): Tu Twój model powinien mieć ~30 dB. Obraz wygląda prawie idealnie.
30 (Średni): Wyraźne ziarno. Tu AI powinno nadal wygrywać z klasyką, ale wynik spadnie do ok. 26-27 dB.
50 (Trudny): Bardzo mocny szum, detale giną. To jest prawdziwy test inteligencji modelu.
80 (Ekstremalny): Obraz wygląda jak "śnieg" w starym telewizorze. Jeśli AI wyciągnie z tego cokolwiek rozpoznawalnego, to sukces.
2. BLUR (Rozmycie)
Parametr: BLUR_KERNEL (Wielkość jądra - musi być liczbą nieparzystą!)
5 (Łatwy): Lekkie zmiękczenie. AI powinno przywrócić idealną ostrość.
9 (Średni / Treningowy): Wyraźne rozmycie. Tu walczysz o wynik w okolicach 29 dB.
15 (Trudny): Obraz wygląda jak za mgłą. Metody klasyczne zaczną tu tworzyć brzydkie obwódki (halo), AI powinno dać gładszy obraz.
21 (Ekstremalny): Bardzo silne rozmycie, małe obiekty znikają całkowicie. Trudne do odratowania.
3. LOW RES (Skalowanie)
Parametr: SCALE_FACTOR (Krotność pomniejszenia)
2 (Łatwy): Obraz zmniejszony o połowę. Tu klasyka (Lanczos) jest bardzo mocna (~31 dB), AI musi walczyć o ostrość krawędzi.
4 (Standard / Treningowy): Standard w badaniach Super Resolution. Tu AI powinno zacząć wygrywać wizualnie (lepsze detale).
6 (Trudny): Obraz jest bardzo mały. Po powiększeniu widać "pikselozę" lub "mydło".
8 (Ekstremalny): Z 1000 pikseli robi się 125. To już prawie abstrakcja. Jeśli AI odgadnie kształty, to jest świetnie.



//////////////////// Wyniki + config na którym był trenowany model bazowo


==================================================
 📊 RAPORT KOŃCOWY (PSNR / SSIM)
==================================================
ZADANIE    | METODA     | PSNR (dB)  | SSIM      
--------------------------------------------------
   [CLASSIC] Liczenie dla: noise (59 plików)...
   [AI] Liczenie dla: noise (300 plików)...                                                                                                                      
noise      | Classic    | 28.83       | 0.7799                                                                                                                   
noise      | AI (Ty)    | 30.65       | 0.7980
--------------------------------------------------
   [CLASSIC] Liczenie dla: blur (300 plików)...
   [AI] Liczenie dla: blur (300 plików)...                                                                                                                       
blur       | Classic    | 29.61       | 0.8229                                                                                                                   
blur       | AI (Ty)    | 31.04       | 0.8539
--------------------------------------------------
   [CLASSIC] Liczenie dla: low_res (300 plików)...
   [AI] Liczenie dla: low_res (300 plików)...                                                                                                                    
low_res    | Classic    | 31.56       | 0.8720                                                                                                                   
low_res    | AI (Ty)    | 31.26       | 0.8697
--------------------------------------------------

✅ Zapisano szczegółowy raport do: data\results\metrics.json


# --- ŚCIEŻKI ---
BASE_DIR = Path('data')
RAW_DIR = BASE_DIR / 'raw'
PROCESSED_DIR = BASE_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = Path('models')

# --- PARAMETRY JAKOŚCIOWE ---
NUM_IMAGES = 300         # Optymalna liczba do szybkiego treningu z augmentacją
NOISE_LEVEL = 15         # Nieco mniejszy szum = łatwiej uzyskać wysoki PSNR
BLUR_KERNEL = 7          # Mniejsze rozmycie
SCALE_FACTOR = 2         # Skala x2

# --- TRENING ---
IMG_SIZE = 128           # Mniejsze kafelki = szybsze epoki i mniejsze zużycie VRAM
BATCH_SIZE = 16          # Stabilny batch
EPOCHS = 40              # Wystarczająco przy augmentacji
LEARNING_RATE = 0.0002   # Mniejszy LR dla precyzji (ważne dla PSNR!)


///////////////////////////////



Wybierz opcję: 5

==================================================
 📊 RAPORT KOŃCOWY (PSNR / SSIM)
==================================================
ZADANIE    | METODA     | PSNR (dB)  | SSIM      
--------------------------------------------------
   [CLASSIC] Liczenie dla: noise (300 plików)...
   [AI] Liczenie dla: noise (300 plików)...                                                                                                                      
noise      | Classic    | 26.38       | 0.6532                                                                                                                   
noise      | AI (Ty)    | 25.40       | 0.4972
--------------------------------------------------
   [CLASSIC] Liczenie dla: blur (300 plików)...
   [AI] Liczenie dla: blur (300 plików)...                                                                                                                       
blur       | Classic    | 28.45       | 0.7937                                                                                                                   
blur       | AI (Ty)    | 29.56       | 0.8280
--------------------------------------------------
   [CLASSIC] Liczenie dla: low_res (300 plików)...
   [AI] Liczenie dla: low_res (300 plików)...                                                                                                                    
low_res    | Classic    | 26.03       | 0.7412                                                                                                                   
low_res    | AI (Ty)    | 26.10       | 0.7452
--------------------------------------------------

✅ Zapisano szczegółowy raport do: data\results\metrics.json

==============================
 🎛️  PANEL STEROWANIA PROJEKTEM
==============================
1. 🏭 Generuj dane (Noise, Blur, LowRes)
2. 🏛️  Uruchom metody klasyczne