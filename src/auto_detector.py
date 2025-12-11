import cv2
import numpy as np

def detect_distortion(image_path_or_array):
    """
    Analizuje obraz i zwraca typ zniekształcenia.
    Ulepszona logika rozróżniania Szumu od Pikselozy.
    """
    
    # Obsługa wejścia (ścieżka lub obraz w pamięci)
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None: return None, "Błąd odczytu"

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Fizycznie mały obrazek -> Low Res (Bez dyskusji)
    if min(h, w) < 400:
        return 'low_res', f"Mała rozdzielczość fizyczna ({w}x{h})"

    # --- ANALIZA MATEMATYCZNA ---

    # A. Ostrość (Laplacian)
    # < 100: Rozmycie
    # > 500: Szum LUB Pikseloza (ostre krawędzie bloków)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # B. Płaskość (Flatness / Blockiness) - KLUCZ DO SUKCESU
    # Liczymy różnicę między pikselem a jego sąsiadem.
    # W pikselozie (Nearest Neighbor) sąsiedzi są często IDENTYCZNI (różnica = 0).
    # W szumie sąsiedzi są prawie zawsze RÓŻNI.
    
    diff_x = np.abs(gray[:, :-1].astype(int) - gray[:, 1:].astype(int))
    diff_y = np.abs(gray[:-1, :].astype(int) - gray[1:, :].astype(int))
    
    # Liczymy ile jest "zerowych" przejść (idealnie płaskich)
    zeros = np.count_nonzero(diff_x == 0) + np.count_nonzero(diff_y == 0)
    total_checks = diff_x.size + diff_y.size
    
    # Wskaźnik płaskości (0.0 - 1.0)
    flatness_ratio = zeros / total_checks

    # --- DRZEWO DECYZYJNE ---

    # KROK 1: Czy to BLUR?
    # Jeśli obraz nie ma krawędzi, to jest rozmyty.
    if laplacian_var < 150:
        return 'blur', f"Wykryto rozmycie (Ostrość: {laplacian_var:.1f})"

    # KROK 2: Czy to LOW RES (Pikseloza)?
    # Jeśli obraz jest ostry (nie Blur), ale ma dużo identycznych sąsiadów -> to są bloki!
    # Normalne zdjęcie ma flatness ok. 0.10 - 0.20.
    # Zaszumione zdjęcie ma flatness < 0.05.
    # Pikseloza (skala 4+) ma flatness > 0.35.
    if flatness_ratio > 0.30:
        return 'low_res', f"Wykryto pikselozę/bloki (Płaskość: {flatness_ratio:.2f})"

    # KROK 3: Domyślnie NOISE
    # Obraz jest ostry (wysoki Laplacian) i chaotyczny (niska płaskość).
    return 'noise', f"Wykryto szum (Wysoka wariancja, brak bloków)"