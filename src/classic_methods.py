import cv2
import numpy as np

class ClassicEnhancer:

    def denoising(self, image, method='nlm'):
        """
        Usuwanie szumu.
        Dostępne metody:
        - 'gaussian': Rozmycie Gaussa (najprostsze, ale traci detale).
        - 'median': Filtr medianowy (świetny na szum typu "sól i pieprz").
        - 'nlm': Non-Local Means (zaawansowany algorytm, standard w fotografii).
        """
        if method == 'gaussian':
            # Rozmywa szum, ale też krawędzie
            return cv2.GaussianBlur(image, (5, 5), 0)
        
        elif method == 'median':
            # Zastępuje piksel medianą z sąsiadów - zachowuje krawędzie lepiej
            return cv2.medianBlur(image, 5)
        
        elif method == 'nlm':
            # Non-Local Means Denoising - szuka podobnych łat w całym obrazie
            # h=10 (siła filtrowania), hColor=10, templateWindowSize=7, searchWindowSize=21
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        else:
            return image

    def deblurring(self, image):
        """
        Redukcja rozmycia poprzez wyostrzanie (Sharpening).
        Prawdziwe 'odwracanie' rozmycia (dekonwolucja) jest trudne bez znajomości jądra,
        więc klasycznie stosuje się maskę wyostrzającą (Unsharp Masking).
        """
        # Tworzymy jądro wyostrzające (kernel)
        # To macierz, która podbija różnice między pikselem środkowym a sąsiadami
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        # Nakładamy filtr (splot)
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    def super_resolution(self, image, scale_factor=4, method='bicubic'):
        """
        Zwiększanie rozdzielczości metodami interpolacji.
        """
        height, width = image.shape[:2]
        new_height = height * scale_factor
        new_width = width * scale_factor
        
        if method == 'bicubic':
            # Interpolacja dwusześcienna - standard w Photoshopie
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        elif method == 'lanczos':
            # Lanczos - zazwyczaj daje nieco ostrzejszy wynik niż bicubic
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
        else:
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# --- TESTOWANIE KODU (Jeśli masz jakiekolwiek jedno zdjęcie) ---
if __name__ == "__main__":
    # 1. Pobierz byle jakie zdjęcie z netu i nazwij je 'test.jpg'
    img = cv2.imread('test.jpg') 
    
    if img is not None:
        enhancer = ClassicEnhancer()
        
        # Test odszumiania (NLM jest wolny, ale dokładny)
        print("Testuję odszumianie...")
        denoised = enhancer.denoising(img, method='nlm')
        cv2.imwrite('test_denoised.jpg', denoised)
        
        # Test wyostrzania
        print("Testuję wyostrzanie...")
        sharp = enhancer.deblurring(img)
        cv2.imwrite('test_sharp.jpg', sharp)
        
        # Test powiększania
        print("Testuję skalowanie x4...")
        upscaled = enhancer.super_resolution(img, scale_factor=4)
        cv2.imwrite('test_upscaled.jpg', upscaled)
        
        print("Gotowe! Sprawdź pliki test_*.jpg w folderze.")
    else:
        print("Nie znaleziono pliku test.jpg - wrzuć coś, żeby przetestować kod.")