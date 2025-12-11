import cv2
import numpy as np

class ClassicEnhancer:

    def denoising(self, image):
        
        # Parametry: h=10 (siła), hColor=10, templateWindowSize=7, searchWindowSize=21
        return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

    def deblurring(self, image):
       
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
        
        sharp = cv2.filter2D(image, -1, kernel)

        # Mieszanie: 70% wyostrzonego + 30% oryginału
        out = cv2.addWeighted(sharp, 0.7, image, 0.3, 0)
        return out

    def super_resolution(self, image, scale_factor=4):
        
        # Zwiększanie rozdzielczości standardową metodą Bicubic (Interpolacja Dwusześcienna).
        
        height, width = image.shape[:2]
        new_height = height * scale_factor
        new_width = width * scale_factor
        
        # INTER_CUBIC to standard przemysłowy (np. Photoshop)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
