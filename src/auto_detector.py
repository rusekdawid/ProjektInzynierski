import cv2
import torch
import numpy as np
import config as cfg
from classifier_model import DistortionClassifier

# Ładujemy model raz (globalnie), żeby było szybko
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = DistortionClassifier().to(device)
model_path = cfg.MODELS_DIR / 'classifier.pth'

model_loaded = False
if model_path.exists():
    try:
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        classifier.eval()
        model_loaded = True
    except:
        print("⚠️ Błąd ładowania klasyfikatora.")

def detect_distortion(image_path):
    """
    Używa sieci neuronowej do rozpoznania problemu.
    """
    if not model_loaded:
        return 'noise', "Błąd: Brak modelu klasyfikatora (uruchom train_classifier.py)"

    img = cv2.imread(str(image_path))
    if img is None: return None, "Błąd odczytu"

    # Przygotowanie obrazu dla sieci (musi być taki sam jak w treningu: 64x64)
    img_resized = cv2.resize(img, (64, 64))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        outputs = classifier(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    idx = predicted.item()
    conf = confidence.item() * 100
    
    labels = {0: 'noise', 1: 'blur', 2: 'low_res'}
    result = labels[idx]
    
    return result, f"Wykryto przez AI: {result.upper()} (Pewność: {conf:.1f}%)"