import torch
from ai_model import SRResNet
import config as cfg

# ŚCIEŻKA DO STAREGO MODELU
OLD_MODEL_PATH = cfg.MODELS_DIR / "model_low_resF.pth"  # tu daj właściwą nazwę

def main():
    device = cfg.DEVICE if hasattr(cfg, "DEVICE") else torch.device("cpu")
    print(f"Urządzenie: {device}")
    print(f"Ładuję wagi z: {OLD_MODEL_PATH}")

    model = SRResNet().to(device)

    # 1. Wczytanie stanu z pliku
    state = torch.load(OLD_MODEL_PATH, map_location=device)

    # 2. Próba normalna (strict=True)
    try:
        model.load_state_dict(state, strict=True)
        print("OK: model wczytany z strict=True")
    except Exception as e:
        print("BŁĄD przy strict=True:", e)

        # 3. Wersja awaryjna: strict=False (ignoruje brakujące/nadmiarowe klucze)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Załadowano z strict=False")
        print("Brakujące klucze:", missing)
        print("Nieoczekiwane klucze:", unexpected)

if __name__ == "__main__":
    main()
