import os
from pathlib import Path

# Gdzie ja jestem?
current_dir = Path.cwd()
print(f"1. Twój terminal jest teraz w folderze:\n   -> {current_dir}")

# Gdzie szukam danych?
data_raw = current_dir / 'data' / 'raw'
print(f"\n2. Szukam zdjęć w folderze:\n   -> {data_raw}")

if data_raw.exists():
    print("\n3. SUKCES: Folder data/raw istnieje!")
    # Zobaczmy co jest w środku (pierwsze 5 plików/folderów)
    items = list(data_raw.iterdir())
    print(f"   Widzę w środku {len(items)} elementów.")
    if len(items) > 0:
        print(f"   Przykładowe elementy: {[x.name for x in items[:5]]}")
        
        # Sprawdźmy czy rglob coś widzi
        pngs = list(data_raw.rglob('*.png'))
        jpgs = list(data_raw.rglob('*.jpg'))
        print(f"\n4. Test szukania (rglob):")
        print(f"   Znaleziono plików .png: {len(pngs)}")
        print(f"   Znaleziono plików .jpg: {len(jpgs)}")
    else:
        print("   OSTRZEŻENIE: Folder jest pusty!")
else:
    print("\n3. BŁĄD: Folder data/raw NIE ISTNIEJE w tej ścieżce.")
    print("   Prawdopodobnie musisz przenieść folder 'data' lub zmienić katalog w terminalu.")