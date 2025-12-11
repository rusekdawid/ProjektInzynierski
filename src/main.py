import sys
import os
import shutil # Do usuwania folderów

# Importujemy moduły logiczne
try:
    import config as cfg
    import data_generator
    import run_classic
    import train
    import predict
    import evaluate
except ImportError as e:
    print(f"Błąd importu: {e}")
    print("Upewnij się, że uruchamiasz skrypt z głównego katalogu projektu.")
    sys.exit(1)

def ask_for_task(action_name):
    print(f"\n--- Wybierz typ wady do {action_name} ---")
    print("a) Noise (Odszumianie)")
    print("b) Blur (Wyostrzanie)")
    print("c) Low Res (Super-Resolution)")
    
    choice = input("Twój wybór (a/b/c): ").lower().strip()
    mapping = {'a': 'noise', 'b': 'blur', 'c': 'low_res'}
    return mapping.get(choice, None)

def clean_all_data():
    # usuwa results i processed
    paths_to_clean = [
        cfg.PROCESSED_DIR, # data/processed
        cfg.RESULTS_DIR    # data/results
    ]
    
    for p in paths_to_clean:
        if p.exists():
            shutil.rmtree(p)
            print(f"Usunięto: {p}")
            
    print("Wyczyszczono. Możesz generować dane od nowa.")

def menu():
    print("PANEL STEROWANIA")
    print("="*50)
    print("1.Generuj dane")
    print("2.Uruchom metody klasyczne (OpenCV)")
    print("3.Trenuj model AI")
    print("4.Predykcja AI")
    print("5.Ewaluacja (PSNR/SSIM)")
    print("8.WYCZYŚĆ WSZYSTKIE DANE (Reset)")
    print("0. Wyjście")
    
    return input("\nWybierz opcję: ").strip()

def main():
    # Konfiguracja startowa
    cfg.BASE_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(os, 'system'): os.system('cls' if os.name == 'nt' else 'clear')

    while True:
        c = menu()
        
        if c == '1':
            data_generator.generate_all_datasets()
            
        elif c == '2':
            run_classic.run_classic_all()
            
        elif c == '3':
            task = ask_for_task("trenowania")
            if task: train.train_model(task) # Zakładam, że import train działa
            else: print("Błędny wybór.")

        elif c == '4':
            task = ask_for_task("przetwarzania (AI)")
            if task: predict.run_prediction(task)
            else: print("Błędny wybór.")
            
        elif c == '5':
            if hasattr(evaluate, 'run_evaluation'):
                evaluate.run_evaluation()
            else:
                os.system(f"{sys.executable} src/evaluate.py")

        elif c == '8':
            confirm = input("Czy na pewno usunąć wszystkie wygenerowane pliki i wyniki? (t/n): ")
            if confirm.lower() == 't':
                clean_all_data()
            else:
                print("Anulowano.")
        elif c == '0':
            sys.exit()

if __name__ == "__main__":
    main()