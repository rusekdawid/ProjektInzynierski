import sys
import data_generator
import run_classic
import train
import predict
import os

def menu():
    print("\n" + "="*30)
    print(" 🎛️  PANEL STEROWANIA PROJEKTEM")
    print("="*30)
    print("1. 🏭 Generuj dane (Noise, Blur, LowRes)")
    print("2. 🏛️  Uruchom metody klasyczne")
    print("3. 🚀 Trenuj model AI (Wybierz zadanie)")
    print("4. 🔮 Generuj wyniki AI (Predict)")
    print("5. 📊 Ewaluacja (Oblicz PSNR/SSIM)")
    print("0. Wyjście")
    
    choice = input("\nWybierz opcję: ")
    return choice

if __name__ == "__main__":
    while True:
        c = menu()
        
        if c == '1':
            data_generator.generate_all_datasets()
            
        elif c == '2':
            run_classic.run_classic_all()
            
        elif c == '3':
            print("\nCo trenować?")
            print("a) Noise")
            print("b) Blur")
            print("c) Low Res")
            t = input("Wybór: ").lower()
            task_map = {'a': 'noise', 'b': 'blur', 'c': 'low_res'}
            if t in task_map:
                train.train_model(task_map[t])
            else:
                print("Błędny wybór.")

        elif c == '4':
            print("\nCo przetworzyć?")
            print("a) Noise")
            print("b) Blur")
            print("c) Low Res")
            t = input("Wybór: ").lower()
            task_map = {'a': 'noise', 'b': 'blur', 'c': 'low_res'}
            if t in task_map:
                predict.run_prediction(task_map[t])
        
        elif c == '5':
            # Uruchamiamy Twój istniejący skrypt evaluate.py
            os.system("python src/evaluate.py")
            
        elif c == '0':
            sys.exit()