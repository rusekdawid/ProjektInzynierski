import matplotlib.pyplot as plt
import json
import config as cfg
import sys

def plot_learning_curve(task_name):
    path = cfg.RESULTS_DIR / f'history_{task_name}.json'
    if not path.exists():
        print(f"❌ Brak historii dla {task_name}. Uruchom trening najpierw.")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    train_loss = data['train_loss']
    val_loss = data['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title(f'Krzywa uczenia: {task_name.upper()}')
    plt.xlabel('Epoki')
    plt.ylabel('Loss (Błąd)')
    plt.legend()
    plt.grid(True)
    
    out_file = cfg.RESULTS_DIR / f'plot_{task_name}.png'
    plt.savefig(out_file)
    print(f"📊 Wykres zapisano jako {out_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_learning_curve(sys.argv[1])
    else:
        print("Podaj nazwę zadania, np: python plot_history.py noise")