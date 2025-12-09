import torch
import torch.nn as nn

class DistortionClassifier(nn.Module):
    def __init__(self):
        super(DistortionClassifier, self).__init__()
        
        # Prosta sieć konwolucyjna (CNN)
        # Wejście: Obrazek zmniejszony do 64x64 (dla szybkości)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 8x8
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 3 wyjścia: Noise, Blur, LowRes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x