import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # --- 1. ENKODER (Zmniejszanie - analiza obrazu) ---
        # Wejście: 3 kanały (RGB) -> Wyjście: 64 cechy
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2) # Zmniejsza 2x (np. 256 -> 128)

        # 64 -> 128 cech
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2) # Zmniejsza 2x (np. 128 -> 64)

        # --- 2. BOTTLENECK (Środek - najgłębsze cechy) ---
        self.center = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # --- 3. DEKODER (Powiększanie - naprawa obrazu) ---
        
        # W górę 1
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 128 z dołu + 128 z boku
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # W górę 2
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # 64 z dołu + 64 z boku
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # --- 4. WYJŚCIE (Powrót do obrazka RGB) ---
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Droga w dół
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Środek
        c = self.center(p2)
        
        # Droga w górę (z łączeniem skip connections)
        u2 = self.up2(c)
        merge2 = torch.cat([u2, e2], dim=1) # Doklejamy informację z enkodera
        d2 = self.dec2(merge2)
        
        u1 = self.up1(d2)
        merge1 = torch.cat([u1, e1], dim=1) # Doklejamy
        d1 = self.dec1(merge1)
        
        return self.final(d1)

if __name__ == "__main__":
    # Test czy kod działa
    model = SimpleUNet()
    print("✅ Model U-Net zdefiniowany poprawnie.")