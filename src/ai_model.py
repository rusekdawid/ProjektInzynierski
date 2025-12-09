import torch
import torch.nn as nn

class ResidualBlockSR(nn.Module):
    def __init__(self, channels):
        super(ResidualBlockSR, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class SRResNet(nn.Module):
    def __init__(self, num_channels=3, num_blocks=16, num_filters=64):
        super(SRResNet, self).__init__()
        
        # 1. Wejście
        self.conv_input = nn.Conv2d(num_channels, num_filters, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        # 2. Blok Resztkowy (16 głębokich warstw)
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlockSR(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 3. Połączenie środkowe
        self.conv_mid = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_filters)
        
        # 4. Wyjście
        self.conv_output = nn.Conv2d(num_filters, num_channels, kernel_size=9, padding=4)

    def forward(self, x):
        # x to obraz wejściowy (zszumiony/rozmyty/powiększony)
        out_first = self.prelu(self.conv_input(x))
        out_res = self.res_blocks(out_first)
        out_mid = self.bn_mid(self.conv_mid(out_res))
        out = out_first + out_mid
        out = self.conv_output(out)
        
        # GLOBAL SKIP CONNECTION
        # Model uczy się tylko poprawki (różnicy), a nie całego obrazu.
        # To klucz do wysokich wyników w Noise i Blur.
        return x + out

# Zostawiamy pustą klasę SimpleUNet tylko po to, żeby stare skrypty nie wywaliły błędu importu,
# ale nie będziemy jej już używać.
class SimpleUNet(nn.Module):
    def __init__(self): super(SimpleUNet, self).__init__()