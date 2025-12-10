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
        
        self.conv_input = nn.Conv2d(num_channels, num_filters, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()
        
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlockSR(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv_mid = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_filters)
        
        self.conv_output = nn.Conv2d(num_filters, num_channels, kernel_size=9, padding=4)

        # --- OPTYMALIZACJA: Inicjalizacja wag (Kaiming) ---
        # Pomaga modelowi szybciej zbiegać na początku treningu
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_first = self.prelu(self.conv_input(x))
        out_res = self.res_blocks(out_first)
        out_mid = self.bn_mid(self.conv_mid(out_res))
        out = out_first + out_mid
        out = self.conv_output(out)
        return x + out