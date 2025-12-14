import torch
import torch.nn as nn


class CNN(nn.Module):
    # VGG-style CNN with BatchNorm
    
    def __init__(
        self,
        input_channels=3,
        num_conv_layers=4,
        base_filters=32,
        fc_dim=512,
        num_classes=6,
        dropout=0.5
    ):
        super().__init__()
        
        # 64 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # 32 -> 16
        self.block2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # 16 -> 8
        self.block3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # 8 -> 4
        self.block4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        flatten_size = 4 * 4 * base_filters * 8
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.BatchNorm1d(fc_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, num_classes)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNN(base_filters=32, fc_dim=512, dropout=0.5)
    print(model)
    
    x = torch.randn(4, 3, 64, 64)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
