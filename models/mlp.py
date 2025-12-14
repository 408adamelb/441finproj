import torch
import torch.nn as nn


class MLP(nn.Module):
    # MLP with BatchNorm + Dropout
    
    def __init__(
        self,
        input_dim=64 * 64 * 3,
        hidden_dim=512,
        num_hidden_layers=3,
        num_classes=6,
        dropout=0.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        layers = []
        
        # input -> hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        
        # hidden -> output
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


if __name__ == "__main__":
    model = MLP(hidden_dim=512, num_hidden_layers=3, dropout=0.5)
    print(model)
    
    x = torch.randn(4, 3, 64, 64)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
