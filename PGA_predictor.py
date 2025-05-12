import torch
import torch.nn as nn
import torch.nn.functional as F
from util.metrics import MSE

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")  


class ApplyNorm(nn.Module):
    """
    According to Facebook Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, dims,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dims = dims
        self.norm = nn.LayerNorm(dims)
    
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.norm(x)
        x = x.permute(0,2,1)
        return x


class ResNETBlock1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation: str = "elu",
                 dropout_rate: float = 0.2,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Add bottleneck architecture for efficiency
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, 
                              out_channels=self.out_channels, 
                              kernel_size=1,  # 1x1 conv for dimension reduction
                              stride=1)
        self.norm1 = ApplyNorm(self.out_channels)
        self.act1 = get_activation(activation)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Main 3x3 convolution
        self.conv2 = nn.Conv1d(in_channels=self.out_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.norm2 = ApplyNorm(self.out_channels)
        self.act2 = get_activation(activation)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Skip connection handling
        self.skip = nn.Conv1d(in_channels=self.in_channels,
                             out_channels=self.out_channels,
                             kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        
        out += identity  # Skip connection
        out = self.act2(out)  # Final activation after skip connection
        
        return out


class CNNLSTM_PGA(nn.Module):
    def __init__(self,
                input_size = 6000,
                input_channels = 3,
                num_layers = 4,
                dropout_rate = 0.2,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.input_size = input_size
        self.input_channels = input_channels
        
        # CNN feature extraction
        self.conv1 = ResNETBlock1d(in_channels=input_channels, out_channels=32, dropout_rate=dropout_rate)
        self.conv2 = ResNETBlock1d(in_channels=32, out_channels=64, dropout_rate=dropout_rate)
        self.conv3 = ResNETBlock1d(in_channels=64, out_channels=128, dropout_rate=dropout_rate)
        
        # More efficient pooling strategy
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Batch normalization before LSTM
        self.batch_norm = nn.BatchNorm1d(128)
        
        # Bidirectional LSTM layers with residual connections
        self.lstm1 = nn.LSTM(input_size=128,  # Reduced size due to more pooling
                            hidden_size=512,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate)
        
        self.lstm2 = nn.LSTM(input_size=1024,  # 512*2 due to bidirectional
                            hidden_size=256,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate)
        
        # Final prediction layers
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 3)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.pool2(x)
        
        x = self.batch_norm(x)
        
        # Reshape for LSTM
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        
        # LSTM processing with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)  # Residual connection
        
        # Self-attention mechanism
        
        # Global average pooling
        out = torch.mean(lstm2_out, dim=1)
        
        # Final prediction
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

    def training_step(self, x, pga, device, *args, **kwargs):
        x = x.to(device)
        pga = pga.to(device)
        out = self(x)
        loss = MSE(out, pga)
        return loss
    
    def test_step(self, x, pga, device, *args, **kwargs):
        x = x.to(device)
        pga = pga.to(device)
        out = self(x)
        loss = MSE(out, pga)
        return loss

if __name__ == "__main__":
    from torchsummary import summary
    from time import time
    model = CNNLSTM_PGA(input_size=6000)
    x = torch.randn(2,3,6000)
    pred = model(x)
    print(pred.shape)
