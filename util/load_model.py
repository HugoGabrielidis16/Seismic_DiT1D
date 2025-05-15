import torch
import xgboost as xgb
from PGA_predictor import CNNLSTM_PGA

XGBBOOST_MODEL_PATH = "model_checkpoint/xgboost.model"
CNNLSTM_PGA_PATH = "model_checkpoint/CNNLSTM_PGA.pth"
DIFFUSION_MODEL_PATH = "model_checkpoint/diffusion_model.pth"

def load_torchscript_model(device="cpu"):
    """
    Load a TorchScript model for inference without original source code.
    
    Args:
        model_path: Path to the TorchScript model file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded TorchScript model
    """
    # Load the TorchScript model
    model = torch.jit.load(DIFFUSION_MODEL_PATH, map_location=device)
    model.to(device)  # Move model to the specified device
    model.eval()  # Set to evaluation mode
    print("Model structure:")
    print(model)
    
    # Method 2: Extract graph structure (more detailed)
    graph = model.inlined_graph
    print("\nGraph structure:")
    print(graph)
    
    # Method 3: Get parameters and buffers
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    print("\nModel buffers:")
    for name, buffer in model.named_buffers():
        print(f"{name}: {buffer.shape}")
    
    return model

def load_pga_model(model_name = "XGBoost",
                   device = "cpu",
                   *args,**kwargs):
    if model_name == "XGBoost":
        model = xgb.Booster()
        model.load_model(XGBBOOST_MODEL_PATH)
    elif model_name== "CNNLSTM":
        model = CNNLSTM_PGA(
            in_channels=3,
            out_channels=3,
            num_layers=2,
            hidden_size=64,
            kernel_size=3,
            dropout=0.1,
            bidirectional=True)
        model = model.load_state_dict(CNNLSTM_PGA_PATH, strict=False)
        model.to(device)
        model.eval()
    return model
