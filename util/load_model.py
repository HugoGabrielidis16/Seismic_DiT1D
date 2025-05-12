import torch

def load_torchscript_model(model_path="./deployed_model.pt", device="cpu"):
    """
    Load a TorchScript model for inference without original source code.
    
    Args:
        model_path: Path to the TorchScript model file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded TorchScript model
    """
    # Load the TorchScript model
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)  # Move model to the specified device
    model.eval()  # Set to evaluation mode
    return model
