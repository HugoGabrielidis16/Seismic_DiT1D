import torch
import xgboost as xgb


def load_torchscript_model(
                path = "model/models_checkpoints/diffusion_model.pt",
                device="cpu"):
    model = torch.jit.load(path, map_location=device)
    model.to(device)  # Move model to the specified device
    model.eval()  # Set to evaluation mode
    return model

