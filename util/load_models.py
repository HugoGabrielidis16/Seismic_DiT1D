import torch


def load_torchscript_model(
                path = "model/models_checkpoints/DiT1D.pt",
                device="cpu"):
    model = torch.jit.load(path, map_location=device)
    model.to(device) 
    model.eval()
    return model

def load_PGA_model(
                path = "model/models_checkpoints/CNNLSTM.pt",
                device="cpu"):
    model = torch.jit.load(path, map_location=device)
    model.to(device) 
    model.eval() 
    print("PGA model loaded successfully.")
    return model