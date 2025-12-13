import torch
import xgboost as xgb


def load_torchscript_model(
                path = "model_checkpoint/diffusion_model.pt",
                device="cpu"):
    model = torch.jit.load(path, map_location=device)
    model.to(device)  # Move model to the specified device
    model.eval()  # Set to evaluation mode
    print(f"Using device: {device}")
    print("Model structure:")
    print(model)
    return model
    ## Method 2: Extract graph structure (more detailed)
    #graph = model.inlined_graph
    #print("\nGraph structure:")
    #print(graph)
    #
    ## Method 3: Get parameters and buffers
    #print("\nModel parameters:")
    #for name, param in model.named_parameters():
    #    print(f"{name}: {param.shape}")
    #
    #print("\nModel buffers:")
    #for name, buffer in model.named_buffers():
    #    print(f"{name}: {buffer.shape}")
    #
    #return model

