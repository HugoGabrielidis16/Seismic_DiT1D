import torch

def normalize(x,y, normalization = "min_max", eps = 0.0001,**kwargs):
    """
    Normalize the input and output tensors based on the specified normalization method.
    """

    if normalization == "max":
        x_magnitude = torch.max(torch.abs(x), dim=1).values
        y_magnitude = torch.max(torch.abs(y), dim=1).values
        x = x / x_magnitude.view(3,1)
        y = y / y_magnitude.view(3,1)
    elif normalization == "standard":
        eps = 1e-8
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + eps
        x = (x - x_mean) / x_std
        y_mean = y.mean(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True) + eps
        y = (y - y_mean) / y_std
    elif normalization == "min_max":
        min_val_x = x.min(dim=1, keepdim=True).values
        max_val_x = x.max(dim=1, keepdim=True).values
        x_normalized = 2 * (x - min_val_x) / (max_val_x - min_val_x + eps) - 1  # Maps to [-1, 1]
        x = x_normalized
        min_val_y = y.min(dim=1, keepdim=True).values
        max_val_y = y.max(dim=1, keepdim=True).values
        y_normalized = 2 * (y - min_val_y) / (max_val_y - min_val_y + eps) - 1
        y = y_normalized
    elif normalization =="by_y":
        y_magnitude_scalar = torch.max(torch.abs(y)) # Finds ONE max value across ALL 3x6000 elements. Shape is scalar [].
        y_magnitude_scalar = torch.clamp(y_magnitude_scalar, min=1e-9)
        x = x / y_magnitude_scalar 
        y = y / y_magnitude_scalar 

    elif normalization == "by_x":
        x_magnitude = torch.max(torch.abs(x), dim=1).values
        x = x / x_magnitude.view(3,1)
        y = y / x_magnitude.view(3,1)
    elif normalization == "use_statistics":
        x_mean = kwargs.get("x_mean")
        x_std = kwargs.get("x_std")
        y_mean = kwargs.get("y_mean")
        y_std = kwargs.get("y_std")
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std
    elif normalization == "none":
        pass
    else:
        raise ValueError("Normalization type not recognized")
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    #x = x.clip(-1, 1)
    #y = y.clip(-1, 1)

    if (len(x.shape) == 2) and (len(y.shape) == 2):
        return x, y
    elif (len(x.shape) == 3) and (len(y.shape) == 3):
        x = x.squeeze(0)
        y = y.squeeze(0)
        return x, y 
