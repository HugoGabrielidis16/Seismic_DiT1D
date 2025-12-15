import os
import torch
import time
import matplotlib.pyplot as plt
from util.metrics import MSE, snr,  calculate_seismic_ssim
from util.visualize import amplitude_graph, frequency_loglog
from util.load_models import load_torchscript_model 
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# DEMO_STEPS = 100 Our model has been compile to use 100 steps on it's inference
BATCH_SIZE = 15

path_dict = {
    "DiT1D" : "model/models_checkpoints/DiT1D.pt"
}


def load_samples(samples_folder = "test_samples/"):
    """
    Args:
        - samples folders: where the data are stored as pytorch tensors under the form of batch_i.pt files
    Returns:
        - dataloader containing the samples
        - each batch contain x (1Hz normalized), y (30Hz normalized), x_magnitude (PGA of 1Hz), y_magnitude (PGA of 30Hz)
    """
    cached_data = torch.load(os.path.join(samples_folder, "batch_0.pt"))
    x = cached_data['x']
    y = cached_data['y']
    x_magnitude = cached_data['x_magnitude']
    y_magnitude = cached_data['y_magnitude']
    dataset = torch.utils.data.TensorDataset(x, y, x_magnitude, y_magnitude)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def generate_sample(device = "cpu"):
    """
    Generate samples function from a pretrained model (Here we used our DiT1D model),
    it's use the sample from the test_samples (x being the normalized 1Hz freq & y normalized the 30Hz signal)
    You can also use your own samples if saved in the same way. Check gen
    """
    model_name = "DiT1D"  
    diffusion_model = load_torchscript_model(path=path_dict[model_name], device=device)

    
    path = f"generated_samples/{model_name}/experiment/"
    os.makedirs(path, exist_ok=True)

    dataloader = load_samples()
    for batch in dataloader:
        x,y,x_magnitude,y_magnitude = batch
        x = x.to(device)
        y = y.to(device)
        x_magnitude = x_magnitude.to(device)
        y_magnitude = y_magnitude.to(device)
        current_time = time.time()
        with torch.no_grad():
            generated = diffusion_model(x)
        elapsed_time = time.time() - current_time
        print(f"Generation time for batch of size {BATCH_SIZE} : {elapsed_time:.4f} seconds")
        for sample_idx, (x_sample, y_sample, generated_sample) in enumerate(zip(x, y, generated)):
            print(f"Sample {sample_idx}")
            print(f"MSE: {MSE(y_sample, generated_sample)}")
            print(f"SNR: {snr(y_sample, generated_sample)}")
            print(f"SSIM: {calculate_seismic_ssim(y_sample,generated_sample)}")
            amplitude_graph(y = y_sample, 
                            generate=generated_sample, 
                            x=x_sample,
                            x_magnitude=x_magnitude[sample_idx], 
                            y_magnitude=y_magnitude[sample_idx],
                            path=path, 
                            idx=sample_idx, 
                            show_x=False)
            frequency_loglog(y = y_sample, 
                               generate=generated_sample, 
                               x=x_sample, 
                               path=path, 
                               idx=sample_idx,
                               x_magnitude=x_magnitude[sample_idx],
                               y_magnitude=y_magnitude[sample_idx])
   
if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"GPU available : {cuda}")
    device = "cuda" if cuda else "cpu"
    generate_sample(device=device)





