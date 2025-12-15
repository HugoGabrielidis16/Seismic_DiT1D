import os
import torch
import time
import matplotlib.pyplot as plt
from util.metrics import MSE, snr,  calculate_seismic_ssim
from util.visualize import amplitude_graph, frequency_loglog
from util.load_models import load_torchscript_model, load_PGA_model
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# DEMO_STEPS = 100 Our model has been compile to use 100 diffusion steps on it's inference
BATCH_SIZE = 10



def load_samples(samples_folder = "test_samples/"):
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
    You can also use your own samples if saved in the same way.
    """
    model_name = "DiT1D"  
    diffusion_model = load_torchscript_model(device=device)
    pga_model = load_PGA_model(device=device)
    
    path = f"generated_samples/{model_name}_withpga"
    os.makedirs(path, exist_ok=True)
    print(f"Saving samples at {path}")


    dataloader = load_samples()
    print(f"DataLoader loaded with {len(dataloader)} batches")

    for batch_idx,batch in enumerate(dataloader):
        x_norm,y_norm,x_magnitude,y_magnitude = batch
        x_norm = x_norm.to(device)
        y_norm = y_norm.to(device)
        x_magnitude = x_magnitude.to(device)
        y_magnitude = y_magnitude.to(device)
        current_time = time.time()
        x = x_norm * x_magnitude.view(-1,3,1)
        y = y_norm * y_magnitude.view(-1,3,1)

        # x_norm is used for the diffusion model generation
        # y_norm is used to get the diffusion model metrics
        # x not normalized is used as the condition for the pga_model
     
        with torch.no_grad():
            generated = diffusion_model(x_norm)
            pga_predicted = pga_model(x,generated)
        elapsed_time = time.time() - current_time
        print(f"Generation time for batch of size {BATCH_SIZE} : {elapsed_time:.4f} seconds")
     
        for sample_idx, (x_sample, y_sample, generated_sample) in enumerate(zip(x, y, generated)):
            print(f"Sample {sample_idx}")
            print(f"MSE: {MSE(y_sample, generated_sample)}")
            print(f"SNR: {snr(y_sample, generated_sample)}")
            print(f"SSIM: {calculate_seismic_ssim(y_sample,generated_sample)}")
            print(f"PGA difference: {torch.abs(pga_predicted[sample_idx] - y_magnitude[sample_idx]).mean()}")
            print("\n \n")
            generated_sample = generated_sample * pga_predicted[sample_idx].view(3,1)
            amplitude_graph(y = y_sample, generate=generated_sample, x=x_sample,path=path, x_magnitude=None,idx=batch_idx * BATCH_SIZE + sample_idx, normalized=False, show_x=False)
            frequency_loglog(y = y_sample, generate=generated_sample, x=x_sample, path=path, y_magnitude=None,idx=batch_idx * BATCH_SIZE + sample_idx, normalized= False)
        break  
   
   
if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"GPU available : {cuda}")
    device = "cuda" if cuda else "cpu"
    generate_sample(device=device)





