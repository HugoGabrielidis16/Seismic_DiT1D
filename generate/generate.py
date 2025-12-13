import os
import torch
import time
import matplotlib.pyplot as plt
from util.metrics import MSE, snr,  calculate_seismic_ssim
from util.visualize import amplitude_graph, frequency_loglog
from util.load_model import load_torchscript_model 
import matplotlib

import xformers.ops

xformers.ops.disable_flash_attention()
xformers.ops.disable_mem_efficient_attention()

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

# DEMO_STEPS = 100 Our model has been compile to use 100 steps on it's inference
BATCH_SIZE = 1

path_dict = {
    "DiT1D" : "model/models_checkpoints/DiT1D_100steps.pt"
}

def load_model(model_name = "DiT1D",
               device = "cpu",
               *args, **kwargs):
   path = path_dict[model_name]
   model = load_torchscript_model(path=path, device=device)
   return model

def load_samples(samples_folder = "test_samples/"):
    cached_data = torch.load(os.path.join(samples_folder, "samples.pt"))
    x = cached_data['x']
    y = cached_data['y']
    return x,y 

def generate_sample(device = "cpu"):
    """
    Generate samples function from a pretrained model (Here we used our DiT1D model),
    it's use the sample from the test_samples (x being the normalized 1Hz freq & y normalized the 30Hz signal)
    You can also use your own samples if saved in the same way.
    """
    model_name = "DiT1D"  
    model = load_model(model_name=model_name, device=device)
    
    path = f"generated_samples/{model_name}/experiment_{0}"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path, exist_ok=True)

    print(f"Saving images at path : {path}")

    x,y = load_samples()
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x,y),
        batch_size = BATCH_SIZE,
        shuffle = False
    )
    for batch in dataloader:
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)
        #print(f"x shape : {x.shape}, y shape : {y.shape}")
        current_time = time.time()
    
        with torch.no_grad():
            generated = model(x)
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
                            path=path, 
                            idx=sample_idx, 
                            normalized=False, 
                            show_x=False)
            """  frequency_loglog(y = y_sample, 
                               generate=generated_sample, 
                               x=x_sample, 
                               path=path, 
                               idx=sample_idx, 
                               normalized= False) """
    
if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"GPU available : {cuda}")
    device = "cuda" if cuda else "cpu"
    generate_sample(device=device)





