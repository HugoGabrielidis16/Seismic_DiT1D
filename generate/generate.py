import os
import torch
from torchsummary import summary
import time
import matplotlib.pyplot as plt
import xgboost as xgb
from model.unet import simple_UNet
from util.visualize import amplitude_graph, frequency_loglog
from diffusion_method.no_diffusion import noDiffusion
from model.DiffusionTransformer.DiT1D import DiT1D
from diffusion_method.ddpm import DDPM
import matplotlib
from util.loading_model import load_model_with_prefix_handling
from util.metrics import MSE, snr, calculate_seismic_ssim

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

DEMO_STEPS = 10 # Can use 100 if you want better quality but slower generation
BATCH_SIZE = 10

PATH_DICT = {
    "DiT1D": "model/models_checkpoints/DiT1D.pth",
}

def load_model(model_name = "dance_diffusion"):
    print(f"Loading model: {model_name}")   
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    path = PATH_DICT.get(model_name, None)
    if model_name == "DiT1D":
        model = DiT1D(
                depth=12,
                hidden_size=1024,
                num_heads=8,
                input_size=6000,
                in_channels=3,
                mlp_ratio=4,
                blocks_nature="cross_attention",
                tokenizer_name="simple_patching",
                learn_sigma=False,
                prediction_type="sample",
                use_xformers=True,
                patch_size=50
            )
        model = DDPM(model,
                     prediction_type="sample")
    elif model_name == "simple_unet":
        from model.conditional_unet import ConditionalUNetDDPM
        model = simple_UNet(in_channels=3,
            out_channels=3,
            features=256,
            time_emb_dim=256,
            cond_channels=3
        )

    
    model = load_model_with_prefix_handling(path, model)
    model.to(device)
    model.eval() 
    summary(model, input_size=(3, 6000))
    return model

def load_samples(samples_folder = "test_samples/"):
    cached_data = torch.load(os.path.join(samples_folder, "samples.pt"))
    x = cached_data['x']
    y = cached_data['y']
    return x,y 

def generate_sample():
    """
    Generate samples function from a pretrained model (Here we used our DiT1D model),
    it's use the sample from the test_samples (x being the normalized 1Hz freq & y normalized the 30Hz signal)
    You can also use your own samples if saved in the same way.
    """
    model_name = "DiT1D"  
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_name=model_name)
    model = model.to(device)
    model = torch.compile(model)
    
    path = f"generated_samples/{model_name}/experiment_{0}"
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path, exist_ok=True)

    print(f"Saving images at path : {path}")

    x,y = load_samples()
    x = x.to(device)
    y = y.to(device)
    current_time = time.time()
    with torch.no_grad():
        generated = model.sample(x,num_steps = DEMO_STEPS, device = device, training=False)
    elapsed_time = time.time() - current_time

    print(f"Batch size of {BATCH_SIZE} generated in {elapsed_time:.2f} seconds")
    for sample_idx, (x_sample, y_sample, generated_sample) in enumerate(zip(x, y, generated)):
        print(f"Sample {sample_idx}")
        print(f"MSE: {MSE(y_sample, generated_sample)}")
        print(f"SNR: {snr(y_sample, generated_sample)}")
        print(f"SSIM: {calculate_seismic_ssim(generated_sample,y_sample)}")
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
    generate_sample()





