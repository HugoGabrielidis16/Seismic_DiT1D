import torch
import numpy as np
import matplotlib.pyplot as plt
from util.normalization import normalize
from util.load_model import load_torchscript_model,load_pga_model
from util.visualize import amplitude_graph, frequencyloglog_graph
from util.load_samples import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"
sample_path = "test_samples/"
saving_path = "generated_samples/"

pga_model ="XGBoost"
use_ddim = True
diffusion_steps = 50
predict_xstart = False

if __name__ == "__main__":
    diffusion_model = load_torchscript_model().to(device)
    pga_model = load_pga_model(model_name = pga_model, device=device)
    dataset = load_data(sample_path)
    with torch.no_grad():
        for i, sample in dataset:
            x,y,y_magnitude = sample
            prediction = diffusion_model(x, diffusion_steps=diffusion_steps)
            amplitude_graph(y, prediction, x, i, path = saving_path, show = True)
            frequencyloglog_graph(y, prediction, x, i, path = saving_path, show = True)

        
