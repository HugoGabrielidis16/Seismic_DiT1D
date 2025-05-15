import os 
import torch
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



DEMO_STEPS = 1000
BATCH_SIZE = 15

def generate_sample():
    from torchsummary import summary
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device : {device}")
    model_name = "SimpleUNet"
    #model_name = "UNetDDPM"
    model = load_model(model_name=model_name)
    path = f"generated_test/6000_data_ruche/{model_name}/experiment_0"
    print(f"Path : {path}")
    i = 0
    while os.path.exists(path[:-1] + str(i)):
        i += 1
    path = path[:-1] + str(i)
    os.makedirs(path)
    
    dataset = AugmentedDataModule(
        batch_size = BATCH_SIZE,
        path="data/6000_data/",
        predict_pga=False,
    )
    dataset.setup()
    dataloader = dataset.train_loader
    for batch_idx,batch in enumerate(dataloader):
        x,y, *other = batch
        x = x.to(device)
        y = y.to(device)
        print(x.shape, y.shape)
        metrics = model.test_step(x,y,50,device)
        print(f"Metrics : {metrics}")
        start_time = time()
        generate = model.sample(x,num_steps = 50, device = device)
        end_time = time()
        print(f"Time taken : {end_time - start_time}")
        print(f"Metrics y, max : {y.max()}, min {y.min()}")
        print(f"Metrics generate, max : {generate.max()}, min {generate.min()}")
        for sample_idx,(x_sample,y_sample,generated_sample) in enumerate(zip(x,y,generate)):
            x_sample = create_lowpass_data(y_sample.cpu(),1)
            amplitude_graph(y_sample, generated_sample, x_sample,path, batch_idx + sample_idx, normalized=False)
            frequency_loglogv2(y_sample, generated_sample, x_sample, path, batch_idx + sample_idx, normalized=False)
        break

        #plot_gofs(ys = y,y_preds= generate, path = path)

if __name__ == "__main__":
    generate_sample()




