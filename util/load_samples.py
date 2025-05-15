import os
import torch
from glob import glob
from util.visualize import visualize_samples

def load_data(data_dir = "test_samples/"):
    print(os.path.join(data_dir, "sample_*_lowpass.pt"))
    lowpass_samples = glob(os.path.join(data_dir, "sample_*_lowpass.pt")) # The lowpass samples are already normalized
    broadband_samples = glob(os.path.join(data_dir, "sample_*_broadband.pt"))
    broadbandmagnitude_samples = glob(os.path.join(data_dir, "sample_*_magnitude.pt"))
    lowpass_samples.sort()
    broadband_samples.sort
    broadbandmagnitude_samples.sort()
    lowpass_samples = [torch.load(sample) for sample in lowpass_samples]
    broadband_samples = [torch.load(sample) for sample in broadband_samples]
    broadbandmagnitude_samples = [torch.load(sample) for sample in broadbandmagnitude_samples]
    return lowpass_samples, broadband_samples, broadbandmagnitude_samples
    

if __name__ == "__main__":
    lowpass_samples, broadband_samples, broadbandmagnitude_samples = load_data()
    print(f"Loaded {len(lowpass_samples)} lowpass samples, {len(broadband_samples)} broadband samples, and {len(broadbandmagnitude_samples)} broadband magnitude samples.")
    for i in range(len(lowpass_samples)):
        lowpass_sample = lowpass_samples[i]
        broadband_sample = broadband_samples[i]
        broadbandmagnitude_sample = broadbandmagnitude_samples[i]
        print(f"Lowpass sample {i}: {lowpass_sample.shape}")
        print(f"Broadband sample {i}: {broadband_sample.shape}")
        print(f"Broadband magnitude sample {i}: {broadbandmagnitude_sample}")
        visualize_samples(lowpass_sample, broadband_sample)