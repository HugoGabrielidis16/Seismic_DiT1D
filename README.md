# Physics-Based Super-Resolved Simulation of 3d Elastic Wave Propagation Adopting Scalable Diffusion Transformer

This repository contains an implementation of the paper : Physics-Based Super-Resolved Simulation of 3d Elastic Wave Propagation Adopting Scalable Diffusion Transformer. 

## 🔍 Overview

In this GitHub repository, we provide a checkpoint of the presentend model and an example model for inference in a notebooks.

## 💻	 Setup

### Clone the repository

```bash
git clone https://github.com/HugoGabrielidis16/Seismic_DiT1D
cd Seismic_DiT1D/
```

Model weights checkpoint is available upon request on the following link : https://drive.google.com/drive/u/0/folders/16Cqdq72sto_WMix2K2tGRlc997N6nadA 

### Install environnements

```bash
conda create -f DiT1D -f env.yaml
```

### How to run

To run the model on the provided samples you can either follow the notebooks 
or run the following command.

```bash
conda activate Seismic_DiT1D
export PYTHONPATH="./"
python3 generate_samples.py --saving_path="generated_samples/"
```

## 📝 Citation

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2025-[project number] made by GENCI. 