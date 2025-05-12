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

Models weights checkpoints are available upon request on the following link : https://drive.google.com/drive/u/0/folders/16Cqdq72sto_WMix2K2tGRlc997N6nadA 

```
model_checkpont/
├── diffusion_model.pt
├── xGBoost.model
```


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
python3 generate.py --saving_path="generated_samples/"
```

## 📝 Citation

```bibtex
@article{Hugo_2025,
   title={Physics-Based Super-Resolved Simulation of 3d Elastic Wave Propagation Adopting Scalable Diffusion Transformer},
   url={http://dx.doi.org/10.2139/ssrn.5228055},
   DOI={10.2139/ssrn.5228055},
   publisher={Elsevier BV},
   author={Hugo, Gabrielidis and Gatti, Filippo and Stephane, Vialle},
   year={2025} }
```

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocation 2025-[project number] made by GENCI. 