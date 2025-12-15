# Physics-Based Super-Resolved Simulation of 3d Elastic Wave Propagation Adopting Scalable Diffusion Transformer

This repository contains an implementation of the paper : Physics-Based Super-Resolved Simulation of 3d Elastic Wave Propagation Adopting Scalable Diffusion Transformer: https://arxiv.org/abs/2504.17308.

## 🔍  Overview 

In this GitHub repository, we provide the model architecture a model checkpoint with is corresponding 
inference file.

## 💻 Setup

### Clone the repository
For the code:
```
git clone https://github.com/HugoGabrielidis16/Seismic_DiT1D
cd Seismic_DiT1D/
```

### Install environnements

Using conda:
```
conda create -f Seismic_DiT1D -f env/environnement.yaml
```
or using pip:
```
pip install -r env/requirements.txt
```
or if you want to use the exact environnement, we provide the docker container at the following link:
```
docker pull yuuuugo/seismic
``` 


### Download model weight

Models weights checkpoints are available upon request on the following link : 
https://drive.google.com/drive/u/0/folders/16Cqdq72sto_WMix2K2tGRlc997N6nadA.

Use the following folder structure

```
model/models_checkpoints/
├── DiT1D.pt
```

### How to run 

To run the model on the provided samples you can either follow the notebooks or run the following command.
If install using pip, you are already inside your envirionment.
If using conda:
```
conda activate Seismic_DiT1D
```

If using the docker container:
```
docker run --gpus all -v $(pwd):/workspace -it yuuuugo/seismic
cd workspace/
```

#### Generate/Inference

To test on the set of samples provided in the repo use: 
```
export PYTHONPATH="./"
python3 generate/generate.py --saving_path="generated_samples/"
```
The results (either saved samples & comparison graph) are then saved in the "generated_samples" folder,


## 🌟 Highlights

- 2025-04-28 Our paper has been submitted to the COMPHY journal and is currently under review.
- 2025-10-28 Our paper has been accepted in the COMPHY journal.

## 📝 Citation

```
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

## 📔 TODO

- [x] Repository with model checkpoint for testing.    
- [ ] Providing PGA model.
- [x] Publish docker images for easier reproduction.
- [ ] Clean and provide training scripts.
- [ ] Publish complete Eida, STEAD and other datasets used to an AWS bucket.