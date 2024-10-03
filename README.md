# DrugChat: Multi-Modal Large Language Model Enables All-Purpose Prediction of Drug Mechanisms and Properties

This repository holds the code of DrugChat.



## Introduction
- In this work, we make an initial attempt towards enabling ChatGPT-like capabilities on drug molecule graphs, by developing a prototype system DrugChat.
- DrugChat works in a similar way as ChatGPT. Users upload a compound molecule graph and ask various questions about this compound. DrugChat will answer these questions in a multi-turn, interactive manner. 
- The DrugChat system consists of a graph neural network (GNN), a convolutional neural network (CNN), a large language model (LLM), and an adaptor. The GNN takes a compound molecule graph as input and learns a representation for this graph. The adaptor transforms the graph/image representation produced by the GNN/CNN into another representation that is acceptable to the LLM. The LLM takes the compound representation transformed by the adaptor and users' questions about this compound as inputs and generates answers. All these components are trained end-to-end.
- To train DrugChat, we collected instruction tuning datasets.

![overview](figs/DrugChat.png)

## Datasets

Please download the data json files from the drive provided in the paper. The json files contain data for the ChEMBL, PubChem, and DrugBank Instruction Tuning Datasets. Save the files in a folder named `data_public` under this repo. The data structure is as follows. 

{SMILES String: [ [Question1 , Answer1], [Question2 , Answer2]... ] }


## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and ativate it via the following command

```bash
git clone https://github.com/youweiliang/drugchat
cd drugchat
conda env create -f environment.yml
conda activate drugchat
```

Verify the installation of `torch` and `torchvision` is successful by running `python -c "import torchvision; print(torchvision.__version__)"`. If it outputs the version number without any warnings or errors, then you can go to the next step (installing PyTorch Geometric). __If it outputs any warnings or errors__, try to uninstall `torch` by `conda uninstall pytorch torchvision torchaudio cudatoolkit` and then reinstall them following [here](https://pytorch.org/get-started/previous-versions/#v1121). You need to find the correct command according to the CUDA version your GPU driver supports (check `nvidia-smi`). For example, I found my GPU driver supported CUDA 11.6, so I run `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`.

### Installing PyTorch Geometric
Run `conda install pyg=2.3.0 pytorch-scatter=2.1.0 -c pyg` to install PyTorch Geometric. If some error related to PyTorch Geometric or pytorch-scatter show up later when running the code, try to follow [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to reinstall them. 


### Installing RDKit
**To get the data conversion to work properly, you need to create another environment (`rdkit`)**

**It takes around 24 GB GPU memory for the demo.**

To create the `rdkit` environment and run the process, run
```
conda create -c conda-forge -n rdkit rdkit
conda activate rdkit
pip install numpy
python dataset/smiles2graph_demo.py
```

**2. Prepare the pretrained Vicuna weights**

The current version of DrugChat is built on the v0 versoin of Vicuna-13B.
Please refer to our instruction [here](PrepareVicuna.md) 
to prepare the Vicuna weights.
The final weights would be in a single folder in a structure similar to the following:

```
vicuna_weights
├── config.json
├── generation_config.json
├── pytorch_model.bin.index.json
├── pytorch_model-00001-of-00003.bin
...   
```

Then, set the path to the vicuna weight in the model config file 
[here](pipeline/configs/models/drugchat.yaml#L16) at Line 16.


Download the GNN and CNN checkpoints and save them in a folder named `ckpt` under this repo. The file would be
```
ckpt
├── gcn_contextpred.pth
├── ImageMol.pth.tar
```


### Training

To prepare the data for training and evaluation, the data json files should be put under a folder named `data_public` in this directory. Then, run `bash convert_data.sh` to convert the data to molecular images and graphs.

**The training process needs at least 40 GB GPU memory.** 

The training configuration file is [train_configs/drugchat.yaml](train_configs/drugchat.yaml). You may want to change the number of epochs and other hyper-parameters there, such as `max_epoch`, `init_lr`, `min_lr`,`warmup_steps`, `batch_size_train`. You need to adjust `iters_per_epoch` so that `iters_per_epoch` * `batch_size_train` = your training set size.

Start training the projection layer that connects the GNN output and the LLaMA model by running:
```
torchrun --nproc_per_node 1 train.py --cfg-path train_configs/drugchat.yaml
```

### Inference by Launching Demo Locally
To get the inference to work properly, you need to activate the environment `rdkit` in a terminal and launch a process which converts SMILES strings to molecular images and graphs. 

To launch the conversion process, run
```
conda activate rdkit
python dataset/smiles2graph_image_demo.py
```
Then, the python process `smiles2graph_image_demo.py` will be running in the backend to serve the `demo.py`.

**It takes around 24 GB GPU memory for the demo to work.**

Find the checkpoint ('.pth' files) the training process saves, which is the `output_dir` specified in the training configuration [train_configs/drugchat.yaml](train_configs/drugchat.yaml) by default. Specify the checkpoint path in the `ckpt` field under the model section in the evaluation configuration file [eval_configs/drugbank.yaml](eval_configs/drugbank.yaml).

Now we launch the `demo.py` in our `drugchat` environment. Make sure you have run `conda activate drugchat` in a new terminal in the same machine where you launch the python process `smiles2graph_image_demo.py`. Then, start the demo [demo.sh](demo.sh) on your local machine by running `bash demo.sh`. Then, open the URL created by the demo and try it out!

### Batch inference
You can launch a Python script to obtain DrugChat's responses for a batch of queries:
```
python inference.py --cfg-path eval_configs/drugbank.yaml --gpu-id 0 --in_file xxx/smiles_img_qa.json --out_file eval_results/aaa.json
```
where the input file `smiles_img_qa.json` should be in this format:  
{index: [ SMILES: [ [Question1 , Answer1], [Question2 , Answer2]... ] ], ... }

The inference results are saved to the out_file specified above.

## Acknowledgement

+ [MiniGPT-4](https://minigpt-4.github.io/) This repo is based on MiniGPT-4, an awesome repo for vision-language chatbot!
+ [Lavis](https://github.com/salesforce/LAVIS)
+ [Vicuna](https://github.com/lm-sys/FastChat)
+ [ImageMol](https://github.com/HongxinXiang/ImageMol)
+ [Snap-stanford-gnn](https://github.com/snap-stanford/pretrain-gnns/)


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) with BSD 3-Clause License [here](LICENSE_MiniGPT4.md), which is based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).


## Disclaimer

This is a prototype system that has not been systematically and comprehensively validated by pharmaceutical experts yet. Please use with caution. 

Trained models will be released after we thoroughly validate the system with pharmaceutical experts.