# BRIDGE User Guide
## 1. Requirements
### Main Dependency Requirements:
Below are the main dependencies and their version requirements for the project:

- `torch == 1.10.1+cu113`
- `torch-geometric == 2.1.0.post1`
- `torch-cluster == 1.5.9`
- `torch-scatter == 2.0.9`
- `torch-sparse == 0.6.12`
- `torch-spline-conv == 1.2.1`
- `scikit-learn == 1.3.2`
- `numpy == 1.24.4`
- `pandas == 2.0.3`
- `matplotlib == 3.7.5`
- `scipy == 1.10.1`
- `datasets == 3.1.0`
- `huggingface-hub == 0.27.1`
- `wandb == 0.18.7`
- `tqdm == 4.67.0`
- `filelock == 3.16.1`
- `protobuf == 5.28.3`
- `requests == 2.32.3`
- `typing_extensions == 4.12.2`
- `attrs == 24.3.0`

### To install the complete requiring packages, use the following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```

### Dataset & pretrained weights download:
https://huggingface.co/datasets/aboutime233/BRIDGE-data/tree/main

This link provides both the dataset archive and the pretrained model weights.

## 2. Quick Start

Unzip the downloaded dataset archive into a folder named `data` in the root directory of the repository.
Extract pretrained_models_with_readme.tar.gz into the /scripts/saved_model directory.

### Pretraining + Fine-tuning

Navigate to the `model-node/scripts` directory and execute the following command:
```setup
python main.py --dataset Cora --shot_num 1
```

### Fine-tuning
In the same directory as `main.py`, execute the following command:
```setup
python main.py --dataset Cora --model_path pretrain_model.pkl
```

Parameter Explanation:
<br>`--dataset` Select the dataset to use.
<br>`--shot_num` Specify the dataset type for fine-tuning, either 1shot or 5shot.
<br>`--model_path` The path to the pretrained model parameters saved during the pretraining phase.

## 3.  Results reproduction:
You can reproduce our results by executing the following commands in the `scripts` directory under the corresponding task type.
### Node Classification
#### 1-shot
```setup
python main.py --dataset Cora
```
The `dataset` parameter can be set from the following Six datasets: Cora, Citeseer, Computers, Pubmed, Reddit, and Photo.
You also need to add parameters as shown below:
- `--lr` == 0.00008094590967608754 
- `--l2_coef` == 0.00004404197581665391
- `--hid_units` == 256
- `--lambda_entropy` == 0.20401015296835048
- `--dropout_rate` == 0.1913510180577923
- `--variance_weight` == 1521434.9368374627
- `--downstreamlr` == 0.000962404050084371
- `--reg_weight` == 1
- `--reg_thres` == 0.4
- `--shot_num` == 1
When the dataset is Reddit, the learning rate `lr` should be set to 0.00001.

#### 5-shot
```setup
python main.py --dataset Cora
```
The `dataset` parameter can be set from the following Six datasets: Cora, Citeseer, Computers, Pubmed, Reddit, and Photo.
You also need to add parameters as shown below:
- `--lr` == 0.00001 
- `--l2_coef` == 0.00004404197581665391
- `--hid_units` == 256
- `--lambda_entropy` == 0.20401015296835048
- `--dropout_rate` == 0.1913510180577923
- `--variance_weight` == 1521434.9368374627
- `--downstreamlr` == 0.000962404050084371
- `--reg_weight` == 1
- `--reg_thres` == 0.4
- `--shot_num` == 5
### Graph Classification
#### 1-shot
```setup
python main.py --dataset Cora
```
The `dataset` parameter can be set from the following Six datasets: Cora, Citeseer, Computers, Pubmed, Reddit, and Photo.
You also need to add parameters as shown below:
- `--lr` == 0.00008094590967608754 
- `--l2_coef` == 0.00004404197581665391
- `--hid_units` == 256
- `--lambda_entropy` == 0.07845469891338196
- `--dropout_rate` == 0.1913510180577923
- `--variance_weight` == 1521434.9368374627
- `--downstreamlr` == 0.001
- `--reg_weight` == 1
- `--reg_thres` == 0.4
- `--shot_num` == 1

#### 5-shot
```setup
python main.py --dataset Cora
```
The `dataset` parameter can be set from the following Six datasets: Cora, Citeseer, Computers, Pubmed, Reddit, and Photo.
You also need to add parameters as shown below:
- `--lr` == 0.00008094590967608754 
- `--l2_coef` == 0.00004404197581665391
- `--hid_units` == 256
- `--lambda_entropy` == 0.07845469891338196
- `--dropout_rate` == 0.1913510180577923
- `--variance_weight` == 1521434.9368374627
- `--downstreamlr` == 0.001
- `--reg_weight` == 1
- `--reg_thres` == 0.4
- `--shot_num` == 5



