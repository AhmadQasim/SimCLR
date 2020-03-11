# SimCLR
PyTorch implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations by T. Chen et al.

[Link to paper](https://arxiv.org/pdf/2002.05709.pdf)

### Quickstart
This downloads a pre-trained model and trains the linear classifier, which should receive an accuracy of ±`72%` on the STL-10 test set.
```
git clone https://github.com/spijkervet/SimCLR.git && cd SimCLR
wget https://github.com/Spijkervet/SimCLR/releases/download/1.0/checkpoint_40.tar
sh setup.sh || python3 -m pip install -r requirements.txt || exit 1
python -m testing.logistic_regression with model_path=. model_num=40
```

### Results
| Method  | Batch Size | ResNet | Projection output dimensionality | STL-10 |
| ------------- |------------- | ------------- | ------------- |------------- |
| SimCLR + Linear eval. | 256 | ResNet18 | 64 | **0.718**  |


## What is SimCLR?
SimCLR is a "simple framework for contrastive learning of visual representations". The contrastive prediction task is defined on pairs of augmented examples, resulting in 2N examples per minibatch. Two augmented versions of an image are considered as a correlated, "positive" pair (x_i and x_j). The remaining 2(N - 1) augmented examples are considered negative examples. The contrastive prediction task aims to identify x_j in the set of negative examples for a given x_i.


![GitHub Logo](/media/architecture.png)

## Usage
Run the following command to setup a conda environment:
```
sh setup.sh
conda activate simclr
```

Or alternatively with pip:
```
pip install -r requirements.txt
```

Then, simply run:
```
python main.py
```

### Testing
To test a trained model, make sure to set the `model_path` variable in the `config/config.yaml` to the log ID of the training (e.g. `logs/0`).
Set the `model_num` to the epoch number you want to load the checkpoints from (e.g. `40`).

```
python -m testing.logistic_regression
```


## Configuration
The configuration of training can be found in: `config/config.yaml`. I personally prefer to use files instead of long strings of arguments when configuring a run. An example `config.yaml` file:
```
# train options
batch_size: 256
workers: 16
start_epoch: 0
epochs: 40

# model options
resnet: "resnet18"
normalize: True
projection_dim: 64

# loss options
temperature: 0.5

# reload options
model_path: "logs/0" # set to the directory containing `checkpoint_##.tar` 
model_num: 40 # set to checkpoint number

# logistic regression options
logistic_batch_size: 256
logistic_epochs: 100
```

## Logging and TensorBoard
The `sacred` package is used to log all experiments into the `logs` directory. To view results in TensorBoard, run:
```
tensorboard --logdir logs
```


#### Dependencies
```
torch
torchvision
tensorboard
sacred
pyyaml
```