# reproducibility of TensorFlow with ResNet

Welcome to the repository accompanying the paper by Miguel Morin and Matthew
Willetts.

## monkey-patching

We "monkey-patched" the Tensorflow code in order to save models before any
non-determinism, verify their equality, and confirm that any difference in
running the model from the same seed came from non-determinism in estimating the
model. We used version 2.2.4 of Keras and changed two files, `training.py` and
`training_arrays.py`, and we include their original versions as `training224.py`
and `training_arrays224.py` (source at `https://raw.githubusercontent.com/keras-team/keras/2.2.4/keras/engine/`).

## equality comparison

The experiment of 11th July with 50 runs and 200 epochs gave these standard deviations:

```
$ python3 stdev.py ../cifar\ results/resnet_same_seed.csv 
[0.01994929 0.00301954]
$ python3 stdev.py ../cifar\ results/resnet_different_seeds.csv 
[0.02699318 0.00346379]
```

So the entropy inherent in GPU calculations accounts for between 74% and 87% of the standard deviation in the scores of a CIFAR neural network. These results are so surprising that Matthew wanted to check that with the same seed, the initial models are the same, that the training data is the same, and that the batches are also the same.

This pull request concludes that comparison. I tested it on 5 runs with 3 epochs for lack of budget to run the full experiment. The results are that, with different seeds, the models and batches are different (validating that the code is indeed performing the comparison); with the same seed, the models are the same (they have the same weights on the same layers across all 5 runs) and the batches are the same (the text files `array 0.txt` for epoch 0 are the same across all 5 runs).

To run this experiment, change `params.txt` to the number of runs in the first line and epochs in the second line, then run this line, possibly with some changes depending on your setup:

```
./deploy_azure.sh
```

The results for an experiment with 5 runs and 3 epochs were:

```
Comparison results for different_seeds:

Models all the same: False(number of models checked: 5/5)
- Different model (layers and weights compared with run #0) in this run: #1 (file = saved_models/different_seeds1/cifar10_ResNet29v2_model.h5, layer = 1, array = 0)
- Different model (layers and weights compared with run #0) in this run: #2 (file = saved_models/different_seeds2/cifar10_ResNet29v2_model.h5, layer = 1, array = 0)
- Different model (layers and weights compared with run #0) in this run: #3 (file = saved_models/different_seeds3/cifar10_ResNet29v2_model.h5, layer = 1, array = 0)
- Different model (layers and weights compared with run #0) in this run: #4 (file = saved_models/different_seeds4/cifar10_ResNet29v2_model.h5, layer = 1, array = 0)
XY training data all the same: True(number of models checked: 5/5)
Batches all the same: False(number of models checked: 5/5)
- Different batches array (compared with run #0) in this run: #1
- Different batches array (compared with run #0) in this run: #2
- Different batches array (compared with run #0) in this run: #3
- Different batches array (compared with run #0) in this run: #4

Comparison results for same_seed:

Models all the same: True(number of models checked: 5/5)
XY training data all the same: True(number of models checked: 5/5)
Batches all the same: True(number of models checked: 5/5)
```

The standard deviations on that same experiment were:

```
$ python3 stdev.py resnet_different_seeds.csv 
[0.08822624 0.03021454]
$ python3 stdev.py resnet_same_seed.csv 
[0.15188129 0.02744777]
```

## Running on Azure (GPUs)

If you have issues in these instructions, make sure that your setup can perform all the steps
in [this
tutorial](https://github.com/miguelmorin/reproducibility/Azure_tutorial.md)
for running TensorFlow on Azure with Docker and GPU support.

### Azure

Set up an Azure Virtual Machine with `NVIDIA GPU Cloud Image` on hardware
`Standard NV6` (as in [this internal Knowledgebase
tutorial](https://github.com/miguelmorin/reproducibility/Azure_tutorial.md). Choose
public key authentication (instead of password) and use the private key saved in
a file, e.g. `~/.ssh/azure` (as that's how the author does it).

If you need to generate an SSH key:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/azure
```

Start the virtual machine and make a note of the IP address.

### Local machine

Start a shell and change directory into this one. Launch the job remotely
with

```bash
./azure.sh IP=<IP address> [private_key=<SSH private key file>]
[connect_mode=detach] directory=<path to model directory> [dockerfile=<dockerfile>]
```

The arguments in brackets are optional. The default is to run in attached mode,
to use your default SSH key, and to use `DOCKERFILE`.

For example, to run the reproducibility tests on Azure on GPUs, run this command:

```bash
./azure.sh IP=51.140.69.203 directory='playground/reproducibility/' dockerfile=DOCKEFILE-GPU
```

and to launch a Docker container on the virtual machine and copy the
model files back every 10 seconds, run this command:

```bash
./azure.sh IP=51.140.69.203 connect_mode=detach directory='playground/VAE on MNIST/'
```

## Running models locally on macOS (CPUs)

Follow [these
instructions](https://github.com/miguelmorin/reproducibility/macOS_tutorial.md)
to install TensorFlow on macOS, then call the models with, for example

```
(venv) $ python 'annotated code snippets/CNN for MNIST with keras/mnist.py'
```

You may need to code relating to the original repo for the annotated
code. Please refer to the `README.md` in that directory for details.

