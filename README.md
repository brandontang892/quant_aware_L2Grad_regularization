# RegulQuant
CS 242 Final Project, Spring 2021

This is the research framework for our CS 242 Project, at Harvard University. The focus is 
discovering more creative regularizers to use as a quantization-aware training protocol, giving us 
better metrics for quantization error and post-training accuracy. Our current direction is
to expand upon [this paper](https://arxiv.org/abs/2002.07520) which uses L1 
norms of gradients as a regularizer. We challenge their dismissal of using L2 norms of 
gradients by asserting a mathematical caveat and running experiments which show that 
regularizing L2 norms of gradients is as competitive in terms of post-quantization
accuracy and post-quantization performance drop. In addition, we also implement the VarianceNorm 
regularizer (and related regularizers), which are used to penalize the variance (and spread) of weights. 
To our knowledge, regularizing the sample variance of weights (and related functions) for the sake of
influencing the weight distribution has not been done before.
We also run numerous experiments to demonstrate how we can influence the weight distributions 
of our models.


An example of how to run:

``
python train.py --run_name YOURNAME_whatisthistestfor_1 --model ConvNet --dataset CIFAR10 --n_epochs 25 --lr 0.1 --criterion CE 
--scheduler exponential --gamma 0.99 --regularizer L2Grad --lmbda 0.01 --seed 242``

- `--model` only has `ConvNet` implemented.
- `--dataset` only has `CIFAR10` implemented. It uses PyTorch's database.
- `--criterion` only has `CE` (cross entropy loss) implemented.
-  `--scheduler` can be omitted, `exponential` or `multistep`. 
- `--gamma` is the argument for the scheduler.
- `--regularizer` can be omitted or one of the following: `LPGrad`, `VarianceNorm`, `InverseVarianceNorm`, `StdDevNorm`, `InverseStdDevNorm`. For `LPGrad`, the `P` must be specified as an integer greater than or equal to 1. 
- `--lambda` is the regularizing factor. 
- `--seed` is the random seed for the experiment.

Other arguments that are important, but only necessary for specific experiments:
- `--save_file filename` determines the file name for storing the weights of the final epoch model. Defaults to `run_name`
- `--first_regularized_epoch 20` determines which epoch we start to use the regularized loss. Defaults to `0`
- `--save_preregularized_model` turns on the setting to save the model right before regularization starts. 
- `--no_activation_gradients` turns on the setting where LPGrad ignores the activation output gradients, and only computes LP norms of weight gradients.
- `--milestones fixed` determines the milestones for a multistep scheduler. Defaults to `default`, which places two milestones
    at `1/2 * n_epochs` and `5/6 * n_epochs`. `fixed` places milestones at epochs 25 and 50.
- `--watch` turns on the setting where Weights&Biases (explained later) logs histograms of weight parameters and weight gradients over epochs.
- `--watch_freq 10` determines how often the backward hook that collects histograms runs. Defaults to `1`. A lower number implies higher epoch runtime.

The optimizer is fixed to be SGD (with momentum). 

Breakdown of the main files in our codebase:
- `train.py` is our main model training script
- `utils.py` contains functions used in `train.py`
- `regularizers.py` contains implementations of our regularizers
- `models.py` contains our singular model, a custom CNN with 9 convolutional layers and a classifier
- `quant_inference.py` is the script we run to evaluate quantization metrics on a model's weights (stored in `.pt` files). It requires `--n_epochs 1`
    and `--lr 0.1` in the command line (due to `argparse`) but does not use them. Filenames to read from must be _hard-coded_ in the `.py` file.
- `quant_utils.py` contains functions used in `quant_inference.py`.
- `combined_protocol.py` is a version of `train.py` we use for running experiments on combinations of regularizers over time.

Any .ipynb files can be ignored. They are not relevant to the current codebase.

We would like to credit Fang et al. for their open-source implementation of their PWLQ scheme, put forth in [their paper](https://arxiv.org/abs/2002.00104).
We use their implementation of certain functions for our evaluation script.

We also log training data, such as losses, accuracies, regularizer values, weight histograms, and parameter histograms over epochs, using
the Weights&Biases package `wandb`. The link to our project interface, which contains a record of our experiments, sweeps, and reports, is
[here](https://wandb.ai/womeiyouleezi/RegulQuant).


Will Zhang's Guidelines for Good Collaboration:
- Include your name in the `run_name` so that we know who is responsible for each run.
- Remove your failed runs from wandb so that it does not clutter, but do not delete other people's runs without consulting them first.
- Use meaningful run names. Not just "run1" or "run5". We can play around with wandb features so that we can organize runs better. If you know how to do this or learn how to, let us know.
