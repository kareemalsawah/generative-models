# Tractable Generative Models (WIP)
This repository contains implementations for tractable generative models from the following papers:

## Autoregressive Models
* MADE
* PixelCNN

## Flow Models
* RealNVP
* Glow

## Results
Detailed results for each model and each dataset (architecture, hyperparameters, training curves, samples throughout training, etc.) can be found in pdf form in the results folder along with LaTeX used for the report (still to be added)
#### Shapes
#### Colored Shapes
#### MNIST
#### Colored MNIST
#### CelebA
#### CIFAR-10
#### Miscellaneous

## Structure of this repository

## How to use

## Running experiments and logging
Logging is done using 3 methods:
* An npy file saved at the end of the training containing the training configuration, training losses, validation losses, final testing loss, and generated samples during training
* Tensorboard
* Neptune