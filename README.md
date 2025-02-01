# Restructuring Vector Quantization with the Rotation Trick

**Restructuring Vector Quantization with the Rotation Trick**  
Christopher Fifty, Ronald G. Junkins, Dennis Duan, Aniketh Iger, Jerry W. Liu, \
Ehsan Amid, Sebastian Thrun, Christopher Ré\
Under Review\
[arXiv](https://arxiv.org/abs/xxxx.yyyyy)

## Note
Copying origin repo from [link](https://github.com/cfifty/rotation_trick). Thanks for this interesting work!


The intention of this repo is to modify the original code to make it work on local machine and to **help us to better understand VQ-VAE architecture and the rotation trick**.

Basically, I modified the code as follows:
1. Pre-download imagenet10 dataset instead of the whole imagenet dataset.
2. Modify the multi-node code to make it work on local machine.
3. Simplify encoder and decoder for clear computation graph.
4. Modify some notation to align with the original paper.


## Approach

In the context of VQ-VAEs, the rotation trick smoothly transforms each encoder output into its corresponding codebook
vector via a rotation and rescaling linear transformation that is treated as a constant during backpropagation. As a
result, the relative magnitude and angle between encoder output and codebook vector becomes encoded into the gradient as
it propagates through the vector quantization layer and back to the encoder.

![method](assets/rot_trick.png)

## Code environment

This code requires Pytorch 2.3.1 or higher with cuda support. It has been tested on Ubuntu 22.04.4 LTS and python 3.8.5.

You can create a conda environment with the correct dependencies using the following command lines:

```
cd rotation_trick
conda env create -f environment.yml
conda activate rotation_trick
```

## Setup

The directory structure for this project should look like:

```
Outer_Directory
│
│───rotation_trick/
│   │   src/
│   │   ...
│
│───imagenet10/
│   │   train/
│   │   │   n03000134/
│   │   |   ...
│   │   val/
│   │   │   n03000247/
│   │   |   ...
```

## Training a Model

Follow the commands in ```src/scripts.sh```.
