# CryoTRANS Overview

CryoTRANS (TRansformations by Artificial NetworkS) is a software that predicts high-resolution maps of rare conformations by constructing a pseudo-trajectory between density maps of varying resolutions for Cryogenic Electron Microscopy (cryo-EM). This trajectory is represented by an ordinary differential equation parameterized by a deep neural network, ensuring the retention of detailed structures from high-resolution density maps.

## Video Tutorial

[TBD]

## Publications

[TBD]

## The List of Available Demo Cases

[TBD]

# Installation

CryoTRANS is an open-source software, developed using Python, and is available as a Python package. Please access our source code [on GitHub](https://github.com/mxhulab/cryotrans).

## Prerequisites

- Python version 3.7 or later.
- NVIDIA CUDA library installed in the user's environment.

## Dependencies

The CryoTRANS package depends on the following libraries:

```
numpy>=1.18
mrcfile>=1.4.3
cupy>=10
torch>=1.10
```

## Preparation of CUDA Environment

We recommend installing CuPy and PyTorch initially, as their installation largely depends on the CUDA environment. Please note, PyTorch should be CUDA-capable. To streamline this process, we suggest preparing a conda environment with the following commands.

For CUDA version <= 11.7:
```
conda create -n CRYOTRANS_ENV python=3.8 cudatoolkit=10.2 cupy=10.0 pytorch=1.10 -c pytorch -c conda-forge
```
Please note that this command is tailored for CUDA version 10.2. To accommodate a different CUDA version, adjust the `cudatoolkit` version accordingly. Modify the versions of Python, [CuPy](https://cupy.dev), and [PyTorch](https://pytorch.org) based on requirements, ensuring compatibility with the minimal requirements of CryoTRANS.

For CUDA version >= 11.8:
```
conda create -n CRYOTRANS_ENV python=3.10 cupy=12.0 pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
```
Please note that this command is tailored for CUDA environment version 12.1. For a different CUDA version, adjust `pytorch-cuda` version accordingly.

## Installing CryoTRANS

After preparing CuPy and PyTorch, it is crucial to activate it before proceeding with the CryoTRANS installation.
```
conda activate CRYOTRANS_ENV
```

Then, we turn to the step of installing CryoTRANS. CryoTRANS can be installed either via `pip` or `conda`.

To install CryoTRANS using `pip`, execute the following command:
```
pip install cryotrans
```
Alternatively, to install CryoTRANS using `conda`, execute the following command:
```
conda install -c mxhulab cryotrans
```

# Tutorial

In this tutorial, we will use the protein TmrAB as an example to demonstrate how to utilize CryoTRANS to construct a pseudo-trajectory and predict high-resolution maps.

## Prepare the Dataset

We will be working with two conformations of TmrAB at resolution of 3Å and 7Å, respectively. The files required for this tutorial are `TmrAB_0_3A.mrc` and `TmrAB_1_7A.mrc`. Our goal is to use the first conformation at 3Å and the second conformation at 7Å to predict a high-resolution density map for the second conformation.

You can directly download the required density maps from the following [link](https://github.com/mxhulab/cryotrans-demos/tree/master/TmrAB). Once downloaded, please place them in a clean working directory. Next, navigate to the working directory and activate the Conda environment by running the following command:
```
cd WORKING_DIR
conda activate CRYOTRANS_ENV
```
Make sure to replace `WORKING_DIR` with the actual path to your working directory.

## Train

CryoTRANS enables flexible registration between two cryo-EM density maps using an ODE-governed deformation field parameterized by a deep neural network (DNN). In this process, the initial map corresponds to a high-resolution density map, while the terminal map corresponds to a low-resolution target map. CryoTRANS will deform the initial map with the neural ODE to a generated map and try to minimize the Wasserstein distance between the generated map and the target map.

To proceed with the training step, execute the following command:
```
cryotrans-train -i0 TmrAB_0_3A.mrc -t0 0.24 -i1 TmrAB_1_7A.mrc -t1 0.3 -d output -n 2000 -p 500
```

Here's the explanation of each option in the CryoTRANS command:
- The `-i0` and `-i1` arguments specify the paths to the initial map and target map, respectively.
- The `-t0` and `-t1` arguments represent the thresholds for the initial and target maps, respectively. CryoTRANS requires thresholding to ensure that the density values are positive for calculating the squared Wasserstein distance.
- The `-d` argument specifies the working directory of CryoTRANS. All intermediate results, including detailed loss logs and DNN weight files, will be written to this directory.
- The `-n` argument determines the number of iterations CryoTRANS will perform during training.
- The `-p` argument sets the period for periodical reporting. In this case, with `-p 500`, CryoTRANS will save the DNN weight file and update the loss log every 500 iterations. This allows you to monitor the progress and performance of the training process.

CryoTRANS will utilize one GPU card for training. The entire process may take over several minutes, depending on your system resources. You will see a brief log of the training progress printed on your screen, which will look like this:
```
|-----------------------------------------|
|   iter   |       loss       |   time    |
|----------|------------------|-----------|
|    0     | -1.369131327e+00 | 2.251e-01 |
|   500    | -1.124812317e+01 | 3.069e+01 |
|   1000   | -1.160682774e+01 | 6.244e+01 |
|   1500   | -1.170444012e+01 | 9.550e+01 |
|   2000   | -1.175122833e+01 | 1.296e+02 |
|-----------------------------------------|
Save network ... done.
```

In addition to the log, multiple files will be generated and saved in the `output` directory. In this specific command with 2000 iterations and saving the weight file every 500 iterations, a total of 7 weight files will be saved. These files will be named `net_0.pt` to `net_5.pt`, representing the weight files at different points during the training process. Finally, the final result weight file will be saved as `net.pt`.

## Predict

Once the training step is complete, proceed with the prediction step by executing the following command:
```
cryotrans-predict -i TmrAB_0_3A.mrc -d output -w output/net.pt
```

Let's understand the options in this CryoTRANS command:
- The `-i` argument specifies the paths to the initial map.
- The `-d` argument determines the working directory for CryoTRANS. The pseudo-trajectory will be written to this directory.
- The `-w` argument specifies the DNN weight file. In this case, we use the network obtained from the training step, which is `output/net.pt`, to perform the prediction.

Similarly, during the prediction step using CryoTRANS, a single GPU card will be utilized. The prediction process typically takes a few seconds to complete. The resulting pseudo-trajectory will consist of a list of density files, named `frame_00.mrc`, `frame_01.mrc`, and so on, up to `frame_10.mrc`, saved in the directory `output`. The last file in the sequence will be the final generated map, which represents the deformed version of the initial map that matches the target map to the best extent possible.

## Advanced technique: L2 refine

When the resolution of the generated map and the target map is very close, and a large portion of the maps overlap in space, one can use L2 loss function instead of squared Wasserstein distance to further refine the results.

To demonstrate this technique, two conformations of TmrAB both at resolution of 3Å are used. The required files, namely `TmrAB_0_3A.mrc` and `TmrAB_1_3A.mrc`, can be downloaded from the following [link](https://github.com/mxhulab/cryotrans-demos/tree/master/TmrAB).

The training process consists of two consecutive steps. In the first step, squared Wasserstein distance is used as the loss function for rough registeration. Then, the the resulting DNN is used as input in the second step, where L2 loss is applied to further refine the DNN. The following commands are used for these steps:
```
cryotrans-train -i0 TmrAB_0_3A.mrc -t0 0.24 -i1 TmrAB_1_3A.mrc -t1 0.24 -d output -n 2000 -p 1000
cryotrans-train -i0 TmrAB_0_3A.mrc -t0 0.24 -i1 TmrAB_1_3A.mrc -t1 0.24 -d output -w output/net.pt -n 15000 -p 1000 --l2 --lr 1e-4
```
We note that the `-w` option specifies the previously obtained weight file (`output/net.pt`) as the input initial weight. The flag `--l2` indicates the use L2 loss instead of Wasserstein loss. In addition, `--lr 1e-4` option sets the learning rate to 1e-4, which is more suitable for L2 loss according to our experience.

## Advanced technique: Enlarging the scale of DNN

By default, CryoTRANS utilizes a multi-layer perception (MLP) structure with 2 hidden layers (depth 3), each having a width of 100. However, for datasets with complex velocity fields for deformation, the limited capacity of the DNN may not be sufficient to represent the velocity field accurately. To address this, CryoTRANS provides two hyperparameters, namely the depth and the width of the MLP, to tune the network structure.

As an example, the technique is applied to two conformations of Mm-cpn (chain D) at a resolution of 3Å. The required files, namely `Mm-cpnD_0_3A.mrc` and `Mm-cpnD_1_3A.mrc`, can be downloaded from the following [link](https://github.com/mxhulab/cryotrans-demos/tree/master/Mm-cpn).

The following commands are used for training process:
```
cryotrans-train -i0 Mm-cpnD_0_3A.mrc -t0 0.24 -i1 Mm-cpnD_1_3A.mrc -t1 0.24 -d output -n 5000 -p 1000 --width 900
cryotrans-train -i0 Mm-cpnD_0_3A.mrc -t0 0.24 -i1 Mm-cpnD_1_3A.mrc -t1 0.24 -d output -w output/net.pt -n 50000 -p 1000 --width 900 --l2 --lr 1e-4
```
It is important to note that the `--width 900` option is used to enlarge the scale of the MLP by setting its width to 900.

## Guideline: Preventing overfitting

In deep learning, overfitting occurs when a model becomes too specialized to the training data and performs poorly on unseen data. In their community, a common approach to prevent overfitting is to use a validation set to monitor the behavior of the DNN during training. The technique known as "early stopping" is employed, where training is stopped when the performance on the validation set starts to deteriorate.

However, in the case of CryoTRANS, there is no explicit validation set available to determine when to stop the training process. Instead, it is recommended to run the training process for as long as possible and save all intermediate weights periodically. This allows users to experiment with different weight files and explore the generated pseudo-trajectories from these intermediate checkpoints. In some cases, the results obtained from an intermediate weight file may prove to be beneficial, even if the final weight file does not yield satisfactory results.

## Guideline: Multiscale

When the input maps are too large to be processed quickly, CryoTRANS provides an option to utilize binning or downscaling. The command-line argument `-b` or `--binning` is used, following by specifying the level of binning or downscaling. For example, a binning level of 4 or 2 can be set, which means the initial and target maps will be downsampled accordingly.

# Options/Arguments

## Options/Arguments of `cryotrans-train`

```
$ cryotrans-train -h
usage: cryotrans-train [-h] [-i0 INITIAL_MAP] [-t0 INITIAL_THRESHOLD] [-i1 TARGET_MAP] [-t1 TARGET_THRESHOLD]
                       [-d DIRECTORY] [-g GPU] [-w WEIGHT] [-b BINNING] [-n N_STEPS] [-p PERIOD] [--depth DEPTH]
                       [--width WIDTH] [--w2_eps W2_EPS] [--w2_iter W2_ITER] [--l2] [--lr LR] [--mu_mesh MU_MESH]

CryoTRANS: Predicting high-resolution maps of rare conformations using neural ODEs in cryo-EM.

options:
  -h, --help            show this help message and exit

Basic arguments:
  -i0 INITIAL_MAP, --initial-map INITIAL_MAP
                        Path of initial map.
  -t0 INITIAL_THRESHOLD, --initial-threshold INITIAL_THRESHOLD
                        Threshold for the initial map.
  -i1 TARGET_MAP, --target-map TARGET_MAP
                        Path of target map.
  -t1 TARGET_THRESHOLD, --target-threshold TARGET_THRESHOLD
                        Threshold for the target map.
  -d DIRECTORY, --directory DIRECTORY
                        Working directory.
  -g GPU, --gpu GPU     Which gpu to use, 0 by default.
  -w WEIGHT, --weight WEIGHT
                        Path of network weight file as initial model.
  -b BINNING, --binning BINNING
                        Binning level, 1 by default.
  -n N_STEPS, --n_steps N_STEPS
                        Number of training steps.
  -p PERIOD, --period PERIOD
                        For periodic report.

Advanced arguments:
  --depth DEPTH         Depth of velocity net (MLP).
  --width WIDTH         Width of velocity net (MLP).
  --w2_eps W2_EPS       Entropic regularisation parameter for W2 loss.
  --w2_iter W2_ITER     Number of Sinkhorn iteration for computing W2 loss.
  --l2                  Use L2 refine instead of W2.
  --lr LR               Learning rate. Suggest 1e-4 when doing L2 refine.
  --mu_mesh MU_MESH     Regularization parameter for tetrahedral mesh loss.
```

## Options/Arguments of `cryotrans-predict`

```
$ cryotrans-predict -h
usage: cryotrans-predict [-h] [-i INITIAL_MAP] [-t INITIAL_THRESHOLD] [-d DIRECTORY] [-p PREFIX] [-g GPU] [-w WEIGHT]
                         [--depth DEPTH] [--width WIDTH]

CryoTRANS: Predicting high-resolution maps of rare conformations using neural ODEs in cryo-EM.

options:
  -h, --help            show this help message and exit

Basic arguments:
  -i INITIAL_MAP, --initial-map INITIAL_MAP
                        Path of initial map.
  -t INITIAL_THRESHOLD, --initial-threshold INITIAL_THRESHOLD
                        Threshold for the initial map.
  -d DIRECTORY, --directory DIRECTORY
                        Working directory.
  -p PREFIX, --prefix PREFIX
                        Prefix for output movie.
  -g GPU, --gpu GPU     Which gpu to use, 0 by default.
  -w WEIGHT, --weight WEIGHT
                        Path of network weight file as initial model.

Advanced arguments:
  --depth DEPTH         Depth of velocity net (MLP).
  --width WIDTH         Width of velocity net (MLP).
```
