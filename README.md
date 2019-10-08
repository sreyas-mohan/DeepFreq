# Code and Pretrained Networks from <br>"Data-driven Estimation of Sinusoid Frequencies"

This repository contains information, code and models from the paper [Data-driven Estimation of Sinusoid Frequencies](https://arxiv.org/abs/1906.00823) by Gautier Izacard, Sreyas Mohan and [Carlos Fernandez-Granda](https://cims.nyu.edu/~cfgranda/).

## Frequency Estimation via Deep Learning

Frequency estimation is a fundamental problem in signal processing, with applications in radar imaging, underwater acoustics, seismic imaging, and spectroscopy. The goal is to estimate the frequency of each component in a multisinusoidal signal from a finite number of noisy samples. The estimation problem is illustrated in the following figure. The blue data are N samples from a multisinudoisal signal (blue dashed lines) sampled at the Nyquist rate. The bottom row of the image shows that the resolution of the frequency estimate obtained by computing the discrete-time Fourier transform from N samples decreases as we reduce N. To solve the frequency-estimation problem we need to super-resolve the frequency locations from the data.

![problem_illustration](./figures/problem_illustration.png) 

In previous work, we proposed to perform frequency estimation by training a neural network, called a PSNet](https://math.nyu.edu/~cfgranda/pages/stuff/LearningBased.pdf), to output a learned representation with local maxima at the position of the frequency estimates. Here, we propose a novel neural-network architecture that produces a significantly more accurate representation, and combine it with an additional neural-network module trained to detect the number of frequencies. 

![architecture](./figures/model.png) 


This yields a fast, fully-automatic method for frequency estimation that achieves state-of-the-art results. In particular, it outperforms existing techniques by a substantial margin at medium-to-high noise levels. The following figure compares our methodology to two subspace-based methods that use the covariance matrix of the data, and a sparse-estimation method based on convex optimization. Estimation accuracy is measured using the [Chamfer distance](https://www.sciencedirect.com/science/article/pii/0734189X84900355) between the true frequencies and the estimates. 

<p align="center"> <img src="./figures/endtoend.png" width='700'></p>


## Code and Pre-trained Models

### pre-trained models
The directory `pretrained_models` contains the pretained models of DeepFreq. 

### Train
Train a model from scratch:

```shell
python train.py \
	--n_training 200000 \
	--n_epochs_fr 200 \
	--n_epochs_c 100 \
	--output_dir /checkpoint/experiment_name \
```

See `train.py` for additional training options.

### Test

Evaluate model performance against several baselines.


```shell
python test.py \
	--data_dir test_dataset/ \
  --output_dir results/ \
  --fr_path pretrained_models/DeepFreq/fr_module.pth \
  --counter_path pretrained_models/DeepFreq/counter_module.pth \
  --psnet_path pretrained_models/PSnet/psnet.pth \
	--psnet_counter_path pretrained_models/PSnet/counter_psnet.pth \
	--overwrite
```

The CBLasso implementation is based on the code available [here](http://www.lsta.upmc.fr/boyer/codes/html_CBlasso_vs_Blasso/script_example1_CBlasso_vs_Blasso.html). Since applying CBLasso to test data is long, the performance of CBLasso on `test_dataset/` is provided in `CBLasso_test/`.


### `generate_dataset.py`: Generate test data


```shell
python generate_dataset.py \
    	--output_dir my_testset/ \
    	--n_test 1000 \
	--signal_dimension 50 \
   	--minimum_separation 1. \
    	--dB 0 5 10 15 20 25 30 35 40 45 50 \
```

The test data used in the original paper is available in the `test_dataset/` directory.

### `example_notebook.ipynb`: Apply a pre-trained model

In this notebook, DeepFreq is applied to different signals and the results are visualized. 
