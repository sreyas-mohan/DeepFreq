

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

The CBLasso implementation is based on the code available [here](http://www.lsta.upmc.fr/boyer/codes/html_CBlasso_vs_Blasso/script_example1_CBlasso_vs_Blasso.html). Since applying CBLasso to test data is long, the performance of CBLasso on `test_dataset` is provided in `CBLasso_test`.


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
