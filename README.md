## Holstein PARC

[![arXiv](https://img.shields.io/badge/arXiv-2412.06631-b31b1b.svg)](https://arxiv.org/abs/2412.06631)

### Environment

Use the provided environment file *(run from the project root)*:

```bash
conda env create -f environment.yml && conda activate holstein-parc
```

Packages installed by `environment.yml`:
- python=3.11
- pytorch
- numpy
- matplotlib
- pillow
- ipython
- tqdm
- tensorboard
- jupyterlab
- pip
- compilers
- make
- armadillo
- arpack

Note that pynvml can optionally be installed to enable utilization-based selection of GPUs.
It is not included in the environment.yml. To install, ensure the holstein-parc environment is active, then run:

```bash
conda install pynvml
```

### C++ simulator (`data/data_src`)

Required to build `qdyn` used by `data/datagen.py`.

Note: The simulator is built with C++17. The provided conda environment already includes a compatible compiler via the `compilers` package - **just activate the env and run the build command below**. If you use a system compiler instead, ensure it supports C++17. The numeric libraries (BLAS/LAPACK/ARPACK/Armadillo) are also installed by `environment.yml` into the conda env.

Build:
```bash
cd data/data_src && make
```


### Generate Data

To generate data for a system with:

    * L = 32 (system size)
    * Deep quench from 0 to 1
    * 1000 prediction steps per trajectory
        * Prediction step size of 2.56 time units

From project root, run:
```bash
python data/datagen.py --n_train=1228 --n_val=32 --n_test=256 --n_workers=32 --dlt_t=0.01 --save_interval=256 --pre_steps=6400 --saved_steps=1000 --L=32 --g_i=0.0 --g_f=1.0 --randomness_level=0.0001 --save_mid_interval=True
```


To generate data for a system with:
    * L = 32 (system size)
    * Shallow quench from 0.5 to 0.8
    * 1200 prediction steps per trajectory
        * Prediction step size of 0.64 time units

From project root, run:
```bash
python data/datagen.py --n_train=64 --n_val=4 --n_test=64 --n_workers=32 --dlt_t=0.01 --save_interval=64 --saved_steps=1200 --L=32 --g_i=0.5 --g_f=0.8 --save_mid_interval=True
```

To change system size from 32 to 16, just change the --L argument to 16.



### Training Model

To train a PARC-based model on the previously generated deep quench data, run from project root:
```bash
python train.py --config_name="L_32-quench_0_1-steps_1000-training/medium_parc_model"
```
Pretrained weights for this model are provided in pretrained_weights/parc-quench_0_1-L_32

To train a tiny standard CNN model on the previously generated L=32 shallow quench data, run from project root:
```bash
python train.py --config_name="L_32-quench_0p5_0p8-steps_1200-training/tiny_simple_model"
```
Pretrained weights for this model are provided in pretrained_weights/tiny-quench_0p5_0p8-L_32

To train the same model on the previously generated L=16 shalow quench data, run:
```bash
python train.py --config_name="L_16-quench_0p5_0p8-steps_1200-training/tiny_simple_model"
```
Pretrained weights for this model are provided in pretrained_weights/tiny-quench_0p5_0p8-L_16

You can suppress tqdm (progress bars) during training with the --suppress_tqdm flag:
```bash
python train.py --config_name="L_16-quench_0p5_0p8-steps_1200-training/tiny_simple_model" --suppress_tqdm=True
```


### Using your own training config

You can add additional config files to the dictionary in /settings/configs.py. To train with them, just specify their name when running:
```bash
python train.py --config_name="<your config name>"
```

You can create and use multiple config files at once. Once added to /settings/configs.py, you can use them by separating them with '/' in the run command:
```bash
python train.py --config_name="<your config name 1>/<your config name 2>"
```

In the case above, the config arguments of <your config name 1> willbe loaded first. Then the arguments of <your config name 2> will be loaded. Any arguments of <your config name 1> that conflict with those in <your config name 2> will be overridden by the value in <your config name 2>.

Training cirriculums can also be added to /settings/cirriculums.py in the same manner. Cirriculums can either be directly specified in the configs, or specified in the run command like a config file:
```bash
python train.py --config_name="<your config name 1>/<your config name 2>/<your cirriculum name>"
```


### Resume training

Training can be resumed by providing the path to the 'LAST_CONFIG.pkl' file:
```bash
python train.py --resume_path=".../checkpoints/<experiment_name>/<experiment_subname>/<timedate>/LAST_CONFIG.pkl"
```
You can override any config arguments when resuming training by specifying a --config_name. The associated config arguments will override the arguments in the loaded config:
```bash
python train.py --resume_path=".../checkpoints/<experiment_name>/<experiment_subname>/<timedate>/LAST_CONFIG.pkl" --config_name="<your override config name>"
```

### Analysis

Various notebooks are provided in /analysis which allow for the testing of trained model checkpoints on generated data.
