from torch import nn
import torch


CONFIG_NAMED_KWARGS = {
    # Main training configs
    "L_32-quench_0_1-steps_1000-training": {
        "name": "L_32-quench_0_1-steps_1000-training",
        # Data variables
        "data_names": ["L-32-g_0_1-rd-1e-04-si_256-ps_6400-steps_1000-zero_mom-MID"],
        # Val Data Details
        "val_n_step_prediction": 64,
        # Training details
        "cirriculum_name": "deep_quench-1a",
        "dropout": 0.2,
        # Convenience features
        "suppress_tqdm": True,
        "save_checkpoint_period": 1,
    },
    "L_32-quench_0p5_0p8-steps_1200-training": {
        "name": "L_32-quench_0p5_0p8-steps_1200-training",
        # Data variables
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_64-steps_1200-zero_mom-MID"],
        # Val data details
        "val_batch_size": 128,
        "val_n_step_prediction": 64,
        # Training details
        "cirriculum_name": "shallow_quench-1a",
        "batch_size": 64,
        "dropout": 0.2,
        "eta_min_scalar": 0.1,
        # Convenience features
        "suppress_tqdm": True,
        "save_checkpoint_period": 1,
    },
    "L_16-quench_0p5_0p8-steps_1200-training": {
        "name": "L_16-quench_0p5_0p8-steps_1200-training",
        # Data variables
        "data_names": ["L-16-g_0p5_0p8-rd-1e-04-si_64-steps_1200-zero_mom-MID"],
        # Val data details
        "val_batch_size": 256,
        "val_n_step_prediction": 64,
        # Training details
        "cirriculum_name": "shallow_quench-1a",
        "batch_size": 64,
        "dropout": 0.2,
        "eta_min_scalar": 0.1,
        # Convenience features
        "suppress_tqdm": True,
        "save_checkpoint_period": 1,
    },
    "L_32-quench_0p5_0p8-steps_1200-training-v2": { # This one faster and just as good
        "name": "L_32-quench_0p5_0p8-steps_1200-training-v2",
        # Data variables
        # "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_64-steps_1200-zero_mom-MID"],
        "data_scalars": None,  # delete later when uncommenting data_names
        # Val data details
        "val_batch_size": 128,
        "val_n_step_prediction": 64,
        # Training details
        "cirriculum_name": "shallow_quench-2a",
        "batch_size": 128,
        "dropout": 0.2,
        "weight_decay": 1e-2,
        "eta_min_scalar": 0,
        "lr": 1e-3,
        "lr_scheduler_linear_decay_multiplier": 0.0,
        # Convenience features
        "suppress_tqdm": True,
        "save_checkpoint_period": 1,
    },
    # Training modifications
    "deep-fast_train-1a": {
        "sub_name": "deep-fast_train-1a",
        "cirriculum_name": "deep_quench-2a",
        "lr": 1e-3,
        "eta_min_scalar": 1e-5,
        "lr_scheduler_linear_decay_multiplier": 0.0,
        "weight_decay": 1e-2,
    },
    "cosine_lr_modifier": {
        "sub_name": "cosine_lr_modifier",
        "lr_scheduler": "cosine",
        "lr_scheduler_linear_decay_multiplier": None,
    },
    "lr_1e-4": {"sub_name": "lr_1e-4", "lr": 1e-4},
    "vpe_1024": {"sub_name": "vpe_1024", "virtual_epoch_size": 1024},
    # Scaling series data modifications
    "deep_scaling_series_si_32": {
        "sub_name": "deep_scaling_series_si_32",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_32-ps_6400-steps_8192-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "deep_scaling_series_si_64": {
        "sub_name": "deep_scaling_series_si_64",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_64-ps_6400-steps_4096-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "deep_scaling_series_si_128": {
        "sub_name": "deep_scaling_series_si_128",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_128-ps_6400-steps_2048-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "deep_scaling_series_si_256": {
        "sub_name": "deep_scaling_series_si_256",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_256-ps_6400-steps_1024-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "deep_scaling_series_si_512": {
        "sub_name": "deep_scaling_series_si_512",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_512-ps_6400-steps_512-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "deep_scaling_series_si_1024": {
        "sub_name": "deep_scaling_series_si_1024",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_1024-ps_6400-steps_256-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "deep_scaling_series_si_2048": {
        "sub_name": "deep_scaling_series_si_2048",
        "data_names": ["L-32-g_0_1-rd-1e-04-si_2048-ps_6400-steps_128-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    ## Shallow scaling series
    "shallow_scaling_series_si_32": {
        "sub_name": "shallow_scaling_series_si_32",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_32-steps_2048-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "shallow_scaling_series_si_64": {
        "sub_name": "shallow_scaling_series_si_64",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_64-steps_1024-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "shallow_scaling_series_si_128": {
        "sub_name": "shallow_scaling_series_si_128",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_128-steps_512-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "shallow_scaling_series_si_256": {
        "sub_name": "shallow_scaling_series_si_256",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_256-steps_256-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "shallow_scaling_series_si_512": {
        "sub_name": "shallow_scaling_series_si_512",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_512-steps_128-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "shallow_scaling_series_si_1024": {
        "sub_name": "shallow_scaling_series_si_1024",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_1024-steps_64-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    "shallow_scaling_series_si_2048": {
        "sub_name": "shallow_scaling_series_si_2048",
        "data_names": ["L-32-g_0p5_0p8-rd-1e-04-si_2048-steps_32-zero_mom-MID-SCALING_SERIES"],
        "data_scalars": "same_as_data",
    },
    # Models
    "large_parc_model": {
        "sub_name": "large_parc_model",
        "channels": 512,
        "n_blocks": 8,
        "kernel_size": 3,
        "use_residual_scalar": False,
        "act_func": nn.Tanh(),
        "init_method": "xavier_uniform",
        "zero_initialize_output": True,
        "dtype": torch.float32,
        "use_layernorm": True,
        "simple_cnn": False,
        "use_derivatives": True,
    },
    "medium_parc_model": {
        "sub_name": "medium_parc_model",
        "channels": 128,
        "n_blocks": 4,
        "kernel_size": 3,
        "use_residual_scalar": False,
        "act_func": nn.Tanh(),
        "init_method": "xavier_uniform",
        "zero_initialize_output": True,
        "dtype": torch.float32,
        "use_layernorm": True,
        "simple_cnn": False,
        "use_derivatives": True,
    },
    "tiny_simple_model": {
        "sub_name": "tiny_simple_model",
        "channels": 16,
        "n_blocks": 1,
        "kernel_size": 3,
        "use_residual_scalar": False,
        "act_func": nn.Tanh(),
        "init_method": "xavier_uniform",
        "zero_initialize_output": True,
        "dtype": torch.float32,
        "use_layernorm": True,
        "simple_cnn": True,
        "use_derivatives": False,
        "deriv_loss_coeff": None,
    },
}
