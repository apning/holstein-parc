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
    # Models
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
