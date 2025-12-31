CIRRICULUM_NAMED_KWARGS = {
    "deep_quench-1a": [
        {"epochs": 6, "n_step": 1, "noise": 0, "virtual_epoch_multiplier": 4},
        {"epochs": 6, "n_step": 1, "noise": 2e-3, "virtual_epoch_multiplier": 4},
        {"epochs": 3, "n_step": 2, "noise": 4e-3, "virtual_epoch_multiplier": 4},
        {"epochs": 3, "n_step": 2, "noise": 6e-3, "virtual_epoch_multiplier": 4},
        {"epochs": 3, "n_step": 4, "noise": 8e-3, "virtual_epoch_multiplier": 2},
        {"epochs": 3, "n_step": 4, "noise": 10e-3, "virtual_epoch_multiplier": 2},
        {"epochs": 3, "n_step": 8, "noise": 12e-3},
        {"epochs": 3, "n_step": 8, "noise": 14e-3},
        {"epochs": 3, "n_step": 8, "noise": 16e-3},
        {"epochs": 3, "n_step": 8, "noise": 18e-3},
        {"epochs": 25, "n_step": 16, "noise": 2e-2},
    ],
    "deep_quench-2a": [  # tries to be faster
        {"epochs": 6, "n_step": 1, "noise": 5e-3, "virtual_epoch_multiplier": 4},
        {"epochs": 3, "n_step": 4, "noise": 1e-2, "virtual_epoch_multiplier": 4},
        {"epochs": 10, "n_step": 8, "noise": 2e-2},
        {"epochs": 10, "n_step": 16, "noise": 2e-2},
    ],
    "shallow_quench-1a": [
        {
            "epochs": 20,
            "n_step": 1,
            "noise": 2e-2,
        },
        {
            "epochs": 40,
            "n_step": 8,
            "noise": 2e-2,
        },
    ],
    "shallow_quench-2a": [
        {
            "epochs": 20,
            "n_step": 1,
            "noise": 2e-2,
        },
        {
            "epochs": 10,
            "n_step": 8,
            "noise": 2e-2,
        },
    ],
}
