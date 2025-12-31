import os
import csv
import multiprocessing as mp

import numpy as np
import torch

from src.modeling import HolsteinPARC
from src.data_utils import unpickle_data
from src.analysis_utils import gen_preds, np_rmse

""" Arguments """

checkpoints = [
    {"si": 32, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_32/best_epoch_29.pth"},
    {"si": 64, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_64/best_epoch_29.pth"},
    {"si": 128, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_128/best_epoch_29.pth"},
    {"si": 256, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_256/epoch_29.pth"},
    {"si": 512, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_512/epoch_29.pth"},
    {"si": 1024, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_1024/epoch_29.pth"},
    {"si": 2048, "path": "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/si_2048/epoch_29.pth"},
]
data_path = "/bigtemp/rnx2bc/holstein-parc/deep_download_scaling_series/L-32-g_0_1-rd-1e-04-si_2048-ps_6400-steps_128-zero_mom-MID-SCALING_SERIES-test.pkl"

# checkpoints = [
#     {
#         "si": 32,
#         "path": "/bigtemp/rnx2bc/holstein-parc/checkpoints/L_32-quench_0p5_0p8-steps_1200-training-v2/tiny_simple_model-shallow_scaling_series_si_32/20251026-113211/best_epoch_23.pth",
#     },
#     {
#         "si": 64,
#         "path": "/bigtemp/rnx2bc/holstein-parc/checkpoints/L_32-quench_0p5_0p8-steps_1200-training-v2/tiny_simple_model-shallow_scaling_series_si_64/20251026-113144/best_epoch_24.pth",
#     },
#     {
#         "si": 128,
#         "path": "/bigtemp/rnx2bc/holstein-parc/checkpoints/L_32-quench_0p5_0p8-steps_1200-training-v2/tiny_simple_model-shallow_scaling_series_si_128/20251026-113359/best_epoch_28.pth",
#     },
#     {
#         "si": 256,
#         "path": "/bigtemp/rnx2bc/holstein-parc/checkpoints/L_32-quench_0p5_0p8-steps_1200-training-v2/tiny_simple_model-shallow_scaling_series_si_256/20251026-113417/best_epoch_23.pth",
#     },
#     {
#         "si": 512,
#         "path": "/bigtemp/rnx2bc/holstein-parc/checkpoints/L_32-quench_0p5_0p8-steps_1200-training-v2/tiny_simple_model-shallow_scaling_series_si_512/20251026-211721/best_epoch_22.pth",
#     },
#     {
#         "si": 1024,
#         "path": "/bigtemp/rnx2bc/holstein-parc/checkpoints/L_32-quench_0p5_0p8-steps_1200-training-v2/tiny_simple_model-shallow_scaling_series_si_1024/20251026-113626/best_epoch_25.pth",
#     },
# ]
# data_path = "/bigtemp/rnx2bc/holstein-parc/data/generated/L-32-g_0p5_0p8-rd-1e-04-si_2048-steps_32-zero_mom-MID-SCALING_SERIES/pickled/test.pkl"

data_si = 2048
prediction_interval = 2048

test_size = 25600
max_batch_size = 512
num_workers = 4
csv_path = "deep_results_2048.csv"


""" Process data """


data = unpickle_data(data_path)


## Get inputs and labels
data_label_shift = prediction_interval // data_si
inputs = [comp[:, :-data_label_shift] for comp in data]
labels = [comp[:, data_label_shift:] for comp in data]

## Flatten inputs and labels in batch and steps dim
inputs = [comp.reshape(-1, *comp.shape[2:]) for comp in inputs]
labels = [comp.reshape(-1, *comp.shape[2:]) for comp in labels]

num_elements = inputs[0].shape[0]
assert test_size <= num_elements, f"test_size {test_size} is greater than the number of elements {num_elements}"

## Sample the same random elements from inputs and labels
rng = np.random.default_rng(seed=42)
sample_indices = rng.choice(range(num_elements), size=test_size, replace=False)

inputs = [comp[sample_indices] for comp in inputs]
labels = [comp[sample_indices] for comp in labels]

""" Functions """


def process_checkpoint(checkpoint, inputs, labels, prediction_interval, max_batch_size):
    model_path = checkpoint["path"]
    si = checkpoint["si"]

    model_kwargs = unpickle_data(os.path.join(os.path.dirname(model_path), "model_kwargs.pkl"))
    model = HolsteinPARC(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))

    n_steps = prediction_interval // si

    preds = gen_preds(
        model, initial_conditions=inputs, n_steps=n_steps, max_batch_size=max_batch_size, suppress_output=True
    )
    preds = [comp[:, -1] for comp in preds]

    rho_preds, Q_preds, P_preds = preds
    rho_labels, Q_labels, P_labels = labels

    avg_rho_RMSE = np_rmse(rho_preds, rho_labels, axis=(-1, -2)).mean().item()
    avg_Q_RMSE = np_rmse(Q_preds, Q_labels, axis=-1).mean().item()
    avg_P_RMSE = np_rmse(P_preds, P_labels, axis=-1).mean().item()
    total_rmse = avg_rho_RMSE + avg_Q_RMSE + avg_P_RMSE

    return {
        "si": si,
        "model_path": model_path,
        "avg_rho_RMSE": avg_rho_RMSE,
        "avg_Q_RMSE": avg_Q_RMSE,
        "avg_P_RMSE": avg_P_RMSE,
        "total_RMSE": total_rmse,
    }


""" For each checkpoint, analyze """

print("Beginning analysis via multiprocessing")

with mp.Pool(processes=num_workers) as pool:
    results = pool.starmap(
        process_checkpoint,
        [(checkpoint, inputs, labels, prediction_interval, max_batch_size) for checkpoint in checkpoints],
    )

## Write results to CSV
with open(csv_path, "w", newline="") as f:
    fieldnames = ["si", "model_path", "avg_rho_RMSE", "avg_Q_RMSE", "avg_P_RMSE", "total_RMSE"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
