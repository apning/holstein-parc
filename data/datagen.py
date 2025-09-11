import argparse
from datetime import datetime
import subprocess
import os
import gc
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
from functools import partial

import numpy as np

from src.datagen_utils import (
    process_simulation_group,
    cycle_sample_range,
)
from src.data_utils import (
    get_component_scalars_dict,
    get_data_scalars_dict,
    get_delta_scalars_dict,
    get_derivative_scalars_dict,
    pickle_data,
)
from src.utils import get_project_root, str2bool, timer

"""
Runs many simulations according to specified parameters
Then saves data from simulations into numpy arrays and pickles them into separate training, validation, and testing datasets for easy use later during training
"""


def run_executable(
    exec_path,
    data_dir,
    save_interval,
    save_mid_interval,
    L,
    dlt_t,
    pre_steps,
    saved_steps,
    g_i,
    g_f,
    randomness_level,
    zero_disp,
    zero_mom,
    onsite_V,
    sim_num,
    phase,
):
    # We've suppressed outputs from the executables via stdout=subprocess.DEVNULL
    subprocess.run(
        [
            exec_path,
            data_dir,
            str(sim_num),
            str(phase),
            str(save_interval),
            str(save_mid_interval),
            str(L),
            str(dlt_t),
            str(pre_steps),
            str(saved_steps),
            str(g_i),
            str(g_f),
            str(randomness_level),
            str(zero_disp),
            str(zero_mom),
            str(onsite_V),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main(
    n_train: int,
    n_val: int,
    n_test: int,
    n_workers: int,
    dlt_t: float,
    save_interval: int,
    pre_steps: int,
    saved_steps: int,
    L: int,
    g_i: float,
    g_f: float,
    randomness_level: float,
    name_override: None | str,
    custom_name_postfix: None | str,
    notes: None | str,
    save_mid_interval: bool,
):
    """THESE PARAMETERS SHOULD PROBABLY NOT BE TOUCHED"""

    ### Parameters not to modify
    # ---------------------------------------

    # scale of random onsite_V vector (domain of uniform dist)
    # Keep set to 0
    onsite_V_term = 0

    # Whether to zero out momentum vector after solving for self-consistent Q
    # Happens BEFORE adding random noise to Q
    # DO NOT SET TO TRUE
    zero_displacement = False

    # Whether to zero out momentum vector after solving for self-consistent Q
    # Can keep to True
    zero_momentum = True

    # NOT used in generation of data. Pickled for later derivative caclulations and used to generate data_scalars
    # G is g_f. For now at least K0 is hard-coded at 0.3 and G1 at 0.8 * G. This is because it is currently hardcoded this way in the cpp code as well
    physical_params = {"K0": 0.3, "G": g_f, "G1": 0.8 * g_f}
    # ----------------------------------------

    ### Some processing of parameters and variables
    # ---------------------------------------

    ## Creating name
    # A directory with this name under /generated must not already exist

    def format_decimal_as_str(x):
        x = round(x, 3)
        return str(x).replace(".", "p") if x % 1 else str(int(x))

    name = f"L-{L}-g_{format_decimal_as_str(g_i)}_{format_decimal_as_str(g_f)}-rd-{randomness_level:.0e}-si_{save_interval}"
    if pre_steps:
        name += f"-ps_{pre_steps}"
    name += f"-steps_{saved_steps}"
    if zero_displacement:
        name += "-zero_disp"
    if zero_momentum:
        name += "-zero_mom"
    if save_mid_interval:
        name += "-MID"
    if custom_name_postfix:
        name += custom_name_postfix

    if name_override is not None:
        name = name_override

    if onsite_V_term:
        raise ValueError("non-zero onsite_V currently not supported")

    # This file should be located in the /data directory, which is what we consider BASE_DIR
    BASE_DIR = os.path.join(get_project_root(), "data")
    DATA_DIR = os.path.join(BASE_DIR, "generated", name, "data")  # path for saving raw data
    PICKLED_DIR = os.path.join(BASE_DIR, "generated", name, "pickled")  # path for processed and pickled data
    # Create the data directory
    os.makedirs(DATA_DIR, exist_ok=False)
    # Create the pickled directory
    os.makedirs(PICKLED_DIR, exist_ok=False)

    # Create file to log variables and details
    LOG_FILE_PATH = os.path.join(BASE_DIR, "generated", name, "info.txt")
    with open(LOG_FILE_PATH, "w") as log_file:
        for var_name, value in list(locals().items()):
            # Ignore built-in or special variables (starting with __) and functions
            if not var_name.startswith("__") and not callable(value):
                log_file.write(f"{var_name} = {value}\n")

    # Ensure that these boolean parameters are integers since that is what the cpp code expects
    if type(zero_displacement) is not int:
        zero_displacement = 1 if zero_displacement else 0
    if type(zero_momentum) is not int:
        zero_momentum = 1 if zero_momentum else 0
    if type(save_mid_interval) is not int:
        save_mid_interval = 1 if save_mid_interval else 0

    # Define path to executable
    EXEC_PATH = os.path.join(BASE_DIR, "data_src", "qdyn")
    if not os.path.exists(EXEC_PATH):
        raise FileNotFoundError(f"Executable at {EXEC_PATH} does not exist!")

    n_total_simulations = n_train + n_val + n_test

    ## Create phase values for each
    train_phases = cycle_sample_range(0, save_interval, n_train, shuffle=True)
    val_phases = cycle_sample_range(0, save_interval, n_val, shuffle=True)
    test_phases = cycle_sample_range(0, save_interval, n_test, shuffle=True)
    all_phases = np.concatenate((train_phases, val_phases, test_phases))

    # ---------------------------------------

    ### Run numerical integration in parallel
    # ---------------------------------------

    print(f"Beginning integration with {n_workers} workers")

    integration_start_time = datetime.now()

    worker_fn = partial(
        run_executable,
        EXEC_PATH,
        DATA_DIR,
        save_interval,
        save_mid_interval,
        L,
        dlt_t,
        pre_steps,
        saved_steps,
        g_i,
        g_f,
        randomness_level,
        zero_displacement,
        zero_momentum,
        onsite_V_term,
    )

    # Use ProcessPoolExecutor to run executable in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        executor.map(worker_fn, range(n_total_simulations), all_phases)

    integration_finish_time = datetime.now()

    print("Integration complete!")
    print(f"Total elapsed time for integration: {integration_finish_time - integration_start_time}")
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write("\n---------------------------------------\n")
        log_file.write(f"Total elapsed time for integration: {integration_finish_time - integration_start_time}\n")
        log_file.write("---------------------------------------\n\n")
    # ---------------------------------------

    ### Processing data into numpy arrays and then pickle
    # ---------------------------------------
    print("Processing data into numpy arrays")

    processed_data = process_simulation_group(data_dir=DATA_DIR, L=L, n_sims=n_total_simulations, n_workers=n_workers)
    process_finish_time = datetime.now()

    print("Data processing complete!")
    print(f"Total elapsed time for data processing: {process_finish_time - integration_finish_time}")
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write("\n---------------------------------------\n")
        log_file.write(f"Total elapsed time for data processing: {process_finish_time - integration_finish_time}\n")
        log_file.write("---------------------------------------\n\n")

    ## Pickle data
    if n_train:
        pickle_data(os.path.join(PICKLED_DIR, "train.pkl"), tuple(component[:n_train] for component in processed_data))

    if n_val:
        pickle_data(
            os.path.join(PICKLED_DIR, "val.pkl"),
            tuple(component[n_train : n_train + n_val] for component in processed_data),
        )

    if n_test:
        pickle_data(
            os.path.join(PICKLED_DIR, "test.pkl"),
            tuple(component[n_train + n_val : n_train + n_val + n_test] for component in processed_data),
        )

    print("Data pickled and saved!")

    if n_train:
        with timer():
            print("Calculating data scalars")
            if save_mid_interval:
                data_scalars = {}
                data_scalars.update(get_component_scalars_dict(processed_data[:n_train]))
                data_scalars.update(get_delta_scalars_dict(processed_data[:n_train]))
            else:
                data_scalars = get_data_scalars_dict(data=processed_data[:n_train], **physical_params)

    # Clear away that data
    del processed_data
    gc.collect()
    # ---------------------------------------

    ### If mid-interval saving was used, process into numpy and pickle
    # ---------------------------------------

    if save_mid_interval:
        print("Processing mid-interval data into numpy arrays")

        mid_process_start_time = datetime.now()
        mid_processed_data = process_simulation_group(
            data_dir=DATA_DIR, L=L, n_sims=n_total_simulations, n_workers=n_workers, file_prefix="mid_data_sim_"
        )
        mid_process_finish_time = datetime.now()

        print("Mid-interval data processing complete!")
        print(
            f"Total elapsed time for mid-interval data processing: {mid_process_finish_time - mid_process_start_time}"
        )
        with open(LOG_FILE_PATH, "a") as log_file:
            log_file.write("\n---------------------------------------\n")
            log_file.write(
                f"Total elapsed time for mid-interval data processing: {mid_process_finish_time - mid_process_start_time}\n"
            )
            log_file.write("---------------------------------------\n\n")

        if n_train:
            pickle_data(
                os.path.join(PICKLED_DIR, "mid_train.pkl"),
                tuple(component[:n_train] for component in mid_processed_data),
            )

        if n_val:
            pickle_data(
                os.path.join(PICKLED_DIR, "mid_val.pkl"),
                tuple(component[n_train : n_train + n_val] for component in mid_processed_data),
            )

        if n_test:
            pickle_data(
                os.path.join(PICKLED_DIR, "mid_test.pkl"),
                tuple(component[n_train + n_val : n_train + n_val + n_test] for component in mid_processed_data),
            )

        print("Mid-interval data pickled and saved!")

        if n_train:
            with timer():
                print("Calculating derivative data scalars with mid-interval data")
                data_scalars.update(get_derivative_scalars_dict(mid_processed_data[:n_train], **physical_params))

        del mid_processed_data
        gc.collect()
    # ---------------------------------------

    if n_train:
        pickle_data(os.path.join(PICKLED_DIR, "data_scalars.pkl"), data_scalars)

    # Save physical parameters for later use when calculating derivatives
    pickle_data(os.path.join(PICKLED_DIR, "physical_params.pkl"), physical_params)

    print("All complete!")

    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write("\nAll complete!\n")


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    ### Simulation Details
    # ---------------------------------------
    parser.add_argument("--n_train", default=None, type=int)
    parser.add_argument("--n_val", default=None, type=int)
    parser.add_argument("--n_test", default=None, type=int)
    parser.add_argument("--n_workers", default=24, type=int)

    ### Parameters
    # ---------------------------------------
    # Integration step size (0.01 is usually fine)
    parser.add_argument("--dlt_t", default=0.01, type=float)

    # After how many iterations of integration do we save one data point
    # if save_interval = 1 then we save every step of integration
    parser.add_argument("--save_interval", default=256, type=int)

    # How many steps to take before we start saving data points
    # These steps are of dlt_t size, not dlt_t * save_interval size
    parser.add_argument("--pre_steps", default=0, type=int)

    # How many steps of integration to save. The initial state does not count as a step, so there will be saved_steps + 1 total data points
    # Will save this many steps after pre_steps steps has been integrated
    parser.add_argument("--saved_steps", default=600, type=int)

    # System size
    parser.add_argument("--L", default=32, type=int)

    # initial value of g
    parser.add_argument("--g_i", default=0.0, type=float)

    # final value of g
    parser.add_argument("--g_f", default=1.0, type=float)

    # Amount of random noise to add to Q and P after solving for self consistency. If desired, 0.0001 (1e-4) is a good value
    parser.add_argument("--randomness_level", default=1e-4, type=float)

    # Don't specify (leave as None) to use auto-generated name
    parser.add_argument("--name_override", default=None, type=str)

    # Don't specify (leave as None) to not use
    parser.add_argument("--custom_name_postfix", default=None, type=str)

    # Don't specify (leave as None) to not use
    parser.add_argument("--notes", default=None, type=str)

    # Whether to save a separate dataset made up of mid-interval datapoints
    parser.add_argument(
        "--save_mid_interval",
        nargs="?",
        const=True,  # bare flag → True
        default=True,  # flag absent → True
        type=str2bool,  # explicit value parsed via helper
    )

    args = parser.parse_args()

    args_dict = vars(args)

    if isinstance(args_dict["name_override"], str) and len(args_dict["name_override"]) == 0:
        args_dict["name_override"] = None

    return args_dict


if __name__ == "__main__":
    # Or else OOMs while saving as each worker requests a huge virtual space
    set_start_method("spawn", force=True)

    args_dict = parse_args()

    main(**args_dict)
