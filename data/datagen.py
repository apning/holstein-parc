from datetime import datetime
import copy
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

from src.utils import process_simulation_group, pickle_data

'''
Runs many simulations according to specified parameters
Then saves data from simulations into numpy arrays and pickles them into separate training, validation, and testing datasets for easy use later during training
'''

### Simulation Details
# ---------------------------------------

# A directory with this name under /generated must not already exist
name = "single-traj-L-8"

n_train_simulations = 1
n_val_simulations = 0
n_test_simulations = 0

n_workers = 1

# ---------------------------------------


### Parameters
# ---------------------------------------

# System size
L = 8

# After how many iterations of integration do we save one data point
# if save_period = 1 then we save every step of integration
save_period = 8

# Integration step size
dlt_t = 0.01

# How many steps of integration to perform
max_steps = 20000

# initial value of g
g_i = 0.5

# final value of g
g_f = 0.8

# AFTER solving for an initial self-consistent system with g_i:
# Whether to make Q and P random vectors with element-wise gaussian values centered at 0 with std randomness_level
random_Q_P = True

# See random_Q_P description above
randomness_level = 0.0001

# AFTER solving for an initial self-consistent system with g_i:
# Whether to zero out the momentum vector
# If use in tandom with random_Q_P then Q will be a random vector and P will be 0
zero_momentum = True
# ---------------------------------------


''' NO EDITABLE PARAMETERS AFTER HERE '''


### Some processing of parameters and variables
# ---------------------------------------


# This file should be already located in the /data directory, which is what we consider BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'generated', name, "data") # path for saving raw data
PICKLED_DIR = os.path.join(BASE_DIR, 'generated', name, "pickled") # path for processed and pickled data
# Create the data directory
os.makedirs(DATA_DIR, exist_ok=False)
# Create the pickled directory
os.makedirs(PICKLED_DIR, exist_ok=False)

# Create file to log variables and details
LOG_FILE_PATH = os.path.join(BASE_DIR, 'generated', name, 'info.txt')
with open(LOG_FILE_PATH, 'w') as log_file:
    for var_name, value in list(globals().items()):
        # Ignore built-in or special variables (starting with __) and functions
        if not var_name.startswith("__") and not callable(value):
            log_file.write(f"{var_name} = {value}\n")

# Ensure that random_Q_P and zero_momentum are integers since that is what the cpp code expects
if type(random_Q_P) != int:
    random_Q_P = 1 if random_Q_P else 0
if type(zero_momentum) != int:
    zero_momentum = 1 if zero_momentum else 0

# Define path to executable
EXEC_PATH = os.path.join(BASE_DIR, 'src', 'qdyn')
if not os.path.exists(EXEC_PATH):
    raise FileNotFoundError(f"Executable at {EXEC_PATH} does not exist!")

n_total_simulations = n_train_simulations + n_val_simulations + n_test_simulations
# ---------------------------------------


### Run numerical integration in parallel
# ---------------------------------------

print(f"Beginning integration with {n_workers} workers")

integration_start_time = datetime.now()

def run_executable(sim_num:int):
    '''
    Run the executable with a unique simulation number and all other arguments
    '''
    # We've suppressed outputs from the executables via stdout=subprocess.DEVNULL
    # TODO: It would be cool to have a tqdm-like progress bar as the parallelized processes complete
    subprocess.run([EXEC_PATH, DATA_DIR, str(sim_num), str(L), str(save_period), str(dlt_t), str(max_steps), str(g_i), str(g_f), str(random_Q_P), str(randomness_level), str(zero_momentum)], check=True, stdout=subprocess.DEVNULL)

# Use ProcessPoolExecutor to run executable in parallel
with ProcessPoolExecutor(max_workers=n_workers) as executor:
    executor.map(run_executable, range(n_total_simulations))

integration_finish_time = datetime.now()

print("Integration complete!")
print(f"Total elapsed time for integration: {integration_finish_time - integration_start_time}")
with open(LOG_FILE_PATH, 'a') as log_file:
    log_file.write("\n---------------------------------------\n")
    log_file.write(f"Total elapsed time for integration: {integration_finish_time - integration_start_time}\n")
    log_file.write("---------------------------------------\n\n")
# ---------------------------------------


### Processing data into numpy arrays
# ---------------------------------------
print("Processing data into numpy arrays")


processed_data = process_simulation_group(
    data_dir=DATA_DIR,
    L=L, 
    n_sims=n_total_simulations,
    n_workers=n_workers
)
process_finish_time = datetime.now()

print("Data processing complete!")
print(f"Total elapsed time for data processing: {process_finish_time-integration_finish_time}")
with open(LOG_FILE_PATH, 'a') as log_file:
    log_file.write("\n---------------------------------------\n")
    log_file.write(f"Total elapsed time for data processing: {process_finish_time-integration_finish_time}\n")
    log_file.write("---------------------------------------\n\n")
# ---------------------------------------

### Pickling data
# ---------------------------------------

if n_train_simulations:
    pickle_data(os.path.join(PICKLED_DIR, 'train.pkl'), processed_data[:n_train_simulations])

if n_val_simulations:
    pickle_data(os.path.join(PICKLED_DIR, 'val.pkl'), processed_data[n_train_simulations:n_train_simulations+n_val_simulations])

if n_test_simulations:
    pickle_data(os.path.join(PICKLED_DIR, 'test.pkl'), processed_data[n_train_simulations+n_val_simulations:n_train_simulations+n_val_simulations+n_test_simulations])

# ---------------------------------------

print(f"Data pickled and saved! All complete!")
with open(LOG_FILE_PATH, 'a') as log_file:
    log_file.write("\nData pickled and saved! All complete!\n")

