import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


from types import NoneType


from src.utils import select_best_device, Running_Avg, Multi_Running_Avgs, str_formatted_datetime, Min_n_Items
from src.data_utils import HolsteinDataset, unpickle_data, Virtual_Epoch_loader
from src.training_utils import calc_val_RMSE, get_warm_up_lr_scheduler, batch_to_output, batch_to_loss
from src.modeling import HolsteinStepSeparate


### Default config variables begin
# ---------------------------------------

# Run details
name = "test-1a"
sub_name = "subtest-1a"

## Data variables
# data_name should be the name of the directory within the /data/generated directory storing the desired dataset. Aka the 'name' variable within datagen.py
data_name = None

## Training details
batch_size = 64
epochs = 100
warmup_steps = 1000
lr = 1e-4
weight_decay = 1e-5
n_step_prediction = 1 # How many steps to set prediction target. If 1 then is next-step prediction
# If n_step_prediction > 1 and True then train on all steps in-between as well. Eg if n_step_prediction == 3 and True then train on steps 1, 2, and 3.
# If n_step_prediction == 1 then predict_multiple_steps makes no difference
predict_multiple_steps = False

# If not None, uses the virtual epoch loader with specified number of batches per epoch. The virtual epoch loader wraps around the dataloader to create epochs with a certain desired batch size
# Internally, the virtual epoch loader iterates through the actual dataloader irrespective of the boundaries of the virtual epochs
# Convenient for comparing datasets with different sizes
# If none, does not use virtual epoch loader
virtual_epoch_size = None


# Number of simulations to use from the training data for training. If None uses all the sims in training data
n_sims = None

# For use with n_sims. If True, the subset of simulations from data will be random
sample_random_subset = False

# Number of steps to use from each simulation (for train, val, and test). If None, uses all steps (or rather, for implementation purposes, uses a number of steps for all simulations equal to the number of steps in the simulation with least steps)
# If not None, uses the first n_steps_per_sim steps of each simulation
n_steps_per_sim = None

## Model details
n_features = 128
pretrained_path = None # If None, we initialize from random

# ---------------------------------------



### Parsing config files
## Credit this section (Parsing config files) to https://github.com/karpathy/nanoGPT
# ---------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, NoneType))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# ---------------------------------------


### Save path variables
# ---------------------------------------
# Get path of directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
START_TIME = str_formatted_datetime()

# When we save we will save in the form
# /BASE_DIR/<save type>/SAVE_PATH_POSTFIX
# Where <save type> could be 'checkpoints', 'runs', etc
SAVE_PATH_POSTIFX = os.path.join(name, sub_name, START_TIME)

CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints', SAVE_PATH_POSTIFX)
os.makedirs(CHECKPOINTS_DIR, exist_ok=False)

DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated', data_name, 'pickled')
# ---------------------------------------


### Load datasets
# ---------------------------------------
train_data = unpickle_data(os.path.join(DATA_DIR, 'train.pkl'))
val_data = unpickle_data(os.path.join(DATA_DIR, 'val.pkl'))
test_data = unpickle_data(os.path.join(DATA_DIR, 'test.pkl'))

train_set = HolsteinDataset(data=train_data,
                            label_step_count=n_step_prediction,
                            multi_step_labels=predict_multiple_steps,
                            n_sims=n_sims,
                            sample_random_subset=sample_random_subset,
                            n_steps=n_steps_per_sim
                            )
val_set = HolsteinDataset(data=val_data,
                            label_step_count=n_step_prediction,
                            multi_step_labels=predict_multiple_steps,
                            n_steps=n_steps_per_sim
                            )
# For consisten[cy between results, we always do next-step-prediction for testing
test_set = HolsteinDataset(data=test_data,
                            label_step_count=1,
                            multi_step_labels=False,
                            n_steps=n_steps_per_sim
                            )

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

if virtual_epoch_size is not None:
    train_loader = Virtual_Epoch_loader(train_loader, batches_per_epoch=virtual_epoch_size)
# ---------------------------------------


### Load model
# ---------------------------------------
device = select_best_device(mode='m')
print(f"Using device: {device}")

model = HolsteinStepSeparate(n_features=n_features)

if pretrained_path is not None:
    model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True))

model.to(device)
# ---------------------------------------


### Training Setup
# ---------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_func = get_warm_up_lr_scheduler(warmup_steps=warmup_steps)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, scheduler_func)

## Create a TensorBoard writer
LOGDIR = os.path.join(BASE_DIR, 'runs', SAVE_PATH_POSTIFX)
writer = SummaryWriter(LOGDIR)
# ---------------------------------------


### Training
# ---------------------------------------

# Used to constantly keep a record of the minimum 5 validation loss scores encountered so far
min_n_val = Min_n_Items(n=5)

# Used to check if there is a previous checkpoint to delete before saving better checkpoint
prev_checkpoint_path = None

training_start_time = datetime.now()

for epoch in tqdm(range(1, epochs+1), desc="Epochs"):

    # Initialize running average tracker for each component training loss
    running_avgs = Multi_Running_Avgs(n=3)

    # Train
    for batch in tqdm(train_loader, desc="Batches", leave=False):

        model.train()
        optimizer.zero_grad()
        loss, separates = batch_to_loss(model, batch, device, return_loss_separates=True, n_step=n_step_prediction, return_multiple_steps=predict_multiple_steps)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_avgs.add(separates)

    # Average training losses for this epoch
    train_loss_rho, train_loss_Q, train_loss_P = running_avgs.get_running_avgs()
    train_loss_total = running_avgs.get_summed_avgs()

    # Write train stats to TensorBoard
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    writer.add_scalar('train/total_loss', train_loss_total, epoch)
    writer.add_scalar('train/rho_loss', train_loss_rho, epoch)
    writer.add_scalar('train/Q_loss', train_loss_Q, epoch)
    writer.add_scalar('train/P_loss', train_loss_P, epoch)

    val_loss_total, val_loss_rho, val_loss_Q, val_loss_P = calc_val_RMSE(model, val_loader, device, n_step=n_step_prediction, return_multiple_steps=predict_multiple_steps)

    writer.add_scalar('val/total_loss', val_loss_total, epoch)
    writer.add_scalar('val/rho_loss', val_loss_rho, epoch)
    writer.add_scalar('val/Q_loss', val_loss_Q, epoch)
    writer.add_scalar('val/P_loss', val_loss_P, epoch)

    min_n_val.record_epoch(val=val_loss_total, epoch_num=epoch)

    # Save checkpoint if lowest validation loss and delete the old saved checkpoint
    if val_loss_total <= min_n_val.get_smallest_val():
        save_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)

        # Delete previous checkpoint
        if prev_checkpoint_path is not None:
            os.remove(prev_checkpoint_path)
        prev_checkpoint_path = save_path

training_end_time = datetime.now()
training_time_elapsed = training_end_time-training_start_time

print("Training complete!")
print(f"Time Elapsed: {training_time_elapsed}")
print(min_n_val.__str__())
# ---------------------------------------


### Test model on best checkpoint
# ---------------------------------------
model.load_state_dict(torch.load(save_path, weights_only=True))
model.to(device)

# For consistency of comparison, we always test with n_step=1 (next step prediction)
test_loss_total, test_loss_rho, test_loss_Q, test_loss_P = calc_val_RMSE(model, test_loader, device, n_step=1, return_multiple_steps=False)
print(f"Total test loss: {test_loss_total}")
# ---------------------------------------


### Record hyperparameters and result metrics
# ---------------------------------------

# Create a dict with most relevant metrics
min_val_loss, min_val_epoch = min_n_val.get_smallest_val(include_epoch=True)
final_metrics = {'min val loss':min_val_loss, 'min val epoch':min_val_epoch, 'total test loss':test_loss_total, 'rho test loss':test_loss_rho, 'Q test loss':test_loss_Q, 'P test loss':test_loss_P}

# Log values in Tensorboard
writer.add_hparams(config, final_metrics)

writer.close()

# I don't trust tensorboard add_hparams() very much... often logged data doesn't show up
# So we do this as backup

txt_log_path = os.path.join(LOGDIR, 'log.txt')
with open(txt_log_path, 'a') as f:
    f.write("HYPERPARAMETERS\n")
    f.write("\n---------------------------------------\n")
    for k, v in config.items():
        f.write(f"{k}\t=\t{v}\n")
    f.write("\n---------------------------------------\n")
    f.write("METRICS\n")
    f.write("\n---------------------------------------\n")
    f.write(f"Training time elapsed: {training_time_elapsed}\n")
    for k, v in final_metrics.items():
        f.write(f"{k}\t=\t{v}\n")

# ---------------------------------------