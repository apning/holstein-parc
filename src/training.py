# Standard library imports
import os
import warnings
from datetime import datetime
from typing import Sequence

# Third-party imports
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Local imports
from src.config import Config
from src.data_utils import HolsteinDataset, Virtual_Epoch_loader, pickle_data, unpickle_data
from src.get_settings import get_cirriculum
from src.modeling import HolsteinPARC
from src.training_utils import (
    batch_to_loss,
    calc_val_stats,
    create_interpolated_lr_dict_for_optim,
    get_cosine_annealing_restart_warmup_scheduler,
)
from src.utils import (
    Min_n_Items,
    Multi_Running_Avgs,
    get_project_root,
    select_best_device,
    str_formatted_datetime,
)


def train(config: Config):
    """
    Main training function for Holstein PARC models.

    Handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization
    - Training loop with curriculum learning
    - Validation and checkpointing
    - Logging to TensorBoard

    Args:
        config (Config): Configuration object containing all training parameters.
    """

    ### Processing/Checking Arguments
    # ---------------------------------------

    n_datasets = len(config.data_names)

    cirriculum = get_cirriculum(cirriculum_name=config.cirriculum_name, noise_coeff=config.input_noise_scale_coeff)

    epochs = config.epochs
    if epochs is None:
        epochs = sum(stage["epochs"] for stage in cirriculum)

    if config.virtual_epoch_size is None and n_datasets > 1:
        warnings.warn(
            "Multiple datasets used but virtual_epoch_size was None. If all datasets do not have the same size then their dataloaders will not complete iteration of their batches (only the dataset with the least number of elements will fully complete iteration)"
        )

    if config.data_nicknames is not None:
        data_nicknames = config.data_nicknames
    else:
        if n_datasets > 1:
            data_nicknames = config.data_names
        else:
            data_nicknames = [""]

    batch_sizes = config.batch_size
    if not isinstance(batch_sizes, Sequence):
        batch_sizes = [batch_sizes] * n_datasets

    val_batch_sizes = config.val_batch_size
    if val_batch_sizes is None:
        val_batch_sizes = batch_sizes
    if not isinstance(val_batch_sizes, Sequence):
        val_batch_sizes = [val_batch_sizes] * n_datasets

    if config.simple_cnn and config.use_derivatives:
        raise ValueError("config.simple_cnn and config.use_derivatives cannot both be True!")

    # ---------------------------------------

    ### Save path variables
    # ---------------------------------------
    if config.START_TIME is None:
        config.START_TIME = str_formatted_datetime()

    # When we save we will save in the form
    # /BASE_DIR/<save type>/SAVE_PATH_POSTFIX
    # Where <save type> could be 'checkpoints', 'runs', etc
    SAVE_PATH_POSTFIX = os.path.join(
        config.name, config.sub_name if config.sub_name is not None else "-", config.START_TIME
    )

    # Get path of project root
    BASE_DIR = get_project_root()

    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints", SAVE_PATH_POSTFIX)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=config.resume)

    if config.data_names is None:
        raise ValueError("data_names must be specified and cannot be None.")

    DATA_DIRS = [BASE_DIR / "data" / "generated" / data_name / "pickled" for data_name in config.data_names]
    # ---------------------------------------

    ### Load datasets
    # ---------------------------------------

    train_datas = [unpickle_data(os.path.join(data_dir, "train.pkl")) for data_dir in DATA_DIRS]
    mid_train_datas = (
        [unpickle_data(os.path.join(data_dir, "mid_train.pkl")) for data_dir in DATA_DIRS]
        if config.use_derivatives
        else [None] * n_datasets
    )
    val_datas = [unpickle_data(os.path.join(data_dir, "val.pkl")) for data_dir in DATA_DIRS]

    val_sets = [
        HolsteinDataset(
            data=val_data,
            label_step_count=config.val_n_step_prediction,
            multi_step_labels=config.val_predict_multiple_steps,
            n_steps=config.val_n_steps_per_sim,
            return_derivatives=False,
        )
        for val_data in val_datas
    ]

    val_loaders = [
        DataLoader(val_set, batch_size=val_batch_size, shuffle=True, num_workers=config.n_workers)
        for val_set, val_batch_size in zip(val_sets, val_batch_sizes)
    ]

    # ---------------------------------------

    ### Load model
    # ---------------------------------------
    device = select_best_device(mode="m")
    print(f"Using device: {device}")

    # Create model kwargs dict
    model_kwargs = config.get_model_kwargs()
    # Save model kwargs
    pickle_data(os.path.join(CHECKPOINTS_DIR, "model_kwargs.pkl"), model_kwargs)

    # Instantiate model
    model = HolsteinPARC(**model_kwargs)

    if config.pretrained_path is not None:
        model.load_state_dict(torch.load(config.pretrained_path, map_location=torch.device("cpu"), weights_only=True))

    model.to(device)

    if not config.resume:
        save_path = os.path.join(CHECKPOINTS_DIR, "initialized.pth")
        torch.save(model.state_dict(), save_path)

    # ---------------------------------------

    ### Training Setup
    # ---------------------------------------

    if config.fine_tuning_layerwise_lr_coeff is None:
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        start_lr = config.fine_tuning_layerwise_lr_coeff * config.lr
        if not config.simple_cnn:
            optim_params = create_interpolated_lr_dict_for_optim(
                model.differentiator, start_lr, config.lr
            ) + create_interpolated_lr_dict_for_optim(model.integrator, start_lr, config.lr)
        else:
            optim_params = create_interpolated_lr_dict_for_optim(model.cnn, start_lr, config.lr)

        optimizer = optim.AdamW(optim_params, weight_decay=config.weight_decay)

    ## Create scheduler
    assert config.virtual_epoch_size is not None, (
        "Use of custom staged cosine annealing with restarts scheduler is currently not available without using a virtual epoch loader as calculations of exact stage lengths would be more complicated without virtual epoch loader and this has yet to be implemented"
    )  # More complicated as with different dataset settings (eg different step prediction values) the dataset size can vary
    stage_lengths = [
        stage["epochs"] * config.virtual_epoch_size * (stage.get("virtual_epoch_multiplier") or 1)
        for stage in cirriculum
    ]
    scheduler = get_cosine_annealing_restart_warmup_scheduler(
        optimizer,
        stage_lengths=stage_lengths,
        warmup_steps=config.warmup_steps,
        eta_min=config.lr * config.eta_min_scalar,
    )

    # Scheduler state must be restored BEFORE optimizer state: https://github.com/pytorch/pytorch/issues/119168
    if config.lr_scheduler_path is not None:
        scheduler.load_state_dict(torch.load(config.lr_scheduler_path, weights_only=True))

    if config.optimizer_path is not None:
        optimizer.load_state_dict(torch.load(config.optimizer_path, map_location=device, weights_only=True))

    # Calculate cumulative epoch numbers for each curriculum stage
    # This helps determine when to switch between stages during training
    cumulative_epochs = 1
    for stage in cirriculum:
        stage["cumulative_epochs"] = cumulative_epochs
        cumulative_epochs += stage["epochs"]

    ## Create a TensorBoard writer
    LOGDIR = os.path.join(BASE_DIR, "runs", SAVE_PATH_POSTFIX)
    writer = SummaryWriter(LOGDIR)

    # Indicates whether each dataset should use gradient checkpointing
    whether_checkpointing = [False] * n_datasets

    # Obtain physical parameters for gradient calculation
    physical_params = [unpickle_data(data_dir / "physical_params.pkl") for data_dir in DATA_DIRS]

    # ---------------------------------------

    ### Log config variables and cirriculum
    # ---------------------------------------
    txt_log_path = os.path.join(LOGDIR, "log.txt")
    with open(txt_log_path, "a") as f:
        f.write("CONFIG\n")
        f.write("\n---------------------------------------\n")
        for k, v in config.__dict__.items():
            f.write(f"{k}\t=\t{v}\n")
        f.write("\n---------------------------------------\n")
        f.write("CIRRICULUM\n")
        for stage in cirriculum:
            f.write(f"{stage}\n")
        f.write("\n---------------------------------------\n")
    # ---------------------------------------

    ### Training
    # ---------------------------------------

    # If these were set but training wasn't for resuming a previous run, they must be set to None now or else the training loop will clear the original files
    if not config.resume:
        config.pretrained_path = None
        config.optimizer_path = None
        config.lr_scheduler_path = None

    # Used to constantly keep a record of the minimum 5 validation loss scores encountered so far
    min_n_val = Min_n_Items(n=5)
    min_n_val_cdw_order = Min_n_Items(n=5)  # Similar but for cdw order difference

    # Used to check if there is a previous checkpoint to delete before saving better checkpoint
    # Track paths of best checkpoints for deletion when better ones are saved
    prev_checkpoint_path = None
    prev_cdw_order_checkpoint_path = None

    training_start_time = datetime.now()

    first_epoch_after_resume = config.resume
    for epoch in tqdm(range(config.next_epoch, epochs + 1), desc="Epochs", disable=config.suppress_tqdm):
        # A very compact way to basically find if there are any stages in the cirriculum that starts on this epoch. If so, returns the dict for that stage
        # If not, returns None
        # cumulative_epochs values have to be unique between the stage dicts
        stage = next((stage for stage in cirriculum if stage["cumulative_epochs"] == epoch), None)

        # For resumed training: find the current stage if we didn't start at a stage boundary
        if first_epoch_after_resume and stage is None:
            for _stage in cirriculum:
                if _stage["cumulative_epochs"] <= epoch:
                    stage = _stage
                else:
                    break

        # Create a new train loader if it is time for a new stage
        if stage is not None:
            n_step_prediction = stage["n_step"]
            training_input_noise_std = stage["noise"]

            train_sets = [
                HolsteinDataset(
                    data=train_data,
                    label_step_count=n_step_prediction,
                    multi_step_labels=config.predict_multiple_steps,
                    n_sims=config.n_sims,
                    n_steps=config.n_steps_per_sim,
                    return_derivatives=config.use_derivatives,
                    deriv_data=mid_train_data,
                    input_gaussian_noise_std=training_input_noise_std,
                    disable_initial_state_noise=config.disable_initial_state_noise,
                    **physical_param,
                )
                for train_data, mid_train_data, physical_param in zip(train_datas, mid_train_datas, physical_params)
            ]

            train_loaders = [
                DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=config.n_workers)
                for train_set, batch_size in zip(train_sets, batch_sizes)
            ]

            if config.virtual_epoch_size is not None:
                train_loaders = [
                    Virtual_Epoch_loader(
                        train_loader,
                        batches_per_epoch=config.virtual_epoch_size * (stage.get("virtual_epoch_multiplier") or 1),
                    )
                    for train_loader in train_loaders
                ]

        # Initialize running average trackers for each component training loss for this epoch for each dataset
        all_running_avgs = [Multi_Running_Avgs() for _ in range(n_datasets)]

        # Train
        for batches in tqdm(zip(*train_loaders), desc="Batches", leave=False, disable=config.suppress_tqdm):
            model.train()
            optimizer.zero_grad()

            for data_idx, batch in enumerate(batches):
                while True:
                    try:
                        loss, loss_dict = batch_to_loss(
                            model=model,
                            batch=batch,
                            device=device,
                            n_step=n_step_prediction,
                            return_multiple_steps=config.predict_multiple_steps,
                            deriv_loss_coeff=config.deriv_loss_coeff,
                            separate_rho_diag=config.separate_rho_diag,
                            return_loss_dict=True,
                            stepwise_loss_scale_coeff=config.stepwise_loss_scale_coeff,
                            dtype=config.dtype,
                            checkpointing=whether_checkpointing[data_idx],
                        )
                        break
                    except torch.cuda.OutOfMemoryError:
                        # Automatically enable gradient checkpointing on OOM
                        if whether_checkpointing[data_idx]:
                            raise torch.cuda.OutOfMemoryError(
                                f"Gradient checkpointing already enabled but still OOM! Dataset: {data_nicknames[data_idx]}, n_step_prediction: {n_step_prediction}, batch size: {batch_sizes[data_idx]}"
                            )
                        with open(txt_log_path, "a") as f:
                            f.write(
                                f"Dataset {data_nicknames[data_idx]} resulted in OOM error with n_step_prediction {n_step_prediction} and batch size {batch_sizes[data_idx]}. Enabling gradient checkpointing for this dataset and retrying batch.\n\n"
                            )

                        # Delete references to these if they exist so their memory can be freed
                        try:
                            del loss
                        except NameError:
                            pass
                        try:
                            del loss_dict
                        except NameError:
                            pass

                        torch.cuda.empty_cache()

                        whether_checkpointing[data_idx] = True

                loss = loss / n_datasets
                loss.backward()
                all_running_avgs[data_idx].add(loss_dict)

            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            scheduler.step()

        # Logs only the last LR for this epoch
        # Also logs only the last lr group if optimizer has mulitple different lr groups
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[-1], epoch)

        # Average training losses for this epoch
        for data_nickname, running_avgs in zip(data_nicknames, all_running_avgs):
            for k, v in running_avgs.get_running_avgs().items():
                writer.add_scalar(f"train/{data_nickname}/{k}_loss", v, epoch)

        ## Calculate Validation Statistics

        all_val_stats = [
            calc_val_stats(
                model,
                val_loader,
                device,
                n_step=config.val_n_step_prediction,
                return_multiple_steps=config.val_predict_multiple_steps,
                return_as_dict=True,
                return_cdw_order_rmse=True,
            )
            for val_loader in val_loaders
        ]

        for data_nickname, val_stats in zip(data_nicknames, all_val_stats):
            for k, v in val_stats.items():
                writer.add_scalar(f"val/{data_nickname}/{k}", v, epoch)

        all_dataset_total_val_RMSE = sum(val_stats["total RMSE"] for val_stats in all_val_stats)
        all_dataset_total_val_CDW_order_RMSE = sum(val_stats["total CDW order RMSE"] for val_stats in all_val_stats)

        min_n_val.record_epoch(val=all_dataset_total_val_RMSE, epoch_num=epoch)
        min_n_val_cdw_order.record_epoch(val=all_dataset_total_val_CDW_order_RMSE, epoch_num=epoch)

        # Save checkpoint every certain number of epochs
        if config.save_checkpoint_period is not None and epoch % config.save_checkpoint_period == 0:
            save_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        # Save checkpoint if lowest validation loss and delete the old saved checkpoint
        if all_dataset_total_val_RMSE <= min_n_val.get_smallest_val():
            save_path = os.path.join(CHECKPOINTS_DIR, f"best_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

            # Delete previous checkpoint
            if prev_checkpoint_path is not None:
                os.remove(prev_checkpoint_path)
            prev_checkpoint_path = save_path

        if all_dataset_total_val_CDW_order_RMSE <= min_n_val_cdw_order.get_smallest_val():
            save_path = os.path.join(CHECKPOINTS_DIR, f"best_cdw_order_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

            # Delete previous checkpoint
            if prev_cdw_order_checkpoint_path is not None:
                os.remove(prev_cdw_order_checkpoint_path)
            prev_cdw_order_checkpoint_path = save_path

        # Save full training state for resuming
        # This includes model, optimizer, scheduler states and updated config
        # Also delete the training state for the previous epoch
        old_pretrained_path = config.pretrained_path
        old_optimizer_path = config.optimizer_path
        old_lr_scheduler_path = config.lr_scheduler_path

        config.pretrained_path = os.path.join(CHECKPOINTS_DIR, f"LAST_epoch_{epoch}.pth")
        config.optimizer_path = os.path.join(CHECKPOINTS_DIR, f"LAST_epoch_{epoch}_OPTIMIZER.pth")
        config.lr_scheduler_path = os.path.join(CHECKPOINTS_DIR, f"LAST_epoch_{epoch}_SCHEDULER.pth")
        config.next_epoch = epoch + 1
        config.resume = True
        config_save_path = os.path.join(CHECKPOINTS_DIR, "LAST_CONFIG.pkl")

        torch.save(model.state_dict(), config.pretrained_path)
        torch.save(optimizer.state_dict(), config.optimizer_path)
        torch.save(scheduler.state_dict(), config.lr_scheduler_path)
        pickle_data(config_save_path, config)

        if old_pretrained_path is not None:
            os.remove(old_pretrained_path)
        if old_optimizer_path is not None:
            os.remove(old_optimizer_path)
        if old_lr_scheduler_path is not None:
            os.remove(old_lr_scheduler_path)

    training_end_time = datetime.now()
    training_time_elapsed = training_end_time - training_start_time

    print("Training complete!")
    print(f"Time Elapsed: {training_time_elapsed}")
    print(min_n_val.__str__())
    # ---------------------------------------
