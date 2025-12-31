# Standard library imports
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

# Third-party imports
import torch
from torch import nn

# Local imports
from src.data_utils import unpickle_data
from src.utils import get_project_root


@dataclass
class Config:
    """
    Configuration class for training Holstein PARC models.

    This dataclass contains all hyperparameters and settings needed for training,
    including data configuration, model architecture, training parameters, and
    optimization settings.

    Attributes are grouped by functionality:
    - Run identification (name, sub_name)
    - Data configuration (data_names, n_sims, etc.)
    - Validation settings (val_batch_size, val_n_step_prediction, etc.)
    - Training parameters (epochs, batch_size, lr, etc.)
    - Model architecture (channels, n_blocks, etc.)
    - Utility settings (suppress_tqdm, save_checkpoint_period, etc.)
    - Resume training settings (resume, pretrained_path, etc.)
    """

    # Run identification

    name: str = None
    sub_name: str = None

    """ Data variables """
    # A list of valid data names
    # A data name is the name of the directory within the /data/generated directory storing the desired dataset. Aka the 'name' variable within datagen.py
    # If a single name is provided, it will be converted to a list of one element
    data_names: list[str] = None
    # When there are multiple data names in data_name, these can be used as a shorter "nickname" for the datasets when logging individual dataset statistics. If specified, must have same number of elements as data_name
    data_nicknames: list[str] | None = None
    # For legacy compatibility. If set, its value will be wrapped in a list and then set as data_names
    data_name: str | None = None

    # Number of simulations to use from the training data for training. If None uses all the sims in training data
    n_sims: int | None = None

    # Number of steps to use from each simulation (for train, val, and test). If None, uses all steps (or rather, for implementation purposes, uses a number of steps for all simulations equal to the number of steps in the simulation with least steps)
    # If not None, uses the first n_steps_per_sim steps of each simulation
    n_steps_per_sim: int | None = None

    """ Val Data Details """
    # Controls batch size for val and test sets. If None uses batch_size. Should be an implementation detail with no impact on result, but may speed up inference of val and test sets
    val_batch_size: int | Sequence[int] | None = 64
    val_n_step_prediction: int = 8  # Also for test set
    val_predict_multiple_steps: bool = True
    val_n_steps_per_sim: int | None = None

    """ Training details """

    ## Settings related to cirriculum
    cirriculum_name: str = "deep_quench-1a"
    # Scales the input noise values in cirriculum. If None does not scale (None is the same as 1.0)
    input_noise_scale_coeff: float | None = None

    ## Other training details
    # If None it sums cirriculum epoch count
    epochs: int | None = None
    # If not None, uses the virtual epoch loader with specified number of batches per epoch. The virtual epoch loader wraps around the dataloader to create epochs with a certain desired batch size
    # Internally, the virtual epoch loader iterates through the actual dataloader irrespective of the boundaries of the virtual epochs
    # Convenient for comparing datasets with different sizes
    # If none, does not use virtual epoch loader
    virtual_epoch_size: int | None = 8192
    batch_size: int | Sequence[int] = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0  # for gradient clipping
    dropout: float = 0.0
    # If n_step_prediction > 1 and True then train on all steps in-between as well. Eg if n_step_prediction == 3 and True then train on steps 1, 2, and 3.
    # If n_step_prediction == 1 then predict_multiple_steps makes no difference
    predict_multiple_steps: bool = True
    use_derivatives: bool = True
    # Prediction loss component has an implicit coeff of 1.0. To equally weigh the two loss components set this to 1.0 as well
    deriv_loss_coeff: float | None = 1.0
    # A term to exponentially scale later steps in the loss calculation when using multi-step loss. None is mathematically the same as 1.0 but skips computation entirely
    stepwise_loss_scale_coeff: float | None = None
    # Disables noise on the input if it is the initial condition
    disable_initial_state_noise: bool = False

    # Whether to separate rho into the diag and off-diag as two separate loss components
    separate_rho_diag: bool = True

    ## Settings that affect lr scheduler
    warmup_steps: int = 500  # The number of steps to warm up for at the start of each cosine restart
    eta_min_scalar: int = 1e-2  # The minimum learning rate multiplier to be used in the cosine annealing. eta_min value is lr * eta_min_scalar
    lr_scheduler: str = "cosine_annealing_restarts"
    # For use with cosine annealing restart with warmups scheduler
    lr_scheduler_linear_decay_multiplier: float | None = None


    # If specified, the first block of the CNN will get a learning rate scaled by this coeff. The last block will get an unscaled learning.
    # And the blocks in-between will have their LR interpolated between the two
    # If None does not apply
    # Meant for fine-tuning where the earlier layers should get a lower LR
    fine_tuning_layerwise_lr_coeff: int | float | None = None


    """ Model details """
    channels: int = 128
    n_blocks: int = 4
    kernel_size: int = 3
    use_residual_scalar: bool = False
    act_func: nn.Module = nn.Tanh()
    init_method: str = "xavier_uniform"
    zero_initialize_output: bool = True
    dtype: torch.dtype = torch.float32
    use_layernorm: bool = True

    # To not use data scalars, use None
    # Otherwise, data scalars may be
    #   * A valid data scalars dict
    #   * The string "same_as_data", in which case the data scalars will be automatically obtained from the data directory
    #       * If mulitple datasets are used, their data scalars will be combined by taking the greatest value for each coefficient
    #   * A path to a directory containing a "data_scalars.pkl"
    #   * A path to a pickled file containing a data scalars dict
    data_scalars: dict[str, float | int] | str | os.PathLike | None = "same_as_data"

    # Simple CNN directly predicts deltas from state values, without PARC integrator/differentiator structure
    # use_derivatives must be false if simple_cnn is True
    simple_cnn: bool = False

    """ Convenience features """
    # If true tqdm will be disabled. Useful for SLURM scripts so that SLURM output files will not be egregiously large
    suppress_tqdm: bool = False
    # Save checkpoint every certain number of epochs. This is separate from best epoch saving. if None will only do best epoch saving
    save_checkpoint_period: int | None = None
    n_workers: int = 0  # n_workers for dataloader
    notes: str | None = ""
    _config_name = None  # If config retrieved with src.get_settings.get_config, it will change this to indicate exactly how the particular config object was made
    # If True, will override all model kwargs except dropout via the one found from the same directory as a pretrained path
    # Consumed by the _validate method and then set to False again after a single use
    set_model_kwargs_from_pretrained_once: bool = False

    """ Training Checkpointing Resuming """
    resume: bool = False  # whether we are resuming from training
    pretrained_path: str | None = None  # If None, we initialize from random
    optimizer_path: str | None = None  # If None, new optimizer is initialized
    lr_scheduler_path: str | None = None  # If None, lr scheduler starts from beginning
    next_epoch: int = 1  # Defaults to 1 since epoch count starts at 1
    START_TIME: str | None = None

    def get_model_kwargs(self):
        """
        Extract model-specific keyword arguments from the config.

        Returns:
            dict: Dictionary containing all parameters needed to instantiate a HolsteinPARC model.
        """
        return {
            "channels": self.channels,
            "n_blocks": self.n_blocks,
            "dropout_p": self.dropout,
            "kernel_size": self.kernel_size,
            "use_residual_scalar": self.use_residual_scalar,
            "act_func": self.act_func,
            "init_method": self.init_method,
            "zero_initialize_output": self.zero_initialize_output,
            "dtype": self.dtype,
            "use_layernorm": self.use_layernorm,
            "data_scalars": self.data_scalars,
            "simple_cnn": self.simple_cnn,
        }

    def __post_init__(self):
        self._validate()

    def _validate(self):
        """
        Validate configuration parameters and apply necessary transformations.

        Performs checks for:
        - Resume/START_TIME consistency
        - Required paths for resuming
        - Data name formatting
        - Cirriculum and model parameter validation
        - Fine-tuning settings

        Raises:
            ValueError: If configuration parameters are invalid or inconsistent.
        """
        if self.resume ^ (self.START_TIME is not None):
            raise ValueError(
                "Either both resume and START_TIME should be set, or neither should be set. "
                "If you are resuming training, set resume=True and START_TIME to the time of the previous run."
                "Otherwise, do not set START_TIME as START_TIME is used to create a unique name for the run"
            )
        if self.resume and (
            self.pretrained_path is None or self.optimizer_path is None or self.lr_scheduler_path is None
        ):
            raise ValueError(
                "If resuming training, pretrained_path, optimizer_path, and lr_scheduler_path must be set."
            )

        # Handle legacy data_name parameter by converting to data_names
        if self.data_name is not None:
            if self.data_names is not None:
                raise ValueError(
                    f"Only one of data_name or data_names may be specified. But got data_name: {self.data_name}, data_names: {self.data_names}"
                )
            if not isinstance(self.data_name, (list, tuple)):
                self.data_names = [self.data_name]
            else:
                self.data_names = self.data_name
            self.data_name = None

        # Ensure data_names and data_nicknames are lists
        if isinstance(self.data_names, str):
            self.data_names = [self.data_names]

        if isinstance(self.data_nicknames, str):
            self.data_nicknames = [self.data_nicknames]

        if self.data_names is not None and self.data_nicknames is not None:
            if len(self.data_names) != len(self.data_nicknames):
                raise ValueError(
                    f"Since logging_data_nicknames was specified, its length (number of elements) must equal the length of data_names! But data_names had {len(self.data_names)} elements while logging_data_nicknames had {len(self.data_nicknames)} elements!"
                    f"data_names: {self.data_names}"
                    f"logging_data_nicknames: {self.data_nicknames}"
                )

        # Load data_scalars
        # The intention is that once this step is past, data_scalars will always be either a dict or None
        if not isinstance(self.data_scalars, (dict, type(None))):
            data_scalars_path = Path(self.data_scalars)

            if self.data_names and self.data_scalars == "same_as_data":
                generated_data_dir = get_project_root() / "data" / "generated"
                data_scalar_dicts = [
                    unpickle_data(generated_data_dir / data_name / "pickled" / "data_scalars.pkl")
                    for data_name in self.data_names
                ]
                # If there are mulitple datasets, create a new data scalars dict by taking the max of each scalar
                self.data_scalars = {
                    k: max(data_scalar_dict[k] for data_scalar_dict in data_scalar_dicts) for k in data_scalar_dicts[0]
                }
            elif data_scalars_path.is_file():
                self.data_scalars = unpickle_data(data_scalars_path)
            elif data_scalars_path.is_dir():
                if (data_scalars_path / "data_scalars.pkl").is_file():
                    self.data_scalars = unpickle_data(data_scalars_path / "data_scalars.pkl")
                else:
                    raise ValueError(
                        f"data_scalars_path was a directory but did not contain a data_scalars.pkl file! Expected a file at: {data_scalars_path / 'data_scalars.pkl'}"
                    )
            else:
                raise ValueError(
                    f"data_scalars was not a data scalars dict, None, 'same_as_data', or a path to a file or directory! But it was {self.data_scalars}"
                )

        if isinstance(self.batch_size, Sequence) and len(self.batch_size) != len(self.data_names):
            raise ValueError(
                f"batch_size was a sequence but did not have the same length as data_names. batch_size: {self.batch_size}, data_names: {self.data_names}"
            )
        if isinstance(self.val_batch_size, Sequence) and len(self.val_batch_size) != len(self.data_names):
            raise ValueError(
                f"val_batch_size was a sequence but did not have the same length as data_names. val_batch_size: {self.val_batch_size}, data_names: {self.data_names}"
            )

        if self.simple_cnn and self.use_derivatives:
            raise ValueError("simple_cnn and use_derivatives cannot both be True!")

        if self.use_derivatives and self.deriv_loss_coeff is None:
            raise ValueError("If use_derivatives is True then deriv_loss_coeff cannot be None!")

        if self.fine_tuning_layerwise_lr_coeff is not None and self.fine_tuning_layerwise_lr_coeff > 1:
            raise ValueError(
                f"fine_tuning_layerwise_lr_coeff should not be greater than 1, since it is supposed to scale DOWN the LR. But it was {self.fine_tuning_layerwise_lr_coeff}"
            )
        if self.fine_tuning_layerwise_lr_coeff is not None and self.fine_tuning_layerwise_lr_coeff < 0:
            raise ValueError(
                f"fine_tuning_layerwise_lr_coeff cannot be negative. But it was {self.fine_tuning_layerwise_lr_coeff}"
            )

        if self.set_model_kwargs_from_pretrained_once:
            if self.pretrained_path is None:
                raise ValueError("set_model_kwargs_from_pretrained was True but pretrained_path was None!")

            model_kwargs_path = os.path.join(os.path.dirname(self.pretrained_path), "model_kwargs.pkl")
            if not os.path.exists(model_kwargs_path):
                raise ValueError(
                    f"set_model_kwargs_from_pretrained was True but pretrained_path directory did not contain a model_kwargs.pkl! Expected a file at: {model_kwargs_path}"
                )

            model_kwargs = unpickle_data(model_kwargs_path)

            # Translation layer for keys which have different names in model vs config
            if "dropout_p" in model_kwargs:
                model_kwargs_dropout = model_kwargs.pop("dropout_p")
                if model_kwargs_dropout != self.dropout:
                    print(
                        f"set_model_kwargs_from_pretrained: loaded model_kwargs had a dropout p of {model_kwargs_dropout}, which differs from the set dropout of {self.dropout}. Although dropout is part of model_kwargs, it is not considered a part of the model, so it will not override the existing dropout value."
                    )

            if len(extra_keys := set(model_kwargs.keys()) - set(vars(self).keys())):
                raise ValueError(
                    f"set_model_kwargs_from_pretrained was True but the loaded model_kwargs contained invalid keys! These were: {extra_keys}"
                )

            diff_kwargs = {k: v for k, v in model_kwargs.items() if getattr(self, k) != v}

            if len(diff_kwargs) == 0:
                print("set_model_kwargs_from_pretrained: pretrained model kwargs had no differences. No changes!")
            else:
                print(
                    f"\n\nset_model_kwargs_from_pretrained: pretrained model kwargs had differences in {len(diff_kwargs)} keys! Making the following changes:"
                )
                for k, v in diff_kwargs.items():
                    print(f"\t{k}: {getattr(self, k)} -> {v}")
                print("\n\n")

            self.set_model_kwargs_from_pretrained_once = False

            self.update_attrs_with_dict(kwargs=diff_kwargs, must_exist=True)

    def update_attrs_with_dict(
        self, kwargs: dict, must_exist=False, combine_notes: bool = False, combine_subname: bool = False
    ):
        """
        Update configuration attributes from a dictionary.

        Args:
            kwargs (dict): Dictionary of attributes to update.
            must_exist (bool): If True, raise error if attribute doesn't exist.
            combine_notes (bool): If True, append to existing notes instead of replacing.
            combine_subname (bool): If True, append to existing sub_name with hyphen separator.

        Raises:
            ValueError: If must_exist is True and attribute doesn't exist.
        """
        for k, v in kwargs.items():
            if must_exist and not hasattr(self, k):
                raise ValueError(f"Attribute {k} does not exist.")

            if combine_notes and k == "notes":
                self.notes += "\n\n" + v
            elif combine_subname and k == "sub_name":
                if v is None:
                    pass
                elif self.sub_name is None:
                    self.sub_name = v
                else:
                    self.sub_name += "-" + v
            else:
                setattr(self, k, v)

        self._validate()
