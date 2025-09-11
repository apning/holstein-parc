# Standard library imports
import pickle
from typing import Tuple
import warnings

# Third-party imports
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Local imports
from src.physics import calc_derivatives, get_deltas, get_max_abs, torch_calc_derivatives_to_numpy


def pickle_data(save_path, data):
    """
    Pickle data to a file.

    Args:
        save_path (str): Path where to save the pickled data. Should end with .pkl.
        data: Any Python object to be pickled.
    """
    with open(save_path, "wb") as file:
        pickle.dump(data, file)


def unpickle_data(save_path):
    """
    Load pickled data from a file.

    Args:
        save_path (str): Path to the pickled file.

    Returns:
        The unpickled Python object.
    """
    with open(save_path, "rb") as file:
        data = pickle.load(file)
    return data


class HolsteinDataset(Dataset):
    """
    PyTorch Dataset for Holstein model trajectory data.

    Handles loading and preprocessing of simulation data including optional
    multi-step predictions, derivative calculations, and input noise.

    Args:
        data (tuple[np.ndarray]): Tuple of (rho, Q, P) arrays with shapes:
            - rho: [n_sims, n_steps+1, L, L] (complex64)
            - Q, P: [n_sims, n_steps+1, L] (float32)
        label_step_count (int): Number of steps ahead for prediction labels. Corresponds to n_step_prediction in train.py
        max_label_step_count (int | None): Specify in case the label_step_count may be externally changed later. If so, set to the largest label_step_count value that will be used.
        the largest label_step_count value that will be used
        multi_step_labels (bool): If True, include all intermediate steps in labels. If not enabled,
        the shape of a return rho label (before batching) would be [L,L]. If enabled, it would instead be [label_step_count,
        L, L]. Corresponds to return_multiple_steps in model forward method
        n_sims (int | None): Number of simulations to use. None uses all.
        n_steps (int | None): Number of time steps per simulation. None uses all.
        return_derivatives (bool): If True, return derivatives in labels.
        deriv_data (tuple[np.ndarray] | None): Mid-point data for derivative calculation. If None then the
        data will be used for beginning-point derivative calculation. The shape of deriv_data should be the same as data,
        except there should be one less step
        K0, G, G1 (float | None): Physical parameters for derivative calculation. If return_derivatives they must be specified
        input_gaussian_noise_std (float): Standard deviation of input noise.
        disable_initial_state_noise (bool): If True, no noise on initial conditions.
    """

    def __init__(
        self,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        label_step_count: int = 1,
        max_label_step_count: int | None = None,
        multi_step_labels: bool = False,
        n_sims: int = None,
        n_steps: int = None,
        return_derivatives: bool = False,
        deriv_data: tuple[np.ndarray] = None,
        K0: float | None = None,
        G: float | None = None,
        G1: float | None = None,
        input_gaussian_noise_std: float = 0.0,
        disable_initial_state_noise: bool = True,
    ):
        super().__init__()

        "Data checks"
        # Verify size of data tuple is 3
        if len(data) != 3:
            raise ValueError(f"HolsteinDataset: Length of data tuple not 3! It was: {len(data)}")

        rho, Q, P = data
        # Verify dimensionality of each component
        _n_sims, _, L = np.shape(Q)
        _n_steps = np.shape(Q)[1] - 1
        if np.shape(rho) != (_n_sims, _n_steps + 1, L, L):
            raise ValueError(
                f"HolsteinDataset: rho shape was not [n_sims, _n_steps+1, L, L], which is expected to be {_n_sims, _n_steps + 1, L, L}. It was: {np.shape(rho)}"
            )
        if np.shape(Q) != (_n_sims, _n_steps + 1, L):
            raise ValueError(
                f"HolsteinDataset: Q shape was not [n_sims, _n_steps+1, L], which is expected to be {_n_sims, _n_steps + 1, L}. It was: {np.shape(Q)}"
            )
        if np.shape(P) != (_n_sims, _n_steps + 1, L):
            raise ValueError(
                f"HolsteinDataset: P shape was not [n_sims, _n_steps+1, L], which is expected to be {_n_sims, _n_steps + 1, L}. It was: {np.shape(P)}"
            )
        # Check dimensions of deriv_data as well if it will be used
        if return_derivatives and deriv_data is not None:
            mid_rho, mid_Q, mid_P = deriv_data
            if np.shape(mid_rho) != (_n_sims, _n_steps, L, L):
                raise ValueError(
                    f"HolsteinDataset: deriv_data rho shape was not [n_sims, _n_steps, L, L], which is expected to be {_n_sims, _n_steps, L, L}. It was: {np.shape(mid_rho)}"
                )
            if np.shape(mid_Q) != (_n_sims, _n_steps, L):
                raise ValueError(
                    f"HolsteinDataset: deriv_data Q shape was not [n_sims, _n_steps, L], which is expected to be {_n_sims, _n_steps, L}. It was: {np.shape(mid_Q)}"
                )
            if np.shape(mid_P) != (_n_sims, _n_steps, L):
                raise ValueError(
                    f"HolsteinDataset: deriv_data P shape was not [n_sims, _n_steps, L], which is expected to be {_n_sims, _n_steps, L}. It was: {np.shape(mid_P)}"
                )

        "Argument checks"

        if max_label_step_count is None:
            max_label_step_count = label_step_count

        if label_step_count < 1:
            raise ValueError(f"HolsteinDataset: label_step_count must be at least one. Received: {label_step_count}")
        if max_label_step_count < 1:
            raise ValueError(
                f"HolsteinDataset: max_label_step_count must be at least one (or None). Received: {max_label_step_count}"
            )
        if max_label_step_count < label_step_count:
            raise ValueError(
                f"HolsteinDataset: max_label_step_count must be at least label_step_count (or None). Received: {max_label_step_count} < {label_step_count}"
            )
        if return_derivatives and any(coeff is None for coeff in (K0, G1, G)):
            raise ValueError("HolsteinDataset: return_derivatives was True but at least one of K0, G1, G was None")

        # If n_sims was specified, make sure it was not greater than the total number of simulations.
        if n_sims is not None:
            if n_sims > _n_sims:
                raise ValueError(
                    f"HolsteinDataset: n_sims was larger than number of simulations in data! n_sims: {n_sims}\tlen(data): {_n_sims}"
                )
        else:
            n_sims = _n_sims
        # If n_steps was specified, make sure it was not greater than the total number of steps.
        if n_steps is not None:
            if n_steps > _n_steps:
                raise ValueError(
                    f"HolsteinDataset: n_steps was larger than number of steps in data! n_steps: {n_steps}\tlen(data): {_n_steps}"
                )
        else:
            n_steps = _n_steps

        if not return_derivatives and deriv_data is not None:
            warnings.warn(
                "HolsteinDataset: If return_derivatives is False, there is no need to provide deriv_data",
                category=UserWarning,
            )

        # The actual number of data points we can use from each simulation
        # For example, if a simulation has n_steps, then it has n_step+1 data data_points. However, we can only get at most n_steps data points from this data given we do next-step-prediction
        # If our prediction target is greater than next-step, then we get even fewer data points from this data
        data_points_per_sim = n_steps - (max_label_step_count - 1)
        if data_points_per_sim <= 0:
            raise ValueError(
                f"HolsteinDataset: max_label_step_count is larger than available timesteps (max_label_step_count={max_label_step_count}, n_steps={n_steps})"
            )

        " Process the simulations according to arugments "

        # Take the subset of data with size n_sims
        if n_sims < _n_sims:
            data = [component[:n_sims] for component in data]
        # Take the subset of data with steps n_steps
        if n_steps < _n_steps:
            data = [component[:, : n_steps + 1] for component in data]

        if return_derivatives and deriv_data is not None:
            # Take the subset of deriv_data with size n_sims
            if n_sims < _n_sims:
                deriv_data = [component[:n_sims] for component in deriv_data]
            # Take the subset of deriv_data so it contains only what it needs
            if n_steps < _n_steps:
                deriv_data = [component[:, :n_steps] for component in deriv_data]

        # If return_derivatives is true but we don't have deriv_data, just use the regular data to perform a start-point derivative
        if return_derivatives and deriv_data is None:
            deriv_data = data

        self.data = data
        self.deriv_data = deriv_data
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.label_step_count = label_step_count
        self.max_label_step_count = max_label_step_count
        self.multi_step_labels = multi_step_labels
        self.return_derivatives = return_derivatives
        self.K0, self.G1, self.G = K0, G1, G
        self.input_gaussian_noise_std = input_gaussian_noise_std
        self.data_points_per_sim = data_points_per_sim
        self.disable_initial_state_noise = disable_initial_state_noise

    def __len__(self):
        return self.n_sims * self.data_points_per_sim

    def __getitem__(self, idx):
        # Extract simulation index and time step from the flattened index

        simulation_idx = idx // self.data_points_per_sim
        step_idx = idx % self.data_points_per_sim

        all_density_mat, all_displacement, all_momentum = [component[simulation_idx] for component in self.data]
        if self.return_derivatives:
            deriv_all_density_mat, deriv_all_displacement, deriv_all_momentum = [
                component[simulation_idx] for component in self.deriv_data
            ]

        input_ = (
            torch.tensor(all_density_mat[step_idx], dtype=torch.complex64),
            torch.tensor(all_displacement[step_idx], dtype=torch.float32),
            torch.tensor(all_momentum[step_idx], dtype=torch.float32),
        )

        if not self.multi_step_labels:
            rho_label = torch.tensor(all_density_mat[step_idx + self.label_step_count], dtype=torch.complex64)
            Q_label = torch.tensor(all_displacement[step_idx + self.label_step_count], dtype=torch.float32)
            P_label = torch.tensor(all_momentum[step_idx + self.label_step_count], dtype=torch.float32)
        else:
            rho_label = torch.tensor(
                all_density_mat[step_idx + 1 : step_idx + self.label_step_count + 1], dtype=torch.complex64
            )
            Q_label = torch.tensor(
                all_displacement[step_idx + 1 : step_idx + self.label_step_count + 1], dtype=torch.float32
            )
            P_label = torch.tensor(
                all_momentum[step_idx + 1 : step_idx + self.label_step_count + 1], dtype=torch.float32
            )

        if self.return_derivatives:
            if not self.multi_step_labels:
                drho_prelabel = torch.tensor(
                    deriv_all_density_mat[step_idx + self.label_step_count - 1], dtype=torch.complex64
                )
                dQ_prelabel = torch.tensor(
                    deriv_all_displacement[step_idx + self.label_step_count - 1], dtype=torch.float32
                )
                dP_prelabel = torch.tensor(
                    deriv_all_momentum[step_idx + self.label_step_count - 1], dtype=torch.float32
                )
            else:
                drho_prelabel = torch.tensor(
                    deriv_all_density_mat[step_idx : step_idx + self.label_step_count], dtype=torch.complex64
                )
                dQ_prelabel = torch.tensor(
                    deriv_all_displacement[step_idx : step_idx + self.label_step_count], dtype=torch.float32
                )
                dP_prelabel = torch.tensor(
                    deriv_all_momentum[step_idx : step_idx + self.label_step_count], dtype=torch.float32
                )

            drho_label, dQ_label, dP_label = calc_derivatives(
                drho_prelabel, dQ_prelabel, dP_prelabel, self.K0, self.G1, self.G
            )

            label = rho_label, Q_label, P_label, drho_label, dQ_label, dP_label
        else:
            label = rho_label, Q_label, P_label

        # Add random gaussian noise to each tensor input IF the input is not the very first data point in the trajectory
        # This is because the initial condition can be extremely sensitive to tiny pertubations
        if self.input_gaussian_noise_std and not (step_idx == 0 and self.disable_initial_state_noise):
            input_ = tuple(tnsr + torch.randn_like(tnsr) * self.input_gaussian_noise_std for tnsr in input_)

        return input_, label


class Virtual_Epoch_loader:
    """
    Wrapper for DataLoader that redefines epoch length.

    Creates "virtual epochs" with a fixed number of batches, regardless of the actual dataset size. Useful for consistent epoch lengths across datasets of different sizes. If the dataloader has more batches per epoch than in a "virtual epoch", than one actual epoch will span multiple "virtual epochs". If the data loader has less, then one "virtual epoch" may have multiple whole or partial actual epochs.

    When a virtual epoch ends and a new one begins, the new one begins at the batch where the previous virtual epoch left off.

    When the underlying dataloader is exhausted, it automatically restarts from
    the beginning, ensuring continuous data flow.

    Args:
        dataloader (DataLoader): The underlying PyTorch DataLoader.
        batches_per_epoch (int): Number of batches in each virtual epoch.
    """

    def __init__(self, dataloader: DataLoader, batches_per_epoch: int):
        self.dataloader = dataloader
        self.batches_per_epoch = batches_per_epoch

        self.iterable = iter(self.dataloader)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        self.batches_left = self.batches_per_epoch
        return self

    def __next__(self):
        if self.batches_left > 0:
            self.batches_left -= 1
            try:
                return next(self.iterable)
            except StopIteration:
                self.iterable = iter(self.dataloader)
                return next(self.iterable)
        else:
            raise StopIteration


def get_component_scalars_dict(data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """
    Calculate max absolute values for the original data components.

    Args:
        data: Tuple containing (rho, Q, P) as numpy arrays

    Returns:
        Dictionary containing scalar statistics with keys: 'rho', 'Q', 'P'
    """
    rho_max, Q_max, P_max = get_max_abs(data)
    return {"rho": rho_max, "Q": Q_max, "P": P_max}


def get_derivative_scalars_dict(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray], K0: float, G: float, G1: float
) -> dict:
    """
    Calculate max absolute values for the derivatives of the data components.

    Args:
        data: Tuple containing (rho, Q, P) as numpy arrays
        K0, G, G1 (float): Physical parameters for derivative calculation

    Returns:
        Dictionary containing scalar statistics with keys: 'drho', 'dQ', 'dP'
    """

    derivatives = torch_calc_derivatives_to_numpy(data, K0=K0, G=G, G1=G1)
    drho_max, dQ_max, dP_max = get_max_abs(derivatives)
    return {"drho": drho_max, "dQ": dQ_max, "dP": dP_max}


def get_delta_scalars_dict(data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
    """
    Calculate max absolute values for the time differences (deltas) of the data components.

    Args:
        data: Tuple containing (rho, Q, P) as numpy arrays

    Returns:
        Dictionary containing scalar statistics with keys: 'delta_rho', 'delta_Q', 'delta_P'
    """
    deltas = get_deltas(data)
    delta_rho_max, delta_Q_max, delta_P_max = get_max_abs(deltas)
    return {"delta_rho": delta_rho_max, "delta_Q": delta_Q_max, "delta_P": delta_P_max}


def get_data_scalars_dict(data: Tuple[np.ndarray, np.ndarray, np.ndarray], K0: float, G: float, G1: float) -> dict:
    """
    Calculate various scalar statistics for the data including max absolute values
    for original data, derivatives, and deltas.

    Args:
        data: Tuple containing (rho, Q, P) as numpy arrays
        K0, G, G1 (float): Physical parameters for derivative calculation

    Returns:
        Dictionary containing scalar statistics with keys:
        'rho', 'Q', 'P', 'drho', 'dQ', 'dP', 'delta_rho', 'delta_Q', 'delta_P'
    """
    data_scalars = {}

    # Combine all scalar dictionaries
    data_scalars.update(get_component_scalars_dict(data))
    data_scalars.update(get_derivative_scalars_dict(data, K0=K0, G=G, G1=G1))
    data_scalars.update(get_delta_scalars_dict(data))

    return data_scalars
