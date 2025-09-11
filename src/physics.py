from typing import List, Tuple
import numpy as np
import torch


def build_hamiltonian(Q, g, t):
    """
    Build the Holstein model Hamiltonian matrix.

    Constructs H = -g * diag(Q) + H_hopping where H_hopping is the
    nearest-neighbor hopping matrix with amplitude t.

    Args:
        Q (torch.Tensor): Displacement field of shape [..., L].
        g (float): Electron-phonon coupling strength.
        t (float): Hopping amplitude.

    Returns:
        torch.Tensor: Hamiltonian matrix of shape [..., L, L].
    """

    if torch.is_complex(Q):
        raise Exception(f"build_hamiltonian: Q cannot be complex! Its dtype was: {Q.dtype}")

    dtype = Q.dtype
    L = Q.shape[-1]
    device = Q.device

    # Generate t (hopping amplitude) Matrix H_t
    H_t = np.identity(L) * -t
    H_t_up = np.roll(H_t, shift=1, axis=0)
    H_t_down = np.roll(H_t, shift=-1, axis=0)
    H_t = H_t_up + H_t_down
    H_t = torch.tensor(H_t, dtype=dtype, device=device)

    H = -g * torch.diag_embed(Q) + H_t  # new shape [*, L, L]
    return H


def calc_drho(rho, Q, G, t):
    """
    Calculate time derivative of density matrix using von Neumann equation.

    Computes drho/dt = -i[H, rho] where H is the Hamiltonian.

    Args:
        rho (torch.Tensor): Complex density matrix of shape [..., L, L].
        Q (torch.Tensor): Real displacement field of shape [..., L].
        G (float): Electron-phonon coupling strength.
        t (float): Hopping amplitude.

    Returns:
        torch.Tensor: Complex time derivative of shape [..., L, L].
    """

    H = build_hamiltonian(Q, G, t).to(rho.dtype)

    if H.shape != rho.shape:
        raise ValueError(
            f"calc_drho(): H and rho do not have the same shape! H.shape: {H.shape}\trho.shape: {rho.shape}"
        )

    return -1j * (torch.matmul(H, rho) - torch.matmul(rho, H))


def calc_dQ(P, K0):
    """
    Calculate time derivative of displacement field.

    Simple kinematic relation: dQ/dt = K0 * P.

    Args:
        P (torch.Tensor): Momentum field of shape [..., L].
        K0 (float): Kinematic coefficient.

    Returns:
        torch.Tensor: Real time derivative of shape [..., L].
    """

    return K0 * P


def calc_dP(rho, Q, K0, G1, filling=0.5):
    """
    Calculate time derivative of momentum field.

    Includes electron-phonon coupling and harmonic restoring force.

    Args:
        rho (torch.Tensor): Complex density matrix of shape [..., L, L].
        Q (torch.Tensor): Displacement field of shape [..., L].
        K0 (float): Spring constant coefficient.
        G1 (float): Electron-phonon coupling coefficient.
        filling (float): Average electron filling. Defaults to 0.5 (half-filling).

    Returns:
        torch.Tensor: Real time derivative of shape [..., L].
    """

    rho_diag = torch.diagonal(rho, dim1=-2, dim2=-1).real

    if rho_diag.shape != Q.shape:
        raise ValueError(
            f"calc_dP(): rho_diag and Q do not have the same shape! rho_diag.shape: {rho_diag.shape}\tQ.shape: {Q.shape}"
        )

    return G1 * (rho_diag - filling) - K0 * Q


def calc_derivatives(rho, Q, P, K0, G1, G, t=1.0):
    """
    Calculate all time derivatives for the Holstein model dynamics.

    Computes drho/dt, dQ/dt, and dP/dt according to the Holstein model equations.

    Args:
        rho (torch.Tensor): Complex density matrix of shape [..., L, L].
        Q (torch.Tensor): Displacement field of shape [..., L].
        P (torch.Tensor): Momentum field of shape [..., L].
        K0 (float): Kinematic/spring coefficient.
        G1 (float): Electron-phonon coupling for momentum.
        G (float): Electron-phonon coupling for Hamiltonian.
        t (float): Hopping amplitude. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor]: (drho, dQ, dP) with same shapes as inputs.
    """

    drho = calc_drho(rho, Q, G, t)
    dQ = calc_dQ(P, K0)
    dP = calc_dP(rho, Q, K0, G1)

    if rho.shape != drho.shape:
        raise ValueError(
            f"calc_derivatives(): rho.shape != drho.shape: rho.shape: {rho.shape}\tdrho.shape: {drho.shape}"
        )
    if Q.shape != dQ.shape:
        raise ValueError(f"calc_derivatives(): Q.shape != dQ.shape: Q.shape: {Q.shape}\tdQ.shape: {dQ.shape}")
    if P.shape != dP.shape:
        raise ValueError(f"calc_derivatives(): P.shape != dP.shape: P.shape: {P.shape}\tdP.shape: {dP.shape}")

    return drho, dQ, dP


def torch_calc_derivatives_to_numpy(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray], K0: float, G1: float, G: float
) -> List[np.ndarray]:
    """
    Calculate derivatives using torch tensors and return as numpy arrays.

    Args:
        data (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing (rho, Q, P) as numpy arrays
        K0 (float): Physical parameter for derivative calculation
        G1 (float): Physical parameter for derivative calculation
        G (float): Physical parameter for derivative calculation

    Returns:
        List[np.ndarray]: List of numpy arrays containing the derivatives [drho, dQ, dP]
    """
    rho, Q, P = data
    derivatives = calc_derivatives(torch.tensor(rho), torch.tensor(Q), torch.tensor(P), K0, G1, G)
    return [comp.numpy() for comp in derivatives]


def get_max_abs(data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float]:
    """
    Get maximum absolute values for rho (real and imaginary parts), Q, and P.

    Args:
        data: Tuple containing (rho, Q, P) as numpy arrays

    Returns:
        Tuple of maximum absolute values (rho_max_abs, Q_max_abs, P_max_abs)
    """
    rho, Q, P = data
    rho_max_abs = max(np.max(np.abs(rho.real)), np.max(np.abs(rho.imag))).item()
    Q_max_abs = np.max(np.abs(Q)).item()
    P_max_abs = np.max(np.abs(P)).item()

    return rho_max_abs, Q_max_abs, P_max_abs


def get_deltas(data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> List[np.ndarray]:
    """
    Calculate time differences (deltas) for each component in the data.

    Args:
        data: Tuple containing (rho, Q, P) as numpy arrays

    Returns:
        List of numpy arrays containing the deltas [delta_rho, delta_Q, delta_P]
    """
    delta_data = [comp[:, 1:] - comp[:, :-1] for comp in data]
    return delta_data


def get_alternating_sign_vec(L: int, backend="torch", device=None, dtype=None) -> torch.Tensor | np.ndarray:
    """
    Generate an alternating sign vector for CDW order calculations.
    Compatible with both PyTorch and NumPy.

    Args:
        L (int): Length of the vector.
        backend (str): Either 'torch' or 'numpy' to specify the backend.
        device: PyTorch device (ignored for NumPy).
        dtype: Data type for the tensor/array.

    Returns:
        torch.Tensor or np.ndarray: Vector of alternating signs [+1, -1, +1, ...] of length L.
    """
    values = [(-1) ** i for i in range(1, L + 1)]

    if backend == "torch":
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(values, device=device, dtype=dtype)
    elif backend == "numpy":
        if dtype is None:
            dtype = np.float32
        return np.array(values, dtype=dtype)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'numpy'.")


def calc_charge_order(x) -> torch.Tensor | np.ndarray:
    """
    Calculate charge order parameter for density matrices (rho).
    Compatible with both PyTorch tensors and NumPy arrays.

    Takes diagonal elements and computes alternating sum divided by L.

    Args:
        x (torch.Tensor or np.ndarray): Density matrices of shape (..., L, L).

    Returns:
        torch.Tensor or np.ndarray: Charge order values of shape (...).
    """

    L = x.shape[-1]

    is_torch = isinstance(x, torch.Tensor)

    if is_torch:
        diag = torch.diagonal(x, dim1=-2, dim2=-1).real  # shape (..., L)
        alternating_vec = get_alternating_sign_vec(L, backend="torch", device=diag.device, dtype=diag.dtype)
        alternating_sign_diag = diag * alternating_vec  # broadcasting
        sum_ = torch.sum(alternating_sign_diag, dim=-1)
    else:
        # NumPy case
        diag = np.diagonal(x, axis1=-2, axis2=-1).real  # shape (..., L)
        alternating_vec = get_alternating_sign_vec(L, backend="numpy", dtype=diag.dtype)
        alternating_sign_diag = diag * alternating_vec  # broadcasting
        sum_ = np.sum(alternating_sign_diag, axis=-1)

    mean = sum_ / L

    return mean


def calc_lattice_order(x) -> torch.Tensor | np.ndarray:
    """
    Calculate lattice order parameter for displacement (Q) vectors.
    Compatible with both PyTorch tensors and NumPy arrays.

    Computes alternating sum of vector elements divided by L.

    Args:
        x (torch.Tensor or np.ndarray): Displacement vectors of shape (..., L).

    Returns:
        torch.Tensor or np.ndarray: lattice values of shape (...).
    """

    L = x.shape[-1]

    is_torch = isinstance(x, torch.Tensor)

    if is_torch:
        alternating_vec = get_alternating_sign_vec(L, backend="torch", device=x.device, dtype=x.dtype)
        alternating_sign_arr = x * alternating_vec
        sum_ = torch.sum(alternating_sign_arr, dim=-1)
    else:
        # NumPy case
        alternating_vec = get_alternating_sign_vec(L, backend="numpy", dtype=x.dtype)
        alternating_sign_arr = x * alternating_vec
        sum_ = np.sum(alternating_sign_arr, axis=-1)

    mean = sum_ / L

    return mean
