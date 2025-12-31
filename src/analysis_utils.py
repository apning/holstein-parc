import io
import math
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageOps
from IPython.display import display
from tqdm.auto import tqdm

from src.data_utils import unpickle_data
from src.utils import select_best_device
from src.physics import calc_charge_order, calc_lattice_order


def get_sim(data: tuple[np.ndarray], sim_idx: int, sim_start: int = None, sim_end: int = None) -> tuple[np.ndarray]:
    """
    Extract a single simulation from the data with optional time slicing.

    Args:
        data (tuple[np.ndarray]): Tuple of arrays (rho, Q, P) containing simulation data.
        sim_idx (int): Index of the simulation to extract.
        sim_start (int, optional): Starting time step. Defaults to None (start from beginning).
        sim_end (int, optional): Ending time step. Defaults to None (end at last step).

    Returns:
        tuple[np.ndarray]: Extracted simulation data (rho, Q, P) for the specified simulation.
    """
    sim = [comp[sim_idx] for comp in data]
    sim = tuple(comp[sim_start:sim_end] for comp in sim)

    return sim


@dataclass
class PLTArgs:
    ylim: tuple[int, int] | None = None
    log_x_axis: bool = False
    x_ticks: ArrayLike | None = None
    y_ticks: ArrayLike | None = None
    figsize: tuple[int, int] | None = None
    dpi: int | None = None
    fontsize: float | str | None = None
    labelsize: float | str | None = None
    disable_text: bool = False
    linewidth: float | None = None
    remove_bottom_tick_labels: bool = False
    return_image: bool = False
    title_suffix: str | None = None
    disable_show: bool = False


def plot_line(
    y_values,
    title=None,
    xlabel=None,
    ylabel=None,
    ylim: tuple[int, int] | None = None,
    log_x_axis: bool = False,
    x_ticks: ArrayLike | None = None,
    y_ticks: ArrayLike | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    fontsize: float | str | None = None,
    labelsize: float | str | None = None,
    disable_text: bool = False,
    linewidth: float | None = None,
    remove_bottom_tick_labels: bool = False,
    return_image: bool = False,
    title_suffix: str | None = None,
    disable_show: bool = False,
):
    """
    Plot a line graph with customizable appearance.

    Args:
        y_values (list): List of y values to plot.
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        ylim (tuple[int, int], optional): Y-axis limits.
        log_x_axis (bool): If True, use logarithmic scale for x-axis.
        x_ticks (list of tuples, optional): Custom x-axis ticks as [(value, label), ...].
        y_ticks (list of tuples, optional): Custom y-axis ticks as [(value, label), ...].
        figsize (tuple[int, int], optional): Figure size. Defaults to (10, 6).
        dpi (int, optional): Figure DPI.
        fontsize (float | str, optional): Font size for labels. Defaults to 20.
        labelsize (float | str, optional): Tick label size. Defaults to 20.
        disable_text (bool): If True, disable all text labels.
        linewidth (float, optional): Line width. Defaults to 2.
        remove_bottom_tick_labels (bool): If True, remove x-axis tick labels.
        return_image (bool): If True, return PIL Image instead of displaying.
        title_suffix (str, optional): Suffix to append to title.
        disable_show (bool): If True, don't display the plot.

    Returns:
        PIL.Image or None: Returns image if return_image is True, otherwise None.
    """

    """ Arugment Processing """

    if figsize is None:
        figsize = (10, 6)
    if fontsize is None:
        fontsize = 20
    if labelsize is None:
        labelsize = 20
    if linewidth is None:
        linewidth = 2

    if title_suffix is not None:
        if title is None:
            title = ""
        title += title_suffix

    """ Begin Plotting Code """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Generate x values as increasing integers
    x_values = range(len(y_values))

    # Plot the line
    ax.plot(x_values, y_values, color="blue", linewidth=linewidth)

    # Set logarithmic scale if enabled
    if log_x_axis:
        ax.set_xscale("log")

    # Set custom tick marks if provided
    if x_ticks is not None:
        ax.set_xticks([value for value, label in x_ticks])
        ax.set_xticklabels([label for value, label in x_ticks], fontsize=labelsize)
    else:
        ax.tick_params(axis="x", which="major", labelsize=labelsize)
    if y_ticks is not None:
        ax.set_yticks([value for value, label in y_ticks])
        ax.set_yticklabels([label for value, label in y_ticks], fontsize=labelsize)
    else:
        ax.tick_params(axis="y", which="major", labelsize=labelsize)

    if remove_bottom_tick_labels:
        ax.tick_params(axis="x", labelbottom=False)

    # Turn off minor ticks
    ax.minorticks_off()

    # Display grid
    ax.grid(True)

    if title is not None and not disable_text:
        ax.set_title(title, fontsize=fontsize)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if xlabel is not None and not disable_text:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None and not disable_text:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if disable_text and title is not None:
        print(title, ":")

    # Show the plot
    if not disable_show:
        plt.show(block=False)

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        image.load()
        buf.close()

    plt.close(fig)

    if return_image:
        return image


def plot_comparison(
    y_label,
    y_pred,
    title=None,
    xlabel=None,
    ylabel=None,
    ylim: tuple[int, int] | None = None,
    log_x_axis: bool = False,
    x_ticks: ArrayLike | None = None,
    y_ticks: ArrayLike | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    fontsize: float | str | None = None,
    labelsize: float | str | None = None,
    disable_text: bool = False,
    linewidth: float | None = None,
    remove_bottom_tick_labels: bool = False,
    return_image: bool = False,
    title_suffix: str | None = None,
    disable_show: bool = False,
):
    """
    Plot a comparison between label and prediction values.

    Displays two lines: blue for labels and red for predictions.

    Args:
        y_label: Label values to plot (can be None).
        y_pred: Prediction values to plot (can be None).
        title (str, optional): Plot title.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        ylim (tuple[int, int], optional): Y-axis limits.
        log_x_axis (bool): If True, use logarithmic scale for x-axis.
        x_ticks (list of tuples, optional): Custom x-axis ticks as [(value, label), ...].
        y_ticks (list of tuples, optional): Custom y-axis ticks as [(value, label), ...].
        figsize (tuple[int, int], optional): Figure size. Defaults to (10, 6).
        dpi (int, optional): Figure DPI.
        fontsize (float | str, optional): Font size for labels. Defaults to 20.
        labelsize (float | str, optional): Tick label size. Defaults to 20.
        disable_text (bool): If True, disable all text labels.
        linewidth (float, optional): Line width. Defaults to 2.
        remove_bottom_tick_labels (bool): If True, remove x-axis tick labels.
        return_image (bool): If True, return PIL Image instead of displaying.
        title_suffix (str, optional): Suffix to append to title.
        disable_show (bool): If True, don't display the plot.

    Returns:
        PIL.Image or None: Returns image if return_image is True, otherwise None.
    """

    """ Arugment Checking/Processing """

    if figsize is None:
        figsize = (10, 6)
    if fontsize is None:
        fontsize = 20
    if labelsize is None:
        labelsize = 20
    if linewidth is None:
        linewidth = 2

    if (y_label is not None and y_pred is not None) and len(y_label) != len(y_pred):
        raise ValueError("y_label and y_pred have different lengths")

    if title_suffix is not None:
        if title is None:
            title = ""
        title += title_suffix

    """ Begin Plotting Code """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Generate x values as increasing integers
    x_values = range(len(y_label)) if y_label is not None else range(len(y_pred))

    # Plot the line
    if y_label is not None:
        ax.plot(x_values, y_label, color="blue", linewidth=linewidth, label="label")
    if y_pred is not None:
        ax.plot(x_values, y_pred, color="red", linewidth=linewidth, label="prediction")

    # Set logarithmic scale if enabled
    if log_x_axis:
        ax.set_xscale("log")

    # Set custom tick marks if provided
    if x_ticks is not None:
        ax.set_xticks([value for value, label in x_ticks])
        ax.set_xticklabels([label for value, label in x_ticks], fontsize=labelsize)
    else:
        ax.tick_params(axis="x", which="major", labelsize=labelsize)
    if y_ticks is not None:
        ax.set_yticks([value for value, label in y_ticks])
        ax.set_yticklabels([label for value, label in y_ticks], fontsize=labelsize)
    else:
        ax.tick_params(axis="y", which="major", labelsize=labelsize)

    if remove_bottom_tick_labels:
        ax.tick_params(axis="x", labelbottom=False)

    # Turn off minor ticks
    ax.minorticks_off()

    # Display grid
    ax.grid(True)
    if not disable_text:
        ax.legend(fontsize=fontsize)

    if title is not None and not disable_text:
        ax.set_title(title, fontsize=fontsize)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if xlabel is not None and not disable_text:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if ylabel is not None and not disable_text:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    if disable_text and title is not None:
        print(title, ":")

    # Show the plot
    if not disable_show:
        plt.show(block=False)

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        image.load()
        buf.close()

    plt.close(fig)

    if return_image:
        return image


def _gen_preds(
    model,
    initial_conditions: Sequence[NDArray, NDArray, NDArray],
    n_steps: int,
    device: torch.device,
    suppress_output: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Core prediction logic for generating multi-step trajectory predictions.

    Args:
        model: Prediction model (already on device and in eval mode).
        initial_conditions (Sequence[NDArray, NDArray, NDArray]): Tuple of (rho, Q, P) label arrays representing the initial conditions. Shape of rho array should be [batch, L, L]. Shape of Q and P should be [batch, L]
        n_steps (int): Number of steps to predict for
        device (torch.device): PyTorch device.
        suppress_output (bool): If True, suppress progress bar output.

    Returns:
        tuple[NDArray, NDArray, NDArray]: Predicted trajectories (rho, Q, P).
    """

    # Prep input data
    preds = [[comp] for comp in initial_conditions]
    inputs = [torch.tensor(comp).to(device) for comp in initial_conditions]

    with torch.no_grad():
        for _ in tqdm(range(n_steps), desc="Steps", leave=False, disable=suppress_output):
            inputs = model(*inputs)
            [pred.append(comp.cpu().numpy()) for pred, comp in zip(preds, inputs)]

    preds = tuple(np.stack(pred, axis=1) for pred in preds)

    return preds


def gen_preds(
    model,
    initial_conditions: Sequence[NDArray, NDArray, NDArray],
    n_steps: int,
    max_batch_size: int | None = None,
    device: None | torch.device = None,
    suppress_output: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate multi-step trajectory predictions from a batch of initial conditions.

    Args:
        model: Prediction model.
        initial_conditions (Sequence[NDArray, NDArray, NDArray]): Tuple of (rho, Q, P) label arrays representing the initial conditions. Shape of rho array should be [batch, L, L]. Shape of Q and P should be [batch, L]
        n_steps (int): Number of steps to predict for
        max_batch_size (int | None): Maximum batch size. If None, process all data at once.
        device (None | torch.device): PyTorch device. If None, one will automatically be chosen.
        suppress_output (bool): If True, suppress progress bar output.

    Returns:
        tuple[NDArray, NDArray, NDArray]: Predicted trajectories (rho, Q, P).
    """

    # Shape checks
    rho, Q, P = initial_conditions
    if rho.ndim != 3:
        raise ValueError(f"rho has invalid shape! Expected 3, got shape {rho.shape}")
    if Q.ndim != 2:
        raise ValueError(f"Q has invalid shape! Expected 2, got shape {Q.shape}")
    if P.ndim != 2:
        raise ValueError(f"P has invalid shape! Expected 2, got shape {P.shape}")

    # Device
    if device is None:
        device = select_best_device("m")
    model.to(device)

    # Set model to eval
    was_training = model.training
    model.eval()

    n_batches = len(rho)
    if max_batch_size is None:
        max_batch_size = n_batches

    # If data is too large for max_batch_size, generate it in mulitple mini-batches
    predicted_batches = []
    for i in tqdm(range(math.ceil(n_batches / max_batch_size)), desc="Batch", leave=False, disable=suppress_output):
        initial_conditions_subset = tuple(
            comp[i * max_batch_size : (i + 1) * max_batch_size] for comp in initial_conditions
        )
        predictions_subset = _gen_preds(
            model=model,
            initial_conditions=initial_conditions_subset,
            n_steps=n_steps,
            device=device,
            suppress_output=suppress_output,
        )
        predicted_batches.append(predictions_subset)

    preds = tuple(np.concatenate(batched_comp_preds, axis=0) for batched_comp_preds in zip(*predicted_batches))

    model.train(was_training)

    return preds


def gen_preds_like(
    model,
    labels: Sequence[NDArray, NDArray, NDArray] | None = None,
    labels_path: str | None = None,
    max_batch_size: int = 64,
    device: None | torch.device = None,
    suppress_output: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate predictions for the same number of steps as labels and using initial conditions from labels.

    Args:
        model: Neural network model for trajectory prediction.
        labels (Sequence[NDArray, NDArray, NDArray] | None): Pre-loaded labels as (rho, Q, P) arrays. Each should include both a batch and time step dim
        labels_path (str | None): Path to pickled data file (if data not provided).
        max_batch_size (int): Maximum batch size.
        device (None | torch.device): PyTorch device. If None, one will automatically be chosen.
        suppress_output (bool): If True, suppress progress bars.

    Returns:
        tuple[NDArray, NDArray, NDArray]: Tuple of predictions.
    """

    if labels is None and labels_path is not None:
        labels = unpickle_data(labels_path)
    elif labels is None and labels_path is None:
        raise ValueError("labels and labels_path both None")

    ## Shape checks
    rho, Q, P = labels
    if rho.ndim != 4:
        raise ValueError(f"rho has invalid shape! Expected 4, got shape {rho.shape}")
    if Q.ndim != 3:
        raise ValueError(f"Q has invalid shape! Expected 3, got shape {Q.shape}")
    if P.ndim != 3:
        raise ValueError(f"P has invalid shape! Expected 3, got shape {P.shape}")

    n_steps = len(labels[0][0]) - 1
    initial_conditions = [comp[:, 0] for comp in labels]

    preds = gen_preds(
        model=model,
        initial_conditions=initial_conditions,
        n_steps=n_steps,
        max_batch_size=max_batch_size,
        device=device,
        suppress_output=suppress_output,
    )

    return preds


def compare_cdw_order_vis(labels, preds, pltargs: PLTArgs | None = None):
    """
    Visualize comparison of CDW order parameters between labels and predictions.

    Args:
        labels: Tuple of (rho, Q, P) label arrays (can be None).
        preds: Tuple of (rho, Q, P) prediction arrays (can be None).
        pltargs (PLTArgs, optional): Plotting arguments configuration.

    Returns:
        dict or None: Dictionary with 'rho_img' and 'Q_img' if return_image is True.
    """
    if labels is not None:
        rho_label, Q_label, _ = labels

    if preds is not None:
        rho_pred, Q_pred, _ = preds

    # Calculate CDW orders
    if labels is not None:
        label_charge_order = calc_charge_order(rho_label)
        label_lattice_order = calc_lattice_order(Q_label)
    else:
        label_charge_order, label_lattice_order = None, None

    if preds is not None:
        pred_charge_order = calc_charge_order(rho_pred)
        pred_lattice_order = calc_lattice_order(Q_pred)
    else:
        pred_charge_order, pred_lattice_order = None, None

    # Plot
    rho_img = plot_comparison(
        label_charge_order,
        pred_charge_order,
        title="Charge Order",
        xlabel="Steps",
        ylabel="Charge Order",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    Q_img = plot_comparison(
        label_lattice_order,
        pred_lattice_order,
        title="Lattice Order",
        xlabel="Steps",
        ylabel="Lattice Order",
        **(vars(pltargs) if pltargs is not None else {}),
    )

    if pltargs.return_image:
        return {"rho_img": rho_img, "Q_img": Q_img}


def compare_cdw_order_autocorrelation(
    labels, preds, max_lag: int, group_avg: bool = True, pltargs: PLTArgs | None = None, return_data: bool = False
):
    """
    Compare autocorrelation functions of CDW order parameters.

    Args:
        labels: Tuple of (rho, Q, P) label arrays.
        preds: Tuple of (rho, Q, P) prediction arrays.
        max_lag (int): Maximum lag value for autocorrelation calculation.
        group_avg (bool): If True, average across batch dimension during calculation.
        pltargs (PLTArgs, optional): Plotting arguments configuration.
        return_data (bool): If True, return autocorrelation data in addition to plots.

    Returns:
        dict or None: Dictionary with images and/or autocorrelation data.
    """
    (rho_label, Q_label, __), (rho_pred, Q_pred, _) = labels, preds

    # Calculate CDW orders
    label_charge_order = calc_charge_order(rho_label)
    pred_charge_order = calc_charge_order(rho_pred)

    label_lattice_order = calc_lattice_order(Q_label)
    pred_lattice_order = calc_lattice_order(Q_pred)

    # Calculate Lags
    label_charge_order_autocorr_traj = calc_autocorrelation_traj(
        label_charge_order, max_lag=max_lag, group_avg=group_avg
    )
    pred_charge_order_autocorr_traj = calc_autocorrelation_traj(pred_charge_order, max_lag=max_lag, group_avg=group_avg)

    label_lattice_order_autocorr_traj = calc_autocorrelation_traj(
        label_lattice_order, max_lag=max_lag, group_avg=group_avg
    )
    pred_lattice_order_autocorr_traj = calc_autocorrelation_traj(
        pred_lattice_order, max_lag=max_lag, group_avg=group_avg
    )

    # If the preds and labels were batched, then these autocorrelation trajectories are now 2d arrays of shape [batch_size, max_lag+1]. We now average across the batch dim
    # Pre-averaged means group was averaged during autocorr calculation. Post-average means autocorrelation traj were averaged across batch after calculation
    annotation_info = "" if group_avg else " [Group Post-Averaged]"
    if label_charge_order_autocorr_traj.ndim == 2:
        if group_avg:
            raise RuntimeError(
                f"If group_avg enabled traj should not have 2 dims. But got traj shape: {label_charge_order_autocorr_traj.shape}"
            )
        annotation_info = " [Group Post-Averaged]"
        label_charge_order_autocorr_traj = np.mean(label_charge_order_autocorr_traj, axis=0)
        pred_charge_order_autocorr_traj = np.mean(pred_charge_order_autocorr_traj, axis=0)
        label_lattice_order_autocorr_traj = np.mean(label_lattice_order_autocorr_traj, axis=0)
        pred_lattice_order_autocorr_traj = np.mean(pred_lattice_order_autocorr_traj, axis=0)

    # Plot
    rho_img = plot_comparison(
        label_charge_order_autocorr_traj,
        pred_charge_order_autocorr_traj,
        title="Charge order autocorr traj" + annotation_info,
        xlabel="Lag",
        ylabel="Autocorr",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    Q_img = plot_comparison(
        label_lattice_order_autocorr_traj,
        pred_lattice_order_autocorr_traj,
        title="Lattice order autocorr traj" + annotation_info,
        xlabel="Lag",
        ylabel="Autocorr",
        **(vars(pltargs) if pltargs is not None else {}),
    )

    return_dict = {}

    if pltargs.return_image:
        return_dict.update({"rho_img": rho_img, "Q_img": Q_img})

    if return_data:
        return_dict.update(
            {
                "rho_label_autocorr": label_charge_order_autocorr_traj,
                "Q_label_autocorr": label_lattice_order_autocorr_traj,
                "rho_pred_autocorr": pred_charge_order_autocorr_traj,
                "Q_pred_autocorr": pred_lattice_order_autocorr_traj,
            }
        )

    return return_dict or None


def plot_label_cdw_order_autocorrelation(labels, max_lag: int, group_avg: bool = True, pltargs: PLTArgs | None = None):
    """
    Plot autocorrelation function of CDW order parameters for labels only.

    Args:
        labels: Tuple of (rho, Q, P) label arrays.
        max_lag (int): Maximum lag value for autocorrelation calculation.
        group_avg (bool): If True, average across batch dimension during calculation.
        pltargs (PLTArgs, optional): Plotting arguments configuration.

    Returns:
        dict or None: Dictionary with 'rho_img' and 'Q_img' if return_image is True.
    """
    (rho_label, Q_label, __) = labels

    # Calculate CDW orders
    label_charge_order = calc_charge_order(rho_label)

    label_lattice_order = calc_lattice_order(Q_label)

    # Calculate Lags
    label_charge_order_autocorr_traj = calc_autocorrelation_traj(
        label_charge_order, max_lag=max_lag, group_avg=group_avg
    )
    label_lattice_order_autocorr_traj = calc_autocorrelation_traj(
        label_lattice_order, max_lag=max_lag, group_avg=group_avg
    )

    # If the labels were batched, then these autocorrelation trajectories are now 2d arrays of shape [batch_size, max_lag+1]. We now average across the batch dim
    # Pre-averaged means group was averaged during autocorr calculation. Post-average means autocorrelation traj were averaged across batch after calculation
    annotation_info = "" if not group_avg else " [Group Pre-Averaged]"
    if label_charge_order_autocorr_traj.ndim == 2:
        if group_avg:
            raise RuntimeError(
                f"If group_avg enabled traj should not have 2 dims. But got traj shape: {label_charge_order_autocorr_traj.shape}"
            )
        annotation_info = " [Group Post-Averaged]"
        label_charge_order_autocorr_traj = np.mean(label_charge_order_autocorr_traj, axis=0)
        label_lattice_order_autocorr_traj = np.mean(label_lattice_order_autocorr_traj, axis=0)

    # Plot
    rho_img = plot_line(
        label_charge_order_autocorr_traj,
        title="Label charge order autocorr traj" + annotation_info,
        xlabel="Lag",
        ylabel="Autocorr",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    Q_img = plot_line(
        label_lattice_order_autocorr_traj,
        title="Label lattice order autocorr traj" + annotation_info,
        xlabel="Lag",
        ylabel="Autocorr",
        **(vars(pltargs) if pltargs is not None else {}),
    )

    if pltargs.return_image:
        return {"rho_img": rho_img, "Q_img": Q_img}


def compare_site_vis(labels, preds, site=0, pltargs: PLTArgs | None = None):
    """
    Visualize comparison of single site dynamics between labels and predictions.

    Args:
        labels: Tuple of (rho, Q, P) label arrays.
        preds: Tuple of (rho, Q, P) prediction arrays.
        site (int): Site index to visualize.
        pltargs (PLTArgs, optional): Plotting arguments configuration.

    Returns:
        dict or None: Dictionary with 'rho_img', 'Q_img', 'P_img' if return_image is True.
    """
    (rho_label, Q_label, P_label), (rho_pred, Q_pred, P_pred) = labels, preds

    """ Extract real diags from rho """

    rho_label_diag = np.diagonal(rho_label, axis1=-2, axis2=-1).real
    rho_pred_diag = np.diagonal(rho_pred, axis1=-2, axis2=-1).real

    """ Extract sites """

    labels = [rho_label_diag, Q_label, P_label]
    preds = [rho_pred_diag, Q_pred, P_pred]

    labels_site = [comp[..., site] for comp in labels]
    preds_site = [comp[..., site] for comp in preds]

    rho_label_site, Q_label_site, P_label_site = labels_site
    rho_pred_site, Q_pred_site, P_pred_site = preds_site

    """ Plot """

    rho_img = plot_comparison(
        rho_label_site,
        rho_pred_site,
        title=f"rho site {site}",
        xlabel="Steps",
        ylabel="val",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    Q_img = plot_comparison(
        Q_label_site,
        Q_pred_site,
        title=f"Q site {site}",
        xlabel="Steps",
        ylabel="val",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    P_img = plot_comparison(
        P_label_site,
        P_pred_site,
        title=f"P site {site}",
        xlabel="Steps",
        ylabel="val",
        **(vars(pltargs) if pltargs is not None else {}),
    )

    if pltargs.return_image:
        return {"rho_img": rho_img, "Q_img": Q_img, "P_img": P_img}


def predict_site_diffs(labels, preds, site=0, pltargs: PLTArgs | None = None):
    """
    Plot the differences between predictions and labels at a specific site.

    Args:
        labels: Tuple of (rho, Q, P) label arrays.
        preds: Tuple of (rho, Q, P) prediction arrays.
        site (int): Site index to analyze.
        pltargs (PLTArgs, optional): Plotting arguments configuration.

    Returns:
        dict or None: Dictionary with 'rho_img', 'Q_img', 'P_img' if return_image is True.
    """
    (rho_label, Q_label, P_label), (rho_pred, Q_pred, P_pred) = labels, preds

    """ Extract real diags from rho """

    rho_label_diag = np.diagonal(rho_label, axis1=-2, axis2=-1).real
    rho_pred_diag = np.diagonal(rho_pred, axis1=-2, axis2=-1).real

    """ Extract sites """

    labels = [rho_label_diag, Q_label, P_label]
    preds = [rho_pred_diag, Q_pred, P_pred]

    labels_site = [comp[..., site] for comp in labels]
    preds_site = [comp[..., site] for comp in preds]

    """ Find Diffs """

    rho_diff, Q_diff, P_diff = [label - pred for label, pred in zip(labels_site, preds_site)]

    """ Plot """

    rho_img = plot_line(
        rho_diff,
        title=f"rho site {site} diffs",
        xlabel="Steps",
        ylabel="Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    Q_img = plot_line(
        Q_diff,
        title=f"Q site {site} diffs",
        xlabel="Steps",
        ylabel="Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    P_img = plot_line(
        P_diff,
        title=f"P site {site} diffs",
        xlabel="Steps",
        ylabel="Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )

    if pltargs.return_image:
        return {"rho_img": rho_img, "Q_img": Q_img, "P_img": P_img}


def predict_rho_offsite_diff_magnitudes(rho_label, rho_pred, site=(0, 0), pltargs: PLTArgs | None = None):
    """Extract site from rho"""

    # rho_pred = rho_pred[..., *site]
    # rho_label = rho_label[..., *site]
    rho_pred = rho_pred[(..., *site)]
    rho_label = rho_label[(..., *site)]

    """ Get diff magnitude """
    diff = np.abs(rho_label - rho_pred)

    # Plot

    return plot_line(
        diff,
        title=f"rho site {site} diff magnitude",
        xlabel="Steps",
        ylabel="Abs Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )


def rho_offsite_diff_magnitudes_at_step(rho_label, rho_pred, row=0, step=1, pltargs: PLTArgs | None = None):
    """Choose Step and Row"""

    rho_pred = rho_pred[step, row]
    rho_label = rho_label[step, row]

    """ Get diff magnitude """
    diff = np.abs(rho_label - rho_pred)

    # Plot

    return plot_line(
        diff,
        title=f"rho diff magnitudes at step {step} and row {row}",
        xlabel="Column #",
        ylabel="Abs Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )


def Q_and_P_diffs_at_step(labels, preds, step=1, pltargs: PLTArgs | None = None):
    """Choose Step"""
    labels = [comp[step] for comp in labels]
    preds = [comp[step] for comp in preds]

    _, Q_label, P_label = labels
    _, Q_pred, P_pred = preds

    """ Get Diff """

    Q_diff = Q_label - Q_pred
    P_diff = P_label - P_pred

    # Plot

    Q_img = plot_line(
        Q_diff,
        title=f"Q diffs at step {step}",
        xlabel="Site #",
        ylabel="Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )
    P_img = plot_line(
        P_diff,
        title=f"P diffs at step {step}",
        xlabel="Site #",
        ylabel="Diff",
        **(vars(pltargs) if pltargs is not None else {}),
    )

    if pltargs.return_image:
        return {"Q_img": Q_img, "P_img": P_img}


def calc_autocorrelation(traj: np.ndarray, lag: int, group_avg: bool = True):
    """
    Calculate autocorrelation of a trajectory at a specific lag.

    The autocorrelation is normalized by the variance.

    Args:
        traj (np.ndarray): Trajectory array of shape [..., N] where N is time dimension.
        lag (int): Time lag for autocorrelation calculation.
        group_avg (bool): If True, average over both batch and time dimensions.

    Returns:
        np.ndarray: Autocorrelation value(s) of shape [...] or scalar if group_avg.
    """
    """
    Args:
        traj (np.ndarray): An array of shape [..., N], representing a (possibly batched) scalar trajectory
        lag (int): The lag at which to compute the autocorrelation
        group_avg (bool): If True, shape must be [batch_size, N]. The mean operations will then be over both the time and batch dims, not just time dim

    Returns:
        np.ndarray: An array of shape [...] representing the calculated autocorrelation value. If group_avg is True then result will be a
    """

    if lag >= traj.shape[-1]:
        raise ValueError("Lag must be less than the length of the trajectory")

    if group_avg and traj.ndim != 2:
        raise ValueError(
            f"If group_avg is enabled then traj must have two dims representing shape [batch_size, N]. Got: {traj.shape}"
        )

    mean = np.mean(traj, axis=-1 if not group_avg else None)

    mean_sq = np.mean(np.square(traj), axis=-1 if not group_avg else None)

    front_lag = traj[..., lag:]
    rear_lag = traj[..., : -lag if lag != 0 else None]
    lagged_prod = front_lag * rear_lag

    lagged_prod_avg = np.mean(lagged_prod, axis=-1 if not group_avg else None)

    autocorrelation = (lagged_prod_avg - mean**2) / (mean_sq - mean**2)

    return autocorrelation


def calc_autocorrelation_traj(traj: np.ndarray, max_lag: int, group_avg: bool = True):
    """
    Calculate autocorrelation function for all lags from 0 to max_lag.

    Args:
        traj (np.ndarray): Trajectory array of shape [..., N] where N is time dimension.
        max_lag (int): Maximum lag value (inclusive) for autocorrelation calculation.
        group_avg (bool): If True, average over both batch and time dimensions.

    Returns:
        np.ndarray: Autocorrelation trajectory of shape [..., max_lag+1].
    """
    """
    Args:
        traj (np.ndarray): An array of shape [..., N], representing a (possibly batched) scalar trajectory
        max_lag (int): The value up to which the autocorrelation traj will be computed (inclusive)

    Returns:
        np.ndarray: An array of shape [..., max_lag+1] representing the (possibly batched) calculated autocorrelation trajectory from a lag of 0 up to max_lag (inclusive)
    """

    autocorr_traj = []
    for lag in range(max_lag + 1):
        autocorr_traj.append(calc_autocorrelation(traj, lag=lag, group_avg=group_avg))

    autocorr_traj = np.stack(autocorr_traj, axis=-1)

    return autocorr_traj


def calc_ranges(x: NDArray) -> NDArray:
    """
    Calculate the range (max - min) along the last dimension.

    Args:
        x (NDArray): Input array.

    Returns:
        NDArray: Range values with last dimension reduced.
    """
    return np.max(x, axis=-1) - np.min(x, axis=-1)


def analyze_cdw_order_div(
    labels: tuple[NDArray],
    preds: tuple[NDArray],
    relative_div_threshold: float = 1.1,
):
    """
    Analyze predictions for CDW order divergence compared to labels.

    A batch element is considered divergent if either its charge order or lattice order exceeds a range
    (max - min) greater than the maximum range seen in the labels multiplied by the threshold.

    Args:
        labels (tuple[NDArray]): Tuple of arrays for rho and Q (additional elements ignored).
            rho: complex-valued with shape (batch, step, L, L)
            Q: real-valued with shape (batch, step, L)
        preds (tuple[NDArray]): Prediction arrays with same structure as labels.
        relative_div_threshold (float): Coefficient multiplied with maximum label range.
            Any batch exceeding this threshold is considered divergent.

    Returns:
        CDW_Order_Div_Returns: Named tuple containing divergence indices and masks.
    """

    """ Check and Process Args """

    rho_pred, Q_pred, *_ = preds
    rho_label, Q_label, *_ = labels

    if rho_pred.shape != rho_label.shape:
        raise ValueError(f"rho_pred ({rho_pred.shape}) did not have same shape as rho_label ({rho_label.shape})")
    if Q_pred.shape != Q_label.shape:
        raise ValueError(f"Q_pred ({Q_pred.shape}) did not have same shape as Q_label ({Q_label.shape})")

    """ Calc CDW orders and Maximum Ranges """

    # Each CDW order tensor has shape (batches, steps)
    pred_charge_order = calc_charge_order(rho_pred)
    pred_lattice_order = calc_lattice_order(Q_pred)
    label_charge_order = calc_charge_order(rho_label)
    label_lattice_order = calc_lattice_order(Q_label)

    charge_range_threshold = relative_div_threshold * max(calc_ranges(label_charge_order))
    lattice_range_threshold = relative_div_threshold * max(calc_ranges(label_lattice_order))

    """ Find Divergent Predictions """

    pred_charge_ranges = calc_ranges(pred_charge_order)
    pred_lattice_ranges = calc_ranges(pred_lattice_order)

    charge_div_mask = pred_charge_ranges > charge_range_threshold
    lattice_div_mask = pred_lattice_ranges > lattice_range_threshold
    combined_div_mask = charge_div_mask | lattice_div_mask

    charge_div_idxs = np.where(charge_div_mask)[0]
    lattice_div_idxs = np.where(lattice_div_mask)[0]
    combined_div_idxs = np.where(combined_div_mask)[0]

    """ Return """

    CDW_Order_Div_Returns = namedtuple(
        "CDW_Order_Div_Returns",
        [
            "combined_div_idxs",
            "charge_div_idxs",
            "lattice_div_idxs",
            "combined_div_mask",
            "charge_div_mask",
            "lattice_div_mask",
        ],
    )

    return CDW_Order_Div_Returns(
        combined_div_idxs, charge_div_idxs, lattice_div_idxs, combined_div_mask, charge_div_mask, lattice_div_mask
    )


def np_rmse(x: NDArray, y: NDArray, axis: int | tuple[int] | None = None) -> float:
    """
    Calculate root mean squared error between two arrays.

    Args:
        x (NDArray): First array.
        y (NDArray): Second array (must have same shape as x).
        axis (int | tuple[int] | None): Axis or axes along which the mean is computed. The default is to compute the mean of the flattened array.
    Returns:
        float: RMSE value.
    """
    return np.sqrt(np.mean(np.abs(np.square(x - y)), axis=axis))


def stitch_images_grid(
    images: Sequence[Image.Image],
    n_rows: int,
    n_columns: int,
    padding_color="white",
    image_padding: int | tuple[int, int] = 0,
):
    """
    Arrange multiple images into a grid layout.

    Args:
        images (Sequence[Image.Image]): List of PIL images to arrange.
        n_rows (int): Number of rows in the grid.
        n_columns (int): Number of columns in the grid.
        padding_color (str): Color for padding between images.
        image_padding (int | tuple[int, int]): Padding around each image (width, height).

    Raises:
        ValueError: If not enough grid positions for all images.
    """
    """Validation & Processing"""

    if isinstance(image_padding, int):
        image_padding_width = image_padding
        image_padding_height = image_padding
    else:
        image_padding_width, image_padding_height = image_padding

    n_images = len(images)

    if n_rows * n_columns < n_images:
        raise ValueError(
            f"Not enough rows/columns for all images. There were {n_images} images but only {n_rows} rows * {n_columns} columns = {n_rows * n_columns} grid spots"
        )

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    if any(image.width != max_width for image in images):
        print(f"Not all images had the same width. They will all be padded to the max width ({max_width})")
    if any(image.height != max_height for image in images):
        print(f"Not all images had the same height. They will all be padded to the max height ({max_height})")

    """ Creating Stitched Image """

    padded_image_width = max_width + 2 * image_padding_width
    padded_image_height = max_height + 2 * image_padding_height

    # Padding
    images = [
        ImageOps.pad(image, size=(padded_image_width, padded_image_height), color=padding_color) for image in images
    ]

    canvas_width = n_columns * padded_image_width
    canvas_height = n_rows * padded_image_height

    canvas = Image.new("RGB", size=(canvas_width, canvas_height), color=padding_color)

    for idx, image in enumerate(images):
        row, col = divmod(idx, n_columns)

        x0 = col * padded_image_width
        y0 = row * padded_image_height

        canvas.paste(image, (x0, y0))

    display(canvas)
