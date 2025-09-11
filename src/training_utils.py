# Standard library imports
import math
import warnings

# Third-party imports
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Local imports
from src.modeling import CNNnd, HolsteinStepCombined
from src.physics import calc_charge_order, calc_lattice_order
from src.utils import get_off_diagonal


def get_warm_up_lr_scheduler_func(warmup_steps):
    """
    Create a learning rate scheduler function for warm-up only.

    Args:
        warmup_steps (int): Number of steps for warm-up. Must be positive.

    Returns:
        callable: Function that takes step number and returns LR multiplier.
    """
    if warmup_steps <= 0:
        warmup_steps = 1
        warnings.warn(
            f"my_utils.py get_warm_up_lr_scheduler(): warmup_steps cannot be 0 or negative! You entered: {warmup_steps}. Setting warmup_steps to 1"
        )

    def lr_scheduler(step_num):
        return min(1, step_num / warmup_steps)

    return lr_scheduler


def get_cosine_annealing_restart_warmup_scheduler(
    optimizer: torch.optim.Optimizer, stage_lengths: list[int], warmup_steps: int = 500, eta_min: float = 1e-2
) -> optim.lr_scheduler.SequentialLR:
    """
    Returns a lr scheduler that implements cosine annealing with restarts on stages of variable length
    Each stage is also begun via a linear warmup that starts at 1e-4 times the learning rate

    Arguments:
        optimizer (torch.optim.Optimizer): The optimizer to be used
        stage_lengths (list[int]): The lengths of each stage in steps
        warmup_steps (int): The number of steps to warm up for at the start of each cosine restart
        eta_min (float): The minimum learning rate multiplier to be used in the cosine annealing

    """
    cumulative = 0

    schedulers, milestones = [], []
    for L in stage_lengths:
        cosine_steps = L - warmup_steps

        schedulers.append(  # linear warm-up
            LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
        )
        schedulers.append(  # cosine decay
            CosineAnnealingLR(optimizer, T_max=L - warmup_steps, eta_min=eta_min)
        )

        cumulative += warmup_steps
        milestones.append(cumulative)
        cumulative += cosine_steps
        milestones.append(cumulative)

    # Pop the last milestone since that is not needed by SequentialLR
    milestones.pop()

    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def torch_RMSE(tnsr_1, tnsr_2) -> torch.Tensor:
    """
    Calculate root mean squared error between two tensors.

    Compatible with complex dtypes by taking absolute value of squared differences.

    Args:
        tnsr_1 (torch.Tensor): First tensor.
        tnsr_2 (torch.Tensor): Second tensor (must have same shape).

    Returns:
        torch.Tensor: Scalar RMSE value.

    Raises:
        ValueError: If tensors have different shapes.
    """
    if tnsr_1.shape != tnsr_2.shape:
        raise ValueError(f"torch_RMSE(): tnsr_1 has shape {tnsr_1.shape} but tnsr_2 has different shape {tnsr_2.shape}")

    return torch.sqrt(torch.mean(torch.abs(torch.square(tnsr_1 - tnsr_2))))


def torch_RMS(tnsr) -> torch.Tensor:
    """
    Calculate root mean square of a tensor.

    Compatible with complex dtypes by taking absolute value before squaring.

    Args:
        tnsr (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Scalar RMS value.
    """

    return torch.sqrt(torch.mean(torch.abs(torch.square(tnsr))))


def torch_MSE(tnsr_1, tnsr_2) -> torch.Tensor:
    """
    Calculate mean squared error between two tensors.

    Compatible with complex dtypes by taking absolute value of squared differences.

    Args:
        tnsr_1 (torch.Tensor): First tensor.
        tnsr_2 (torch.Tensor): Second tensor (must have same shape).

    Returns:
        torch.Tensor: Scalar MSE value.

    Raises:
        ValueError: If tensors have different shapes.
    """
    if tnsr_1.shape != tnsr_2.shape:
        raise ValueError(f"torch_RMSE(): tnsr_1 has shape {tnsr_1.shape} but tnsr_2 has different shape {tnsr_2.shape}")

    return torch.mean(torch.abs(torch.square(tnsr_1 - tnsr_2)))


def batch_to_output(
    model, batch, device, return_labels=False, n_step=1, return_multiple_steps=False, checkpointing: bool = False
):
    """
    Process a batch through the model and return predictions.

    Args:
        model: Neural network model.
        batch: Tuple of (inputs, labels) from dataloader.
        device: PyTorch device to use for computation.
        return_labels (bool | str): If True, return labels. If 'deviced', return labels on device.
        n_step (int): Number of prediction steps.
        return_multiple_steps (bool): If True, return all intermediate steps.
        checkpointing (bool): If True, use gradient checkpointing.

    Returns:
        tuple: Model predictions, optionally with labels.

    Raises:
        ValueError: If shapes or dtypes of predictions and labels don't match.
    """

    deviced = False
    if return_labels == "deviced":
        deviced = True
        return_labels = True

    input_, labels = batch
    rho_in, Q_in, P_in = [tnsr.to(device) for tnsr in input_]

    if len(labels) == 6:
        return_derivatives = True
        rho_label, Q_label, P_label, drho_label, dQ_label, dP_label = labels
    elif len(labels) == 3:
        return_derivatives = False
        rho_label, Q_label, P_label = labels
    else:
        raise ValueError(f"batch_to_output(): len of labels was {len(labels)}! Only 6 or 3 are acceptable!")

    preds = model(
        rho_in,
        Q_in,
        P_in,
        return_derivatives=return_derivatives,
        n_step=n_step,
        return_multiple_steps=return_multiple_steps,
        checkpointing=checkpointing,
    )

    ## Test for shape and dtype consistency

    rho_pred, Q_pred, P_pred, *_ = preds
    if rho_label.shape != rho_pred.shape or Q_label.shape != Q_pred.shape or P_label.shape != P_pred.shape:
        raise ValueError(
            f"batch_to_output: Shape inconsistency between rho/Q/P predictions and labels detected! Shapes:\n\trho label:\t{rho_label.shape}rho pred:\t{rho_pred.shape}\n\tQ label:\t{Q_label.shape}Q pred:\t{Q_pred.shape}\n\tP label:\t{P_label.shape}P pred:\t{P_pred.shape}"
        )
    if rho_label.dtype != rho_pred.dtype or Q_label.dtype != Q_pred.dtype or P_label.dtype != P_pred.dtype:
        raise ValueError(
            f"batch_to_output: dtype inconsistency between rho/Q/P predictions and labels detected! dtypes:\n\trho label:\t{rho_label.dtype}rho pred:\t{rho_pred.dtype}\n\tQ label:\t{Q_label.dtype}Q pred:\t{Q_pred.dtype}\n\tP label:\t{P_label.dtype}P pred:\t{P_pred.dtype}"
        )

    if return_derivatives:
        drho_pred, dQ_pred, dP_pred = preds[3:]
        if drho_label.shape != drho_pred.shape or dQ_label.shape != dQ_pred.shape or dP_label.shape != dP_pred.shape:
            raise ValueError(
                f"batch_to_output: Shape inconsistency between drho/dQ/dP predictions and labels detected! Shapes:\n\tdrho label:\t{drho_label.shape}drho pred:\t{drho_pred.shape}\n\tdQ label:\t{dQ_label.shape}dQ pred:\t{dQ_pred.shape}\n\tdP label:\t{dP_label.shape}dP pred:\t{dP_pred.shape}"
            )
        if drho_label.dtype != drho_pred.dtype or dQ_label.dtype != dQ_pred.dtype or dP_label.dtype != dP_pred.dtype:
            raise ValueError(
                f"batch_to_output: dtype inconsistency between drho/dQ/dP predictions and labels detected! dtypes:\n\tdrho label:\t{drho_label.dtype}drho pred:\t{drho_pred.dtype}\n\tdQ label:\t{dQ_label.dtype}dQ pred:\t{dQ_pred.dtype}\n\tdP label:\t{dP_label.dtype}dP pred:\t{dP_pred.dtype}"
            )

    if not return_labels:
        return preds
    else:
        if deviced:
            labels = tuple([tnsr.to(device) for tnsr in labels])
        return preds, labels


def batch_to_loss(
    model,
    batch,
    device,
    n_step=1,
    return_multiple_steps=False,
    deriv_loss_coeff=1.0,
    separate_rho_diag=False,
    return_loss_dict=False,
    stepwise_loss_scale_coeff=None,
    dtype=torch.float32,
    checkpointing: bool = False,
):
    """
    Calculate loss for a batch of data.

    Args:
        model: Neural network model.
        batch: Tuple of (inputs, labels) from dataloader.
        device: PyTorch device.
        n_step (int): Number of prediction steps.
        return_multiple_steps (bool): If True, use multi-step predictions.
        deriv_loss_coeff (float): Coefficient for derivative loss component.
        separate_rho_diag (bool): If True, separate diagonal and off-diagonal rho losses.
        return_loss_dict (bool): If True, return detailed loss breakdown.
        stepwise_loss_scale_coeff (float | None): Exponential scaling for later steps.
        dtype (torch.dtype): Data type for computations.
        checkpointing (bool): If True, use gradient checkpointing.

    Returns:
        torch.Tensor | tuple: Total loss, optionally with loss dictionary.
    """
    preds, labels = batch_to_output(
        model,
        batch,
        device,
        return_labels="deviced",
        n_step=n_step,
        return_multiple_steps=return_multiple_steps,
        checkpointing=checkpointing,
    )

    assert len(preds) == len(labels), (
        f"batch_to_loss(): preds and labels have different lengths! {len(preds)} vs {len(labels)}"
    )
    if len(preds) == 6:
        deriv_components = True
    elif len(preds) == 3:
        deriv_components = False
    else:
        raise ValueError(
            f"batch_to_loss(): preds and labels have invalid length! Only 3 or 6 are acceptable! But they had length {len(preds)}"
        )

    if separate_rho_diag:

        def separate_rho_in_components(comps: list[torch.Tensor]):
            """
            Separate density matrix components into diagonal and off-diagonal parts.

            Args:
                comps (list[torch.Tensor]): Components [rho, Q, P, ...]

            Returns:
                list[torch.Tensor]: Expanded components with rho split into diagonal and off-diagonal parts.
                    Returns [rho_diag, rho_off_diag, Q, P, ...] if deriv_components is False.
                    Returns [rho_diag, rho_off_diag, Q, P, drho_diag, drho_off_diag, ...] if deriv_components is True.
            """

            rho = comps[0]

            off_diag_comps = [torch.diagonal(rho, dim1=-2, dim2=-1), get_off_diagonal(rho), *comps[1:3]]

            if deriv_components:
                drho = comps[3]
                off_diag_comps += [
                    torch.diagonal(drho, dim1=-2, dim2=-1),
                    get_off_diagonal(drho),
                    *comps[4:],
                ]

            return off_diag_comps

        preds = separate_rho_in_components(preds)
        labels = separate_rho_in_components(labels)

    errors = [label - pred for pred, label in zip(preds, labels)]

    # Scale the error exponentially according to the step number and the stepwise_loss_scale_coeff
    # Emphasizes later predicted steps
    # The scaling is a little weird due to RMSE. Because the error for all steps is calculated at once, the scaling is applied to the error before RMS
    if stepwise_loss_scale_coeff is not None and return_multiple_steps:

        def broadcast_to_second_dim(x: torch.Tensor, dims: int):
            """
            Given a 1d tensor x return a reshaped version which has dims dimensions and length 1 in all dims except for dim 1 (0-indexed)

            Example:
            x.shape
            >> [9]
            broadcast_to_second_dim(x, 5).shape
            >> [1, 9, 1, 1, 1]

            Args:
                x (torch.Tensor): 1D tensor.
                dims (int): Target number of dimensions.

            Returns:
                torch.Tensor: Reshaped tensor with shape [1, len(x), 1, ...].
            """
            assert x.dim() == 1, f"broadcast_to_second_dim(): x has {x.dim()} dims! It should have 1 dim!"
            assert dims >= 2, f"broadcast_to_second_dim(): dims was {dims}! It should be at least 2!"

            shape = [1] * dims
            shape[1] = -1

            return x.reshape(shape)

        step_indices = torch.arange(n_step, dtype=dtype, device=device)
        stepwise_scaling_weights = stepwise_loss_scale_coeff**step_indices
        stepwise_scaling_weights = stepwise_scaling_weights / stepwise_scaling_weights.sum()
        errors = [error * broadcast_to_second_dim(stepwise_scaling_weights, error.dim()) for error in errors]

    losses = [torch_RMS(error) for error in errors]

    if deriv_components:
        total_loss = sum(losses[: len(losses) // 2]) + deriv_loss_coeff * sum(losses[len(losses) // 2 :])
    else:
        total_loss = sum(losses)

    if return_loss_dict:
        losses = [comp.item() for comp in losses]

        loss_dict = {"total": total_loss.item()}
        if separate_rho_diag:
            loss_dict |= {"rho_diag": losses[0], "rho_off_diag": losses[1], "Q": losses[2], "P": losses[3]}
        else:
            loss_dict |= {"rho": losses[0], "Q": losses[1], "P": losses[2]}
        if deriv_components:
            if separate_rho_diag:
                loss_dict |= {"drho_diag": losses[4], "drho_off_diag": losses[5], "dQ": losses[6], "dP": losses[7]}
            else:
                loss_dict |= {"drho": losses[3], "dQ": losses[4], "dP": losses[5]}

    if return_loss_dict:
        return total_loss, loss_dict
    return total_loss


def calc_val_stats(
    model,
    val_loader,
    device,
    n_step=1,
    return_multiple_steps=False,
    return_as_dict=False,
    deriv_loss_coeff=None,
    return_cdw_order_rmse=False,
):
    """
    Calculate validation statistics across a dataset.

    Computes RMSE for each component (rho, Q, P) and optionally their derivatives
    and CDW order parameters. Uses batch-wise MSE calculation for memory efficiency.

    Args:
        model (nn.Module): Holstein model to evaluate.
        val_loader (torch.DataLoader): Validation data loader.
        device: PyTorch device for computation.
        n_step (int): Number of prediction steps.
        return_multiple_steps (bool): If True, use multi-step predictions.
        return_as_dict (bool): If True, return results as dictionary.
        deriv_loss_coeff (float | None): Coefficient for derivative loss.
        return_cdw_order_rmse (bool): If True, calculate CDW order RMSE.

    Returns:
        tuple | dict: RMSE values for total and each component.
    """

    training_mode = model.training

    return_derivatives = val_loader.dataset.return_derivatives

    all_batch_MSEs = [[] for _ in range(6 if return_derivatives else 3)]

    if return_cdw_order_rmse:
        # Will contain the MSEs for the order parameters of rho and Q for each batch
        all_batch_cdw_order_MSEs = [[] for _ in range(2)]

    def calculate_batch_MSEs(all_batch_MSEs, preds, labels, this_batch_size, loader_batch_size):
        """
        Calculate and store batch-wise MSE values.

        Computes MSE for each component, scales by batch size for proper averaging.

        Args:
            all_batch_MSEs: List to store MSE values.
            preds: Model predictions.
            labels: Ground truth labels.
            this_batch_size: Size of current batch.
            loader_batch_size: Standard batch size for scaling.
        """
        # Calculate MSE for each component and put onto CPU
        MSEs = [torch_MSE(pred, label).item() for pred, label in zip(preds, labels)]
        # Weigh MSEs by the size of this batch
        MSEs = [MSE * this_batch_size / val_loader.batch_size for MSE in MSEs]
        # Append to the list of all batch MSEs
        [all_batch_MSEs[i].append(MSE) for i, MSE in enumerate(MSEs)]

    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            this_batch_size = len(batch[0][0])

            preds, labels = batch_to_output(
                model,
                batch,
                device,
                return_labels="deviced",
                n_step=n_step,
                return_multiple_steps=return_multiple_steps,
            )

            calculate_batch_MSEs(all_batch_MSEs, preds, labels, this_batch_size, val_loader.batch_size)

            if return_cdw_order_rmse:
                rho_pred, Q_pred, rho_label, Q_label = preds[0], preds[1], labels[0], labels[1]
                cdw_order_preds = [calc_charge_order(rho_pred), calc_lattice_order(Q_pred)]
                cdw_order_labels = [calc_charge_order(rho_label), calc_lattice_order(Q_label)]
                calculate_batch_MSEs(
                    all_batch_cdw_order_MSEs, cdw_order_preds, cdw_order_labels, this_batch_size, val_loader.batch_size
                )

    # Return model to original training mode
    model.train(training_mode)

    # Calculate the RMSE for each component by taking the mean of the MSEs across all batches and then taking the square root
    RMSEs = [math.sqrt(sum(component_MSEs) / len(component_MSEs)) for component_MSEs in all_batch_MSEs]
    if return_cdw_order_rmse:
        total_cdw_order_RMSEs = [
            math.sqrt(sum(component_MSEs) / len(component_MSEs)) for component_MSEs in all_batch_cdw_order_MSEs
        ]
        total_cdw_order_RMSE = total_cdw_order_RMSEs[0] + total_cdw_order_RMSEs[1]

    if return_derivatives:
        total_RMSE = sum(RMSEs[:3]) + deriv_loss_coeff * sum(RMSEs[3:])
    else:
        total_RMSE = sum(RMSEs)

    if not return_as_dict:
        if return_cdw_order_rmse:
            return total_RMSE, *RMSEs, total_cdw_order_RMSE
        else:
            return total_RMSE, *RMSEs
    else:
        return_dict = {"total RMSE": total_RMSE, "RMSE rho": RMSEs[0], "RMSE Q": RMSEs[1], "RMSE P": RMSEs[2]}

        if return_derivatives:
            return_dict.update({"RMSE drho": RMSEs[3], "RMSE dQ": RMSEs[4], "RMSE dP": RMSEs[5]})

        if return_cdw_order_rmse:
            return_dict["total CDW order RMSE"] = total_cdw_order_RMSE

        return return_dict


def create_interpolated_lr_dict_for_optim(
    cnn: CNNnd | HolsteinStepCombined, start_lr: float, end_lr: float
) -> list[dict]:
    """
    Given a CNNnd instance, returns a list of dict representing parameter groups with corresponding learning rates for input into an optim optimizer.
    The first block will be given start_lr. The last block end_lr. And the blocks in-between will have their LR linearly interpolated.


    Args:
        cnn (CNNnd | HolsteinStepCombined): The CNN to create the dict with. If a HolsteinStepCombined instance is given its CNNnd will be extracted
        start_lr (float): The LR for the first block of the CNN
        end_lr (float): The LR for the last block of the CNN

    Returns:
        list[dict]: The list of dicts representing parameter groups and their corresponding LR
    """

    if isinstance(cnn, HolsteinStepCombined):
        cnn = cnn.CNN

    n_blocks = cnn.n_blocks

    lrs = np.linspace(start_lr, end_lr, n_blocks)

    param_dicts = []

    for block, lr in zip(cnn.blocks, lrs):
        param_dicts.append({"params": block.parameters(), "lr": lr})

    # Add input conv to first group
    first_dict = param_dicts[0]
    first_dict["params"] = list(first_dict["params"]) + list(cnn.input_conv.parameters())

    # Add output conv to last group
    last_dict = param_dicts[-1]
    last_dict["params"] = list(last_dict["params"]) + list(cnn.output_conv.parameters())

    return param_dicts
