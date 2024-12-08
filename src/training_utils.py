import torch

# a lr scheduler only for initial warm-up
def get_warm_up_lr_scheduler(warmup_steps):
    if warmup_steps <= 0:
        warmup_steps = 1
        warnings.warn(f"my_utils.py get_warm_up_lr_scheduler(): warmup_steps cannot be 0 or negative! You entered: {warmup_steps}. Setting warmup_steps to 1")
    def lr_scheduler(step_num):
        return min(1, step_num/warmup_steps)
    return lr_scheduler



def batch_to_output(model, batch, device, return_labels=False, n_step=1, return_multiple_steps=False):
    '''
    Batch does not need to be sent to device beforehand 
    if return_labels = True it will return the labels along with the prediction
    if return_labels = 'deviced' it will return the labels along with predictions AFTER sending the labels to device
    
    '''

    deviced = False
    if return_labels == "deviced":
        deviced = True
        return_labels = True

    input, labels = batch
    rho_in, Q_in, P_in = [tnsr.to(device) for tnsr in input]

    rho_pred, Q_pred, P_pred = model(rho_in, Q_in, P_in, n_step=n_step, return_multiple_steps=return_multiple_steps)
    preds = rho_pred, Q_pred, P_pred

    rho_label, Q_label, P_label = labels
    if rho_label.shape != rho_pred.shape or Q_label.shape != Q_pred.shape or P_label.shape != P_pred.shape:
        raise Exception(f"batch_to_output: Shape inconsistency between rho/Q/P predictions and labels detected! Shapes:\n\trho label:\t{rho_label.shape}rho pred:\t{rho_pred.shape}\n\tQ label:\t{Q_label.shape}Q pred:\t{Q_pred.shape}\n\tP label:\t{P_label.shape}P pred:\t{P_pred.shape}")
    if rho_label.dtype != rho_pred.dtype or Q_label.dtype != Q_pred.dtype or P_label.dtype != P_pred.dtype:
        raise Exception(f"batch_to_output: dtype inconsistency between rho/Q/P predictions and labels detected! dtypes:\n\trho label:\t{rho_label.dtype}rho pred:\t{rho_pred.dtype}\n\tQ label:\t{Q_label.dtype}Q pred:\t{Q_pred.dtype}\n\tP label:\t{P_label.dtype}P pred:\t{P_pred.dtype}")

    if not return_labels:
        return preds
    else:
        if deviced:
            labels = tuple([tnsr.to(device) for tnsr in labels])
        return preds, labels



def batch_to_loss(model, batch, device, return_loss_separates=True, n_step=1, return_multiple_steps=False):

    preds, labels = batch_to_output(model, batch, device, return_labels='deviced', n_step=n_step, return_multiple_steps=return_multiple_steps)
    rho_pred, Q_pred, P_pred = preds
    rho_label, Q_label, P_label = labels

    rho_loss = torch_RMSE(rho_pred, rho_label)
    Q_loss = torch_RMSE(Q_pred, Q_label)
    P_loss = torch_RMSE(P_pred, P_label)

    total_loss = rho_loss + Q_loss + P_loss

    if not return_loss_separates:
        return total_loss
    else:
        return total_loss, (rho_loss, Q_loss, P_loss)


def torch_RMSE(tnsr_1, tnsr_2)->torch.Tensor:
    '''
    Compatible with complex dtypes
    '''
    return torch.sqrt(torch.mean(torch.abs(torch.square(tnsr_1-tnsr_2))))


def calc_val_RMSE(model, val_loader, device, n_step=1, return_multiple_steps=False, return_as_dict=False):
    '''
    model (nn.Module): HolsteinStepSeparate model
    val_loader (torch.DataLoader): A dataloader containing the val set
    n_step, return_multiple_steps: Settings of val_loader (known in HolsteinDataset as label_step_count and multi_step_labels respectively)
    return_as_dict (bool): If True, returns RMSEs as dictionary values. Else just returns as a tupel


    Given a model and a data loader for a validation set, find the RMSE of the model across all datapoints on the validation set across the three components and their sum (total, rho, Q, P)
    '''

    preds_rho = []
    preds_Q = []
    preds_P = []
    labels_rho = []
    labels_Q = []
    labels_P = []
 
    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            preds, labels = batch_to_output(model, batch, device, return_labels=True, n_step=n_step, return_multiple_steps=return_multiple_steps)

            rho_pred, Q_pred, P_pred = [tnsr.cpu() for tnsr in preds]
            rho_label, Q_label, P_label = [tnsr.cpu() for tnsr in labels]

            preds_rho.append(rho_pred)
            preds_Q.append(Q_pred)
            preds_P.append(P_pred)
            labels_rho.append(rho_label)
            labels_Q.append(Q_label)
            labels_P.append(P_label)

    # Return model to training mode
    model.train()

    # Concat all data points along batch dimension
    preds_rho = torch.cat(preds_rho, dim=0)
    preds_Q = torch.cat(preds_Q, dim=0)
    preds_P = torch.cat(preds_P, dim=0)
    labels_rho = torch.cat(labels_rho, dim=0)
    labels_Q = torch.cat(labels_Q, dim=0)
    labels_P = torch.cat(labels_P, dim=0)

    if preds_rho.shape != labels_rho.shape or preds_Q.shape != labels_Q.shape or preds_P.shape != labels_P.shape:
        raise Exception("calc_val_statistics: shape of labels and preds not the same!")


    RMSE_rho = torch_RMSE(preds_rho, labels_rho)
    RMSE_Q = torch_RMSE(preds_Q, labels_Q)
    RMSE_P = torch_RMSE(preds_P, labels_P)

    total_RMSE = RMSE_rho + RMSE_Q + RMSE_P
    
    if not return_as_dict:
        return total_RMSE, RMSE_rho, RMSE_Q, RMSE_P
    else:
        return {'total RMSE':total_RMSE, 'RMSE rho':RMSE_rho, 'RMSE Q':RMSE_Q, 'RMSE P':RMSE_P}
