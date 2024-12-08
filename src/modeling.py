import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
import datetime
import time

from .modeling_utils import hamiltonian, ComplexActivation, Robust_Dropout, Robust_Dropout1d, Robust_Dropout2d


class PARC_Conv(nn.Module):
    def __init__(self, input_channels=2, output_channels=2, n_features=128, dropout=0.2, conv_dimension=2, dtype=torch.float32, extra_dropout=False, padding_mode='circular'):

        '''
        Modeled after: https://github.com/stephenbaek/parc/blob/main/parc/model.py
        Additional changes may have been made, however. Not intended as an exact replica

        Preserves the height and width of input
        
        '''

        super().__init__()

        if conv_dimension == 2:
            self.conv_type = nn.Conv2d
        elif conv_dimension == 1:
            self.conv_type = nn.Conv1d
        else:
            raise Exception(f"{self.__class__.__name__}: {conv_dimension} is not an acceptable argument. Please use either 1 or 2")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_features = n_features
        self.dtype = dtype
        self.is_complex = dtype.is_complex
        self.extra_dropout = extra_dropout

        self.act_func = nn.ReLU()
        if self.is_complex:
            self.act_func = ComplexActivation(self.act_func)

        self.layer_1_0 = self.conv_type(input_channels, n_features//2, 3, padding='same', dtype=dtype, padding_mode=padding_mode)
        self.layer_1_1 = self.conv_type(n_features//2, n_features//2, 3, padding='same', dtype=dtype, padding_mode=padding_mode)
        self.layer_1_2 = self.conv_type(n_features//2, n_features//2, 3, padding='same', dtype=dtype, padding_mode=padding_mode)

        self.layer_2_0 = self.conv_type(n_features//2, n_features, 3, padding='same', dtype=dtype, padding_mode=padding_mode)
        self.layer_2_1 = self.conv_type(n_features, n_features, 3, padding='same', dtype=dtype, padding_mode=padding_mode)
        self.layer_2_2 = self.conv_type(n_features, n_features, 3, padding='same', dtype=dtype, padding_mode=padding_mode)

        self.layer_3_0 = self.conv_type(n_features, n_features, 7, padding='same', dtype=dtype, padding_mode=padding_mode)
        self.layer_3_1 = self.conv_type(n_features, n_features//2, 1, padding='same', dtype=dtype, padding_mode=padding_mode)
        self.layer_3_2 = self.conv_type(n_features//2, n_features//4, 1, padding='same', dtype=dtype, padding_mode=padding_mode)

        # self.dropout_layer = Robust_Dropout(p=dropout)

        if conv_dimension == 1:
            self.dropout_layer = Robust_Dropout1d(p=dropout)
        elif conv_dimension == 2:
            self.dropout_layer = Robust_Dropout2d(p=dropout)

        self.layer_out = self.conv_type(n_features//4, output_channels, 3, padding='same', dtype=dtype, padding_mode=padding_mode)

        self.out_act_func = nn.Tanh()
        if self.is_complex:
            self.out_act_func = ComplexActivation(self.out_act_func)

    def forward(self, x_0):

        if x_0.dtype != self.dtype:
            raise Exception(f"{self.__class__.__name__}: dtype mismatch! x_0 dtype: {x_0.dtype}\tConvolution dtype: {self.dtype}")

        ## BLOCK 1
    
        x_1_0 = self.layer_1_0(x_0)
        x_1_0 = self.act_func(x_1_0)

        x_1_1 = self.layer_1_1(x_1_0)
        x_1_1 = self.act_func(x_1_1)

        x_1_2 = self.layer_1_2(x_1_1)
        x_1_add = self.act_func(x_1_0 + x_1_2)

        ## BLOCK 2

        x_2_0 = self.layer_2_0(x_1_add)
        x_2_0 = self.act_func(x_2_0)

        x_2_1 = self.layer_2_1(x_2_0)
        x_2_1 = self.act_func(x_2_1)

        x_2_2 = self.layer_2_2(x_2_1)
        x_2_add = self.act_func(x_2_0 + x_2_2)

        # In PARC the integrator had an extra dropout here
        if self.extra_dropout:
            x_2_add = self.dropout_layer(x_2_add)

        ## BLOCK 3

        x_3_0 = self.layer_3_0(x_2_add)
        x_3_0 = self.act_func(x_3_0)

        x_3_1 = self.layer_3_1(x_3_0)
        x_3_1 = self.act_func(x_3_1)

        x_3_2 = self.layer_3_2(x_3_1)
        x_3_2 = self.act_func(x_3_2)

 
        x_3_2 = self.dropout_layer(x_3_2)

        ## FINALE
        x_out = self.layer_out(x_3_2)
        x_out = self.out_act_func(x_out)

        return x_out


# pass in rho, Q, and P and perhaps some coefficients and get out d_rho, d_Q, and d_P
# "separate" because we derive rho, Q, and P via different CNNs
class DeriveSeparate(nn.Module):
    def __init__(self, n_features=128, calc_hamil=False):
        super().__init__()

        '''
        calc_hamil (bool): Whether or not to analytically calculate the Hamiltonian of Q before passing it into convolution. If false, we just pass Q directly after embedding it in the diagonal of a 2d matrix
        
        '''

        if calc_hamil:
            raise Exception("DeriveSeparate: calc_hamil not implemented yet. Cannot specify 'True'")

        self.calc_hamil = calc_hamil

        self.conv_rho = PARC_Conv(input_channels=2, output_channels=1, n_features=n_features, conv_dimension=2, dtype=torch.complex64, extra_dropout=False)
        self.conv_Q = PARC_Conv(input_channels=1, output_channels=1, n_features=n_features, conv_dimension=1, dtype=torch.float32, extra_dropout=False)
        self.conv_P = PARC_Conv(input_channels=2, output_channels=1, n_features=n_features, conv_dimension=1, dtype=torch.float32, extra_dropout=False)
       

    def derive_rho(self, rho, Q, L):
        rho_reshaped = rho.reshape(-1, 1, L, L)
        Q_embedded = torch.diag_embed(Q).to(torch.complex64).reshape(-1, 1, L, L)
        rho_input = torch.concat([rho_reshaped, Q_embedded], dim=1)
        d_rho = self.conv_rho(rho_input).reshape(-1, L, L)
        return d_rho

    def derive_Q(self, P, L):
        Q_input = P.reshape(-1, 1, L)
        d_Q = self.conv_Q(Q_input).reshape(-1, L)
        return d_Q

    def derive_P(self, rho, Q, L):
        rho_diag = torch.diagonal(rho, dim1=-2, dim2=-1).real.reshape(-1, 1, L)
        Q_reshaped = Q.reshape(-1, 1, L)
        P_input = torch.concat([rho_diag, Q_reshaped], dim=1)
        d_P = self.conv_P(P_input).reshape(-1, L)
        return d_P

    def forward(self, rho, Q, P):

        '''
        rho: A 3d complex64 tensor of shape [batch, L, L]
        Q, P: A 2d float32 tensor of shape [batch, L]

        The outputs of d_rho, d_Q, and d_P will be of same shape
        '''

        if rho.dtype != torch.complex64:
            raise Exception(f"DerivSeparate: dtype for rho was {rho.dtype}. It MUST be torch.complex64")
        if Q.dtype != torch.float32:
            raise Exception(f"DerivSeparate: dtype for Q was {Q.dtype}. It MUST be torch.float32")
        if P.dtype != torch.float32:
            raise Exception(f"DerivSeparate: dtype for P was {P.dtype}. It MUST be torch.float32")

        if len(rho.shape) != 3:
            raise Exception(f"DerivSeparate: rho was not 3 dimensional! Its shape was: {rho.shape}")
        if len(Q.shape) != 2:
            raise Exception(f"DerivSeparate: Q was not 2 dimensional! Its shape was: {Q.shape}")
        if len(P.shape) != 2:
            raise Exception(f"DerivSeparate: P was not 2 dimensional! Its shape was: {P.shape}")

        if not (rho.shape[1] == rho.shape[2] == Q.shape[1] == P.shape[1] and rho.shape[0] == rho.shape[0] == Q.shape[0] == P.shape[0]):
            raise Exception(f"DerivSeparate: Inconsistency in shape detected!\n\trho shape: {rho.shape}\n\tQ shape: {Q.shape}\n\tP shape: {P.shape}")

        "As a rule, do not directly modify the values of any inputs (eg. rho, Q, P, coefficients), as there are several different parts below all dependent on them. Instead, choose a new name for the results of any transformations "

        L = Q.shape[1]

        d_rho = self.derive_rho(rho, Q, L)
        d_Q = self.derive_Q(P, L)
        d_P = self.derive_P(rho, Q, L)

        return d_rho, d_Q, d_P
        

# pass in d_rho, d_Q, and d_P and perhaps some coefficients and get out delta_rho, delta_Q, and delta_P
# "separate" because we integrate d_rho, d_Q, and d_P via different CNNs
class IntSeparate(nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
        
        self.conv_rho = PARC_Conv(input_channels=1, output_channels=1, n_features=n_features, conv_dimension=2, dtype=torch.complex64, extra_dropout=True)
        self.conv_Q = PARC_Conv(input_channels=1, output_channels=1, n_features=n_features, conv_dimension=1, dtype=torch.float32, extra_dropout=True)
        self.conv_P = PARC_Conv(input_channels=1, output_channels=1, n_features=n_features, conv_dimension=1, dtype=torch.float32, extra_dropout=True)
            

    def forward(self, d_rho, d_Q, d_P):

        '''
        d_rho: A 3d complex64 tensor of shape [batch, L, L]
        d_Q, d_P: A 2d float32 tensor of shape [batch, L]

        The outputs of delta_rho, delta_Q, and delta_P will be of same shape
        '''

        if d_rho.dtype != torch.complex64:
            raise Exception(f"IntSeparate: dtype for d_rho was {d_rho.dtype}. It MUST be torch.complex64")
        if d_Q.dtype != torch.float32:
            raise Exception(f"IntSeparate: dtype for d_Q was {d_Q.dtype}. It MUST be torch.float32")
        if d_P.dtype != torch.float32:
            raise Exception(f"IntSeparate: dtype for d_P was {d_P.dtype}. It MUST be torch.float32")

        if len(d_rho.shape) != 3:
            raise Exception(f"IntSeparate: d_rho was not 3 dimensional! Its shape was: {d_rho.shape}")
        if len(d_Q.shape) != 2:
            raise Exception(f"IntSeparate: d_Q was not 2 dimensional! Its shape was: {d_Q.shape}")
        if len(d_P.shape) != 2:
            raise Exception(f"IntSeparate: d_P was not 2 dimensional! Its shape was: {d_P.shape}")

        if not (d_rho.shape[1] == d_rho.shape[2] == d_Q.shape[1] == d_P.shape[1] and d_rho.shape[0] == d_rho.shape[0] == d_Q.shape[0] == d_P.shape[0]):
            raise Exception(f"IntSeparate: Inconsistency in shape detected!\n\td_rho shape: {d_rho.shape}\n\td_Q shape: {d_Q.shape}\n\td_P shape: {d_P.shape}")

        "As a rule, do not directly modify the values of any inputs (eg. d_rho, d_Q, d_P, coefficients), as there are several different parts below all dependent on them. Instead, choose a new name for the results of any transformations "

        L = d_Q.shape[1]

        d_rho_reshaped = d_rho.reshape(-1, 1, L, L)
        delta_rho = self.conv_rho(d_rho_reshaped).reshape(-1, L, L)

        d_Q_reshaped = d_Q.reshape(-1, 1, L)
        delta_Q = self.conv_Q(d_Q_reshaped).reshape(-1, L)

        d_P_reshaped = d_P.reshape(-1, 1, L)
        delta_P = self.conv_P(d_P_reshaped).reshape(-1, L)

        return delta_rho, delta_Q, delta_P


## returns the next values of rho, Q, and P
class HolsteinStepSeparate(nn.Module):
    def __init__(self, n_features=128):
        super().__init__()
    
        self.derive = DeriveSeparate(n_features=n_features)
        self.integrate = IntSeparate(n_features=n_features)
    
    def step(self, rho, Q, P, return_intermediaries=False):

        d_rho, d_Q, d_P = self.derive(rho, Q, P)
        delta_rho, delta_Q, delta_P = self.integrate(d_rho, d_Q, d_P)


        if delta_rho.shape != rho.shape or delta_Q.shape != Q.shape or delta_P.shape != P.shape:
            raise Exception(f"HolsteinStepSeparate: Shape inconsistency between original rho/Q/P and delta detected! Shapes:\n\trho:\t{rho.shape}delta rho:\t{delta_rho.shape}\n\tQ:\t{Q.shape}delta Q:\t{delta_Q.shape}\n\tP:\t{P.shape}delta P:\t{delta_P.shape}")
        if delta_rho.dtype != rho.dtype or delta_Q.dtype != Q.dtype or delta_P.dtype != P.dtype:
            raise Exception(f"HolsteinStepSeparate: dtype inconsistency between original rho/Q/P and delta detected! dtypes:\n\trho:\t{rho.dtype}delta rho:\t{delta_rho.dtype}\n\tQ:\t{Q.dtype}delta Q:\t{delta_Q.dtype}\n\tP:\t{P.dtype}delta P:\t{delta_P.dtype}")

        rho = rho + delta_rho
        Q = Q + delta_Q
        P = P + delta_P

        if not return_intermediaries:
            return rho, Q, P
        else:
            return rho, Q, P, (d_rho, d_Q, d_P), (delta_rho, delta_Q, delta_P)

    def forward(self, rho, Q, P, return_intermediaries=False, n_step=1, return_multiple_steps=False):
        '''
        return_multiple_steps (bool): Add a time step dimension to the output, returning multiple consecutive predictions. Corresponds to multi_step_labels in HolsteinDataset
        
        '''

        if return_intermediaries and (n_step != 1 or return_multiple_steps):
            raise Exception(f"HolsteinStepSeparate: Currently enabling both return_intermediaries and either setting n_step > 1 or return_multiple_steps is not implemented")

        if n_step < 1:
            raise Exception(f"HolsteinStepSeparate: n_step cannot be less than 1. Received: {n_step}")

        if n_step == 1 and not return_multiple_steps:
            return self.step(rho, Q, P, return_intermediaries=return_intermediaries)
        
        if return_multiple_steps:
            steps_sequential = []

        for _ in range(n_step):
            rho, Q, P = self.step(rho, Q, P)

            if return_multiple_steps:
                steps_sequential.append((rho, Q, P))
        
        if not return_multiple_steps:
            return rho, Q, P

        # process return multiple steps
        all_rho, all_Q, all_P = tuple(zip(*steps_sequential))

        rho_tensor = torch.stack(all_rho, dim=1) #shape [batch, n_step, L, L]
        Q_tensor = torch.stack(all_Q, dim=1) #shape [batch, n_step, L]
        P_tensor = torch.stack(all_P, dim=1) #shape [batch, n_step, L]

        return rho_tensor, Q_tensor, P_tensor
        






    




        


        



    