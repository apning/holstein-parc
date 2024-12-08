import torch
import torch.nn as nn

# A simple implementation of plain dropout with same interface as the PyTorch class nn.Dropout
# compatible with complex dtypes and should be more theoretically sound (does not apply to real and imag separately)
class Robust_Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = torch.bernoulli(torch.full_like(input, 1 - self.p, dtype=torch.float32)).to(input.device)
            if self.inplace:
                input.mul_(mask)
                input.div_(1 - self.p)
                return input
            else:
                return input * mask / (1 - self.p)
        return input


# simple implementation of dropout2d mimicing the function and interface of Pytorch's nn.Dropout2d
# only difference is that is should work with complex dtypes. It does not treat real and imag components separately so is theoretically sound
class Robust_Dropout2d(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            if input.dim() != 4:
                raise ValueError(f"Expected 4D input (got {input.dim()}D input). Input shape: {input.shape}")

            # Create a mask for the channels
            mask_shape = (input.size(0), input.size(1), 1, 1)
            mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, dtype=torch.float32)).to(input.device)

            if self.inplace:
                input.mul_(mask)
                input.div_(1 - self.p)
                return input
            else:
                return input * mask / (1 - self.p)
        return input


class Robust_Dropout1d(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            if input.dim() not in [2, 3]:
                raise ValueError("Expected 2D or 3D input (got {}D input)".format(input.dim()))

            # Create a mask for the channels
            if input.dim() == 2:
                mask_shape = (input.size(0), 1)
            else:  # input.dim() == 3
                mask_shape = (input.size(0), input.size(1), 1)

            mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, dtype=torch.float32)).to(input.device)

            if self.inplace:
                input.mul_(mask)
                input.div_(1 - self.p)
                return input
            else:
                return input * mask / (1 - self.p)
        return input


def hamiltonian(Q, g, t):

    '''
    Q: real-valued tensor of shape [Batch, L]
    g,t: scalars
    '''

    if len(Q.shape) != 2:
        raise Exception(f"hamiltonian: Q was not 2 dimensional! Its shape was: {Q.shape}")
    if Q.is_complex:
        raise Exception(f"hamiltonian: Q cannot be complex! Its dtype was: {Q.dtype}")

    dtype = Q.dtype
    L = Q.shape[1]
    device = Q.device
    
    # Generate t (hopping amplitude) Matrix H_t
    H_t = np.identity(L) * -t
    H_t_up = np.roll(H_t, shift=1, axis=0)
    H_t_down = np.roll(H_t, shift=-1, axis=0)
    H_t = H_t_up + H_t_down
    H_t = torch.tensor(H_t, dtype=dtype, device=device).reshape(1, L, L) #reshape for broadcasting

    H = -g * torch.diag_embed(Q) + H_t #new shape [batch, L, L]
    return H


" Turns any nn.Module activation function complex "
class ComplexActivation(nn.Module):
    def __init__(self, act_func=nn.ReLU()):
        super().__init__()
        self.act_func = act_func
        
    def forward(self, input):
        # Apply the act func to both real and imaginary parts separately
        real_part = self.act_func(input.real)
        imag_part = self.act_func(input.imag)
        return torch.complex(real_part, imag_part)
