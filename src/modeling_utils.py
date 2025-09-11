# Standard library imports
from typing import Any

# Third-party imports
import torch
import torch.nn as nn


class Conv2DLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Any | None = None,
        dtype: Any | None = None,
    ):
        """
        A wrapper around layernorm which automatically transposes the input from shape [..., C, H, W] to [..., W, H, C] before layernorm (and then returns the shape afterwards) so that layernorm can be applied channel-wise.

        Args:
            normalized_shape (int|list[int]|torch.Size): Passed directly to 'normalized_shape' argument in nn.LayerNorm
                If list or list-like, must have only 1-dimension as transpose of dimensions only makes sense for channel-wise layernorm.
            eps (float): Passed directly to 'eps' argument in nn.LayerNorm
            elementwise_affine (bool): Passed directly to 'elementwise_affine' argument in nn.LayerNorm
            bias (bool): Passed directly to 'bias' argument in nn.LayerNorm
            device (Any | None): Passed directly to 'device' argument in nn.LayerNorm
            dtype (Any | None): Passed directly to 'dtype' argument in nn.LayerNorm
        """
        super().__init__()

        if type(normalized_shape) != int and len(normalized_shape) != 1:
            raise ValueError(
                f"{self.__class__.__name__}: normalized_shape must be an int or a list-like with only 1 dimension! But got: {normalized_shape}"
            )

        self.ln = nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"{self.__class__.__name__}: Expected input tensor with at least 3 dimensions (..., channels, height, width), but got {x.dim()} dimensions."
            )

        x = x.transpose(-1, -3)  # Change shape from [..., C, H, W] to [..., W, H, C]

        x = self.ln(x)

        x = x.transpose(-1, -3)  # Change shape back to [..., C, H, W]

        return x
