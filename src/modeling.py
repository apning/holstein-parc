from collections.abc import Sequence
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.data_utils import unpickle_data
from src.modeling_utils import Conv2DLayerNorm


class CNNnd_Block(nn.Module):
    """
    CNNnd_Block implements a ResNet-v2 style pre-activation residual block
    with support for 1D or 2D convolutions, optional dropout, and SkipInit-like residual scaling.
    Attributes:
    channels : int
        Number of channels (input, output, and intermediate)
    conv_dim : int
        Dimensionality of the convolution (1 for 1D, 2 for 2D). Defaults to 2.
    dropout_p : float
        Probability of an element to be zeroed during dropout. Defaults to 0.0 (no dropout).
    kernel_size : int or tuple[int]
        Size of the convolutional kernel. Defaults to 3.
    use_residual_scalar : bool
        Whether to use a learnable residual scalar to scale the output. Defaults to True.
    act_func : nn.Module
        Activation function to use. Defaults to `nn.ReLU()`.
    dtype : torch.dtype
        Data type for the tensors. Defaults to `torch.float32`.
    use_layernorm : bool
        Whether to apply Layer Normalization after each convolution. Defaults to False.
    Methods:
    __init__(self, inputs, outputs, intermediate_channels=None, conv_dim=2, dropout_p=0.0, kernel_size=3,
             use_residual_scalar=True, act_func=nn.ReLU(), dtype=torch.float32):
        Initializes the CNNnd_Block with the specified parameters.
    forward(self, x):
        Performs the forward pass of the CNN block.
    """

    def __init__(
        self,
        channels: int,
        conv_dim: int = 2,
        dropout_p: float = 0.0,
        kernel_size: int | tuple[int] = 3,
        use_residual_scalar: bool = True,
        act_func: nn.Module = nn.ReLU(),
        dtype: torch.dtype = torch.float32,
        use_layernorm: bool = False,
    ):
        super().__init__()

        """ Argument Checks """
        if conv_dim not in [1, 2]:
            raise ValueError(
                f"{self.__class__.__name__}: '{conv_dim}' is not a valid argument for conv_dim. Please use either 1 or 2"
            )

        """ Saving Args to Self """

        self.channels = channels
        self.conv_dim = conv_dim
        self.dropout_p = dropout_p
        self.kernel_size = kernel_size
        self.use_residual_scalar = use_residual_scalar
        self.act_func = act_func
        self.dtype = dtype
        self.use_layernorm = use_layernorm

        """ Creating Components """

        if self.conv_dim == 2:
            self.conv_type = nn.Conv2d
            if self.dropout_p:
                self.dropout = nn.Dropout2d(p=self.dropout_p)
        elif self.conv_dim == 1:
            self.conv_type = nn.Conv1d
            if self.dropout_p:
                self.dropout = nn.Dropout1d(p=self.dropout_p)

        self.conv1 = self.conv_type(
            self.channels,
            self.channels,
            self.kernel_size,
            padding="same",
            padding_mode="circular",
            dtype=self.dtype,
        )
        self.conv2 = self.conv_type(
            self.channels,
            self.channels,
            self.kernel_size,
            padding="same",
            padding_mode="circular",
            dtype=self.dtype,
        )

        if self.use_residual_scalar:
            self.residual_scalar = nn.Parameter(torch.tensor(0, dtype=dtype))

        if self.use_layernorm:
            self.ln1 = Conv2DLayerNorm(normalized_shape=self.channels, dtype=self.dtype)
            self.ln2 = Conv2DLayerNorm(normalized_shape=self.channels, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor. For 2D convolutions, the input must have shape
            [batch, channels, height, width] or [channels, height, width].
            For 1D convolutions, the input must have shape
            [batch, channels, length] or [channels, length].
        Returns:
        --------
        torch.Tensor
            Output tensor after applying the sequence of operations including
            activation functions, convolutions, optional dropout, and optional
            residual scaling.
        Raises:
        -------
        ValueError
            If the input tensor does not have the expected dimensions for the
            specified convolutional dimension (`conv_dim`).
        """

        if self.conv_dim == 2:
            if x.dim() == 4:
                pass
            elif x.dim() == 3:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"{self.__class__.__name__}: Input must have shape [batch, channels, height, width] or [channels, height, width]; got tensor with shape {tuple(x.shape)}"
                )
        if self.conv_dim == 1:
            if x.dim() == 3:
                pass
            elif x.dim() == 2:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"{self.__class__.__name__}: Input must have shape [batch, channels, length] or [channels, length]; got tensor with shape {tuple(x.shape)}"
                )

        if self.use_layernorm:
            x = self.ln1(x)

        x = self.act_func(x)

        x = self.conv1(x)

        if self.use_layernorm:
            x = self.ln2(x)

        x = self.act_func(x)

        if self.dropout_p:
            x = self.dropout(x)

        x = self.conv2(x)

        if self.use_residual_scalar:
            x = self.residual_scalar * x

        return x


class CNNnd(nn.Module):
    """
    CNNnd is a configurable convolutional neural network (CNN) module that supports both 1D and 2D convolutions.
    It is designed to allow flexible configuration of the number of layers, channels, activation functions,
    and initialization methods.
    Attributes:
        SUPPORTED_INITS (set): A set of supported initialization methods for the network weights.
        SUPPORTED_ACT_FUNCS (set): A set of supported activation functions for the network.
        SUPPORTED_OUT_ACT_FUNCS (set): A set of supported output activation functions for the network.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels in the intermediate layers.
        n_blocks (int): Number of residual blocks in the network. Must be >= 1.
        conv_dim (int, optional): Dimensionality of the convolution (1 for 1D, 2 for 2D). Defaults to 2.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        kernel_size (int or tuple[int], optional): Size of the convolutional kernel. Defaults to 3.
        use_residual_scalar (bool, optional): Whether to use a residual scalar in the residual blocks. Defaults to True.
        act_func (nn.Module, optional): Activation function to use. Defaults to nn.ReLU().
        out_act_func (nn.Module or None, optional): Output activation function. Defaults to None.
        init_method (str, optional): Initialization method for the weights. Must be one of the supported methods. Defaults to "kaiming_uniform".
        zero_initialize_output (bool, optional): Whether to initialize the output convolution weights to zero. Defaults to True.
        dtype (torch.dtype, optional): Data type for the model parameters. Defaults to torch.float32.
        use_layernorm (bool): Whether to use Layer Normalization in the residual blocks. Defaults to False.
        ValueError: If `n_blocks` is less than 1.
        ValueError: If `init_method` is not in the supported initialization methods.
        ValueError: If `conv_dim` is not 1 or 2.
        ValueError: If `act_func` is not in the supported activation functions.
        ValueError: If `out_act_func` is not in the supported output activation functions.
    Methods:
        init_weights():
            Initializes the weights of the network based on the specified initialization method and activation function.
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the network. Applies input convolution, activation function, residual blocks,
            and output convolution. Optionally applies an output activation function.
                x (torch.Tensor): Input tensor. For 2D convolutions, the input must have shape
                    [batch, channels, height, width] or [channels, height, width].
                torch.Tensor: Output tensor after processing through the network.
                ValueError: If the input tensor does not have the expected dimensions for the specified convolutional dimension (`conv_dim`).
    """

    SUPPORTED_DIMS = {1, 2}
    SUPPORTED_INITS = {
        "kaiming_uniform",
        "kaiming_normal",
        "xavier_uniform",
        "xavier_normal",
        None,
    }
    SUPPORTED_ACT_FUNCS = {nn.ReLU, nn.Tanh}
    SUPPORTED_OUT_ACT_FUNCS = {type(None)}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_blocks: int,
        conv_dim: int = 2,
        dropout_p: float = 0.0,
        kernel_size: int | tuple[int] = 3,
        use_residual_scalar: bool = True,
        act_func: nn.Module = nn.ReLU(),
        out_act_func: nn.Module | None = None,
        init_method: str = "kaiming_uniform",
        zero_initialize_output: bool = True,
        dtype: torch.dtype = torch.float32,
        use_layernorm: bool = False,
    ):
        super().__init__()

        """ Argument Checks """

        if n_blocks < 1:
            raise ValueError(f"{self.__class__.__name__}: n_blocks must be >= 1. Received: '{n_blocks}'")
        if init_method not in self.SUPPORTED_INITS:
            raise ValueError(
                f"{self.__class__.__name__}: init_method must be one of {self.SUPPORTED_INITS}, got '{init_method}', which is either invalid or currently not supported"
            )
        if conv_dim not in self.SUPPORTED_DIMS:
            raise ValueError(
                f"{self.__class__.__name__}: '{conv_dim}' is not a valid argument for conv_dim. Please use either 1 or 2"
            )
        if type(act_func) not in self.SUPPORTED_ACT_FUNCS:
            raise ValueError(
                f"{self.__class__.__name__}: {act_func} is invalid or not supported. To add support a new activation function, implement an initialization scheme for it."
            )
        if type(out_act_func) not in self.SUPPORTED_OUT_ACT_FUNCS:
            raise ValueError(
                f"{self.__class__.__name__}: {out_act_func} is invalid or not supported. To add support a new out activation function, implement an initialization scheme for it."
            )

        """ Saving Args to Self """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks
        self.conv_dim = conv_dim
        self.dropout_p = dropout_p
        self.kernel_size = kernel_size
        self.use_residual_scalar = use_residual_scalar
        self.act_func = act_func
        self.out_act_func = out_act_func
        self.init_method = init_method
        self.zero_initialize_output = zero_initialize_output
        self.dtype = dtype
        self.use_layernorm = use_layernorm

        """ Creating Components """

        if self.conv_dim == 2:
            self.conv_type = nn.Conv2d
        elif self.conv_dim == 1:
            self.conv_type = nn.Conv1d

        self.input_conv = self.conv_type(
            self.in_channels,
            self.hidden_channels,
            self.kernel_size,
            padding="same",
            padding_mode="circular",
            dtype=self.dtype,
        )

        self.blocks = nn.ModuleList(
            [
                CNNnd_Block(
                    channels=self.hidden_channels,
                    conv_dim=self.conv_dim,
                    dropout_p=self.dropout_p,
                    kernel_size=self.kernel_size,
                    use_residual_scalar=self.use_residual_scalar,
                    act_func=self.act_func,
                    dtype=self.dtype,
                    use_layernorm=self.use_layernorm,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.output_conv = self.conv_type(self.hidden_channels, self.out_channels, 1, dtype=self.dtype)

        # Initialize weights
        if self.init_method is not None:
            self.apply(self.init_weights)

        # Initialize output conv to zero if this option is specified
        if self.zero_initialize_output:
            nn.init.zeros_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.zeros_(self.output_conv.bias)

    def init_weights(self, m: nn.Module) -> None:
        """
        Initializes the weights of the neural network layers based on the specified
        activation function and initialization method.
        This method applies initialization to all modules of the specified convolution
        type (`self.conv_type`) within the model. The initialization method and
        nonlinearity are determined by the attributes `self.init_method` and
        `self.act_func`, respectively.
        Args:
            m (nn.Module): The module to initialize. This argument is not directly
                           used in the method but is required for compatibility with
                           PyTorch's `apply` method.
        Raises:
            RuntimeError: If the activation function (`self.act_func`) is not supported
                          or cannot be matched to a string representation.
            ValueError: If the initialization method (`self.init_method`) is not
                        supported for the given nonlinearity.
        """

        if type(self.act_func) is nn.ReLU:
            nonlinearity = "relu"
        elif type(self.act_func) is nn.Tanh:
            nonlinearity = "tanh"
        else:  # Should be impossible due to earlier validation
            raise RuntimeError(
                f"{self.__class__.__name__}.init_weights(): Could not match the activation function {self.act_func} to a string. This could indicate that this activation function is somehow in the supported set of activation functions, but complete support has yet to be implemented."
            )

        if isinstance(m, self.conv_type):
            if nonlinearity == "relu":
                if self.init_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif self.init_method == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    raise ValueError(
                        f"{self.__class__.__name__}.init_weights(): The initialization method '{self.init_method}' is not a supported initialiation method for the nonlinearity '{nonlinearity}'"
                    )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif nonlinearity == "tanh":
                if self.init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                elif self.init_method == "xavier_normal":
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                else:
                    raise ValueError(
                        f"{self.__class__.__name__}.init_weights(): The initialization method '{self.init_method}' is not a supported initialiation method for the nonlinearity '{nonlinearity}'"
                    )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:  # Should be impossible due to earlier validation
                raise RuntimeError(
                    f"{self.__class__.__name__}.init_weights(): Could not match the nonlinearity '{nonlinearity}' to a supported nonlinearity. This likely indicates that support for '{nonlinearity}' has been partially implemented but is not complete"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor. For 2D convolutions, the input must have
                shape [batch, channels, height, width] or [channels, height, width].
                For 1D convolutions, the input must have shape [batch, channels, length]
                or [channels, length].
        Returns:
            torch.Tensor: Output tensor after applying the input convolution, activation
            function, residual blocks, and output convolution. If `out_act_func` is
            specified, it is applied to the output tensor.
        Raises:
            ValueError: If the input tensor does not have the expected dimensions for
            the specified convolutional dimension (`conv_dim`).
        """

        if self.conv_dim == 2:
            if x.dim() == 4:
                pass
            elif x.dim() == 3:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"{self.__class__.__name__}: Input must have shape [batch, channels, height, width] or [channels, height, width]; got tensor with shape {tuple(x.shape)}"
                )
        if self.conv_dim == 1:
            if x.dim() == 3:
                pass
            elif x.dim() == 2:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"{self.__class__.__name__}: Input must have shape [batch, channels, length] or [channels, length]; got tensor with shape {tuple(x.shape)}"
                )

        x = self.input_conv(x)
        x = self.act_func(x)

        for block in self.blocks:
            x = x + block(x)

        x = self.act_func(x)
        x = self.output_conv(x)

        if self.out_act_func is not None:
            x = self.out_act_func(x)

        return x


class HolsteinStepCombined(nn.Module):
    def __init__(
        self,
        channels: int,
        n_blocks: int,
        dropout_p: float = 0.0,
        kernel_size: int | tuple[int] = 3,
        use_residual_scalar: bool = True,
        act_func: nn.Module = nn.ReLU(),
        init_method: str = "kaiming_uniform",
        zero_initialize_output: bool = True,
        dtype: torch.dtype = torch.float32,
        use_layernorm: bool = False,
        in_scale: Sequence[float | int, float | int, float | int] | None = None,
        out_scale: Sequence[float | int, float | int, float | int] | None = None,
    ):
        super().__init__()

        """ Arg Checking """

        if in_scale is not None:
            if len(in_scale) != 3:
                raise ValueError(
                    f"{self.__class__.__name__}: in_scale must be a Sequence of 3 scalars, got {len(in_scale)}"
                )
            for scalar in in_scale:
                if not isinstance(scalar, (int, float)):
                    raise ValueError(
                        f"{self.__class__.__name__}: in_scale must be a Sequence of 3 scalars. Got: {scalar}"
                    )
        if out_scale is not None:
            if len(out_scale) != 3:
                raise ValueError(
                    f"{self.__class__.__name__}: out_scale must be a Sequence of 3 scalars, got {len(out_scale)}"
                )
            for scalar in out_scale:
                if not isinstance(scalar, (int, float)):
                    raise ValueError(
                        f"{self.__class__.__name__}: out_scale must be a Sequence of 3 scalars. Got: {scalar}"
                    )

        """ Save Args to Self """

        self.channels = channels
        self.n_blocks = n_blocks
        self.dropout_p = dropout_p
        self.kernel_size = kernel_size
        self.use_residual_scalar = use_residual_scalar
        self.act_func = act_func
        self.init_method = init_method
        self.zero_initialize_output = zero_initialize_output
        self.dtype = dtype
        self.use_layernorm = use_layernorm
        self.in_scale = in_scale
        self.out_scale = out_scale

        """ Create Components """

        self.CNN = CNNnd(
            in_channels=4,
            out_channels=4,
            hidden_channels=self.channels,
            n_blocks=self.n_blocks,
            conv_dim=2,
            dropout_p=self.dropout_p,
            kernel_size=self.kernel_size,
            use_residual_scalar=self.use_residual_scalar,
            act_func=self.act_func,
            out_act_func=None,
            init_method=self.init_method,
            zero_initialize_output=self.zero_initialize_output,
            dtype=self.dtype,
            use_layernorm=self.use_layernorm,
        )

        ## Create buffers for input/output scaling if applicable
        if self.in_scale is not None:
            self.register_buffer("in_scale_rho", torch.tensor(self.in_scale[0], dtype=self.dtype))
            self.register_buffer("in_scale_Q", torch.tensor(self.in_scale[1], dtype=self.dtype))
            self.register_buffer("in_scale_P", torch.tensor(self.in_scale[2], dtype=self.dtype))
        if self.out_scale is not None:
            self.register_buffer("out_scale_rho", torch.tensor(self.out_scale[0], dtype=self.dtype))
            self.register_buffer("out_scale_Q", torch.tensor(self.out_scale[1], dtype=self.dtype))
            self.register_buffer("out_scale_P", torch.tensor(self.out_scale[2], dtype=self.dtype))

    def forward(
        self, rho: torch.Tensor, Q: torch.Tensor, P: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward function specifically to handle holstein data

        Args:
            rho: torch.Tensor
                Complex-valued. Shape is [batch, L, L]
            Q: torch.Tensor
                Real-valued. Shape is [batch, L]
            P: torch.Tensor
                Real-valued. Shape is [batch, L]

        Returns:
            rho_pred: torch.Tensor
                Complex-valued. Shape is [batch, L, L]
            Q_pred: torch.Tensor
                Real-valued. Shape is [batch, L]
            P_pred: torch.Tensor
                Real-valued. Shape is [batch, L]
        """

        """ Shape Checks """
        L = rho.size(-1)
        batch_size = rho.size(0)

        assert rho.shape == (batch_size, L, L), (
            f"{self.__class__.__name__}: rho did not have shape [batch, L, L]. Got: {rho.shape}"
        )
        assert Q.shape == (batch_size, L), f"{self.__class__.__name__}: Q did not have shape [batch, L]. Got: {Q.shape}"
        assert P.shape == (batch_size, L), f"{self.__class__.__name__}: P did not have shape [batch, L]. Got: {P.shape}"

        """ Type Checks """
        assert torch.is_complex(rho), f"{self.__class__.__name__}: rho was not complex. Got: {rho.dtype}"
        assert not torch.is_complex(Q), f"{self.__class__.__name__}: Q was not real. Got: {Q.dtype}"
        assert not torch.is_complex(P), f"{self.__class__.__name__}: P was not real. Got: {P.dtype}"

        """ Scale Input """

        if self.in_scale is not None:
            rho = rho * self.in_scale_rho if self.in_scale_rho != 1 else rho
            Q = Q * self.in_scale_Q if self.in_scale_Q != 1 else Q
            P = P * self.in_scale_P if self.in_scale_P != 1 else P

        """ Build tensor for CNN input """

        ## Build the tensor for input into CNN
        # First, turn the complex rho into a real tensor of shape [batch, 2, L, L]
        rho = torch.permute(torch.view_as_real(rho), (0, -1, 1, 2)).contiguous()

        # Embed Q and P as the diagonals of 2d matrices. So each now has shape [batch, L, L]. Then add a channel dimension so they are [batch, 1, L, L]
        Q = torch.diag_embed(Q).unsqueeze(1)
        P = torch.diag_embed(P).unsqueeze(1)

        # Concatenate rho, Q, and P along the channel dimension. So now we have a tensor of shape [batch, 4, L, L]
        x = torch.cat((rho, Q, P), dim=1)

        """ Input into CNN """

        ## Now we can pass this through the CNN. The output should be same shape
        x = self.CNN(x)

        """ Deconstruct data back into original form """

        ## Now we split the output back into rho, Q, and P
        # First we work on rho
        rho_pred = x[:, :2, :, :]
        rho_pred = torch.view_as_complex(torch.permute(rho_pred, (0, 2, 3, 1)).contiguous())  # [batch, L, L]

        # Now we work on Q and P. We can do them together since they are the same shape
        QP_pred = x[:, 2:, :, :]
        QP_pred = torch.diagonal(QP_pred, dim1=-2, dim2=-1)  # [batch, 2, L]
        Q_pred = QP_pred[:, 0]  # [batch, L]
        P_pred = QP_pred[:, 1]  # [batch, L]

        """ Scale Output """

        if self.out_scale is not None:
            rho_pred = rho_pred * self.out_scale_rho if self.out_scale_rho != 1 else rho_pred
            Q_pred = Q_pred * self.out_scale_Q if self.out_scale_Q != 1 else Q_pred
            P_pred = P_pred * self.out_scale_P if self.out_scale_P != 1 else P_pred

        return rho_pred, Q_pred, P_pred


class HolsteinPARC(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        n_blocks: int = 4,
        dropout_p: float = 0.0,
        kernel_size: int | tuple[int] = 3,
        use_residual_scalar: bool = True,
        act_func: nn.Module = nn.ReLU(),
        init_method: str = "kaiming_uniform",
        zero_initialize_output: bool = True,
        dtype: torch.dtype = torch.float32,
        use_layernorm: bool = False,
        data_scalars: dict[str, float | int] | None = None,
        simple_cnn: bool = False,
    ):
        super().__init__()

        """ Save Args to Self """

        self.channels = channels
        self.n_blocks = n_blocks
        self.dropout_p = dropout_p
        self.kernel_size = kernel_size
        self.use_residual_scalar = use_residual_scalar
        self.act_func = act_func
        self.init_method = init_method
        self.zero_initialize_output = zero_initialize_output
        self.dtype = dtype
        self.use_layernorm = use_layernorm
        self.data_scalars = data_scalars
        self.simple_cnn = simple_cnn

        """ Create Components """

        if not self.simple_cnn:
            self.differentiator = HolsteinStepCombined(
                channels=self.channels,
                n_blocks=self.n_blocks,
                dropout_p=self.dropout_p,
                kernel_size=self.kernel_size,
                use_residual_scalar=self.use_residual_scalar,
                act_func=self.act_func,
                init_method=self.init_method,
                zero_initialize_output=False,
                dtype=self.dtype,
                use_layernorm=self.use_layernorm,
                in_scale=(1 / self.data_scalars["rho"], 1 / self.data_scalars["Q"], 1 / self.data_scalars["P"])
                if self.data_scalars is not None
                else None,
                out_scale=(self.data_scalars["drho"], self.data_scalars["dQ"], self.data_scalars["dP"])
                if self.data_scalars is not None
                else None,
            )

            self.integrator = HolsteinStepCombined(
                channels=self.channels,
                n_blocks=self.n_blocks,
                dropout_p=self.dropout_p,
                kernel_size=self.kernel_size,
                use_residual_scalar=self.use_residual_scalar,
                act_func=self.act_func,
                init_method=self.init_method,
                zero_initialize_output=self.zero_initialize_output,
                dtype=self.dtype,
                use_layernorm=self.use_layernorm,
                in_scale=(1 / self.data_scalars["drho"], 1 / self.data_scalars["dQ"], 1 / self.data_scalars["dP"])
                if self.data_scalars is not None
                else None,
                out_scale=(self.data_scalars["delta_rho"], self.data_scalars["delta_Q"], self.data_scalars["delta_P"])
                if self.data_scalars is not None
                else None,
            )
        else:
            self.cnn = HolsteinStepCombined(
                channels=self.channels,
                n_blocks=self.n_blocks,
                dropout_p=self.dropout_p,
                kernel_size=self.kernel_size,
                use_residual_scalar=self.use_residual_scalar,
                act_func=self.act_func,
                init_method=self.init_method,
                zero_initialize_output=self.zero_initialize_output,
                dtype=self.dtype,
                use_layernorm=self.use_layernorm,
                in_scale=(1 / self.data_scalars["rho"], 1 / self.data_scalars["Q"], 1 / self.data_scalars["P"])
                if self.data_scalars is not None
                else None,
                out_scale=(self.data_scalars["delta_rho"], self.data_scalars["delta_Q"], self.data_scalars["delta_P"])
                if self.data_scalars is not None
                else None,
            )

    def _step(
        self,
        rho: torch.Tensor,
        Q: torch.Tensor,
        P: torch.Tensor,
        return_derivatives: bool = False,
    ):
        """
        A single step of the Holstein integration

        Args:
            rho: torch.Tensor
                Complex-valued. Shape is [batch, L, L]
            Q: torch.Tensor
                Real-valued. Shape is [batch, L]
            P: torch.Tensor
                Real-valued. Shape is [batch, L]
            return_derivatives : bool
                Whether to return the derivatives from the differentiator CNN. Cannot be used if self.simple_cnn is True (as simple CNN has no integrator/differentiator, just one CNN)

        Returns:
            rho_step: torch.Tensor
                Complex-valued. Shape is [batch, L, L]
            Q_step: torch.Tensor
                Real-valued. Shape is [batch, L]
            P_step: torch.Tensor
                Real-valued. Shape is [batch, L]
            (optional) d_rho: torch.Tensor
                Complex-valued. Shape is [batch, L, L]
            (optional) d_Q: torch.Tensor
                Real-valued. Shape is [batch, L]
            (optional) d_P: torch.Tensor
                Real-valued. Shape is [batch, L]
        """

        if return_derivatives and self.simple_cnn:
            raise ValueError("Simple CNN cannot return derivatives!")

        """ Differentiate and integrate using CNNs """
        if not self.simple_cnn:
            d_rho, d_Q, d_P = self.differentiator(rho, Q, P)
            delta_rho, delta_Q, delta_P = self.integrator(d_rho, d_Q, d_P)
        else:
            delta_rho, delta_Q, delta_P = self.cnn(rho, Q, P)

        """ Shape and dtype check """
        if delta_rho.shape != rho.shape or delta_Q.shape != Q.shape or delta_P.shape != P.shape:
            raise RuntimeError(
                f"{self.__class__.__name__}: Shape inconsistency between original rho/Q/P and delta detected! Shapes:\n\trho:\t{rho.shape}delta rho:\t{delta_rho.shape}\n\tQ:\t{Q.shape}delta Q:\t{delta_Q.shape}\n\tP:\t{P.shape}delta P:\t{delta_P.shape}"
            )
        if delta_rho.dtype != rho.dtype or delta_Q.dtype != Q.dtype or delta_P.dtype != P.dtype:
            raise RuntimeError(
                f"{self.__class__.__name__}: dtype inconsistency between original rho/Q/P and delta detected! dtypes:\n\trho:\t{rho.dtype}delta rho:\t{delta_rho.dtype}\n\tQ:\t{Q.dtype}delta Q:\t{delta_Q.dtype}\n\tP:\t{P.dtype}delta P:\t{delta_P.dtype}"
            )
        if not self.simple_cnn:
            if d_rho.shape != rho.shape or d_Q.shape != Q.shape or d_P.shape != P.shape:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Shape inconsistency between original rho/Q/P and d_rho/d_Q/d_P detected! Shapes:\n\trho:\t{rho.shape}d rho:\t{d_rho.shape}\n\tQ:\t{Q.shape}d Q:\t{d_Q.shape}\n\tP:\t{P.shape}d P:\t{d_P.shape}"
                )
            if d_rho.dtype != rho.dtype or d_Q.dtype != Q.dtype or d_P.dtype != P.dtype:
                raise RuntimeError(
                    f"{self.__class__.__name__}: dtype inconsistency between original rho/Q/P and d_rho/d_Q/d_P detected! dtypes:\n\trho:\t{rho.dtype}d rho:\t{d_rho.dtype}\n\tQ:\t{Q.dtype}d Q:\t{d_Q.dtype}\n\tP:\t{P.dtype}d P:\t{d_P.dtype}"
                )

        """ Add deltas into original states and return """

        rho_step = rho + delta_rho
        Q_step = Q + delta_Q
        P_step = P + delta_P

        if not return_derivatives:
            return rho_step, Q_step, P_step
        else:
            return rho_step, Q_step, P_step, d_rho, d_Q, d_P

    def forward(
        self,
        rho: torch.Tensor,
        Q: torch.Tensor,
        P: torch.Tensor,
        return_derivatives: bool = False,
        n_step: int = 1,
        return_multiple_steps: bool = False,
        checkpointing: bool = False,
    ):
        """
        Iteratively feeds its own output back as the next input. Returns all outputs from all steps.

        Args:
            rho: torch.Tensor
                Complex-valued. Shape is [batch, L, L]
            Q: torch.Tensor
                Real-valued. Shape is [batch, L]
            P: torch.Tensor
                Real-valued. Shape is [batch, L]
            return_derivatives : bool
                Whether to return the derivatives from the differentiator CNN
            n_step : int
                How many steps to predict (cycles of recurrence)
            return_multiple_steps : bool
                If False, return only the last step. If True, return all steps. Does this by adding a time step dimension to the output (dim 1, the one AFTER batch), returning multiple consecutive predictions. Corresponds to multi_step_labels in HolsteinDataset
            checkpointing : bool
                Whether to use gradient checkpointing. If used, it will checkpoint each step of a multi-step forward pass.
                    Presents no benefit if n_step = 1 (unless the multi-step functionality is trying to be emulated outside of the model itself). Therefore, will error if n_step = 1

        Returns:
            rho_preds: torch.Tensor
                Complex-valued. Shape is [batch, n_step, L, L] if return_multiple_steps is True. Else it is [batch, L, L]
            Q_preds: torch.Tensor
                Real-valued. Shape is [batch, n_step, L] if return_multiple_steps is True. Else it is [batch, L]
            P_preds: torch.Tensor
                Real-valued. Shape is [batch, n_step, L] if return_multiple_steps is True. Else it is [batch, L]
            (optional) d_rho: torch.Tensor
                Complex-valued. Shape is [batch, n_step, L, L] if return_multiple_steps is True. Else it is [batch, L, L]
            (optional) d_Q: torch.Tensor
                Real-valued. Shape is [batch, n_step, L] if return_multiple_steps is True. Else it is [batch, L]
            (optional) d_P: torch.Tensor
                Real-valued. Shape is [batch, n_step, L] if return_multiple_steps is True. Else it is [batch, L]

        """

        if n_step < 1:
            raise ValueError(f"{self.__class__.__name__}: n_step cannot be less than 1. Received: {n_step}")

        if n_step == 1 and checkpointing:
            raise ValueError(
                f"{self.__class__.__name__}: checkpointing was enabled but n_step was 1. Checkpointing is not useful (and indeed inefficient) if n_step is not > 1"
            )

        if return_multiple_steps:
            steps_sequential = []

        for _ in range(n_step):
            if return_derivatives:
                if not checkpointing:
                    rho, Q, P, d_rho, d_Q, d_P = self._step(rho, Q, P, return_derivatives=True)
                else:
                    rho, Q, P, d_rho, d_Q, d_P = checkpoint(
                        self._step, rho, Q, P, return_derivatives=True, use_reentrant=False
                    )

                if return_multiple_steps:
                    steps_sequential.append((rho, Q, P, d_rho, d_Q, d_P))
            else:
                if not checkpointing:
                    rho, Q, P = self._step(rho, Q, P, return_derivatives=False)
                else:
                    rho, Q, P = checkpoint(self._step, rho, Q, P, return_derivatives=False, use_reentrant=False)

                if return_multiple_steps:
                    steps_sequential.append((rho, Q, P))

        if not return_multiple_steps:
            if return_derivatives:
                return rho, Q, P, d_rho, d_Q, d_P
            return rho, Q, P

        # process return multiple steps
        items = tuple(zip(*steps_sequential))  # Lists of rho, Q, P, and possibly d_rho, d_Q, and d_P from all steps

        items = [
            torch.stack(item, dim=1) for item in items
        ]  # Turn each into a tensor of shape [batch, n_step, L, ...] by stacking time along 1st dim

        return tuple(
            items
        )  # Returns a tuple of tensors (either rho, Q, P or rho, Q, P, d_rho, d_Q, d_P depending on return_derivatives) where each tensor is of shape [batch, n_step, L, ...]


def load_HolsteinPARC_pretrained(path: Path | str):
    path = Path(path)

    model_kwargs = unpickle_data(path.parent / "model_kwargs.pkl")

    model = HolsteinPARC(**model_kwargs)

    model.load_state_dict(torch.load(path, map_location=torch.device("cpu"), weights_only=True))

    return model
