# Partially based on https://github.com/zongyi-li/fourier_neural_operator

from typing import Optional

import torch


class FourierNeuralLayer1D(torch.nn.Module):
    """A Fourier neural layer object.

    A Fourier neural layer involves a Fourier transform over the spatial
    dimensions of the input, followed by a learned elementwise multiplication
    and an inverse Fourier transform.

    Attributes:
        in_channels: An integer indicating the number of input channels.
        out_channels: An integer indicating the number of output channels.
        modes: An integer indicating the number of Fourier modes to update.
        weights: A list containing two Tensors of size `(in_channels,
            out_channels, modes, modes)` as weights for the layer.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """Initializes a Fourier neural layer object.

        Args:
            in_channels: An integer indicating the number of input channels.
            out_channels: An integer indicating the number of output channels.
            modes: An integer indicating the number of Fourier modes to update.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # A scalar factor for initializing the Fourier neural layer weights.
        scale = 1.0 / (in_channels * out_channels)

        # Initialize weights of the layer.
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(scale * torch.rand(
                in_channels, out_channels, self.modes, dtype=torch.cfloat)),
            torch.nn.Parameter(scale * torch.rand(
                in_channels, out_channels, self.modes, dtype=torch.cfloat))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier neural layer.

        Args:
            x: A torch.Tensor of size `(batch_size, in_channels, length)`.

        Returns:
            A torch.Tensor of size `(batch_size, out_channels, length)`,
            which is The outcome of the `FourierNeuralLayer1D`.
        """
        # Compute 1D Fourier transform of the real values input, and return the
        # one-sided representation, i.e., only the positive frequencies with
        # shape `(batch_size, in_channels, length // 2 + 1)`.
        x_fft = torch.fft.rfft(x)

        # Placeholder for the one-sided representation of the output Tensor in
        # the Fourier domain with shape `(batch_size, out_channels, length // 2
        # + 1)`
        out = torch.zeros(
            [x.shape[0], self.out_channels, x.shape[-1] // 2 + 1],
            dtype=torch.cfloat,
            device=x.device)

        # Elementwise multiplication in Fourier domain. This operation involves
        # an elementwise multiplication followed by sum of `x_fft` and
        # `self.weights[:, o, :]` where `o` is the index of the output
        # channel.
        out[:, :, :self.modes] = torch.einsum("bix, iox -> box",
                                              x_fft[..., :self.modes],
                                              self.weights[0])
        out[:, :, -self.modes:] = torch.einsum("bix, iox -> box",
                                               x_fft[..., -self.modes:],
                                               self.weights[1])

        # Inverse 1D Fourier transform to get a real valued signal.
        return torch.fft.irfft(out, n=x.shape[-1])


class FourierNeuralLayer2D(torch.nn.Module):
    """A Fourier neural layer object.

    A Fourier neural layer involves a Fourier transform over the spatial
    dimensions of the input, followed by a learned elementwise multiplication
    and an inverse Fourier transform.

    Attributes:
        in_channels: An integer indicating the number of input channels.
        out_channels: An integer indicating the number of output channels.
        modes: An integer indicating the number of Fourier modes to update.
        weights: A list containing two Tensors of size `(in_channels,
            out_channels, modes, modes)` as weights for the layer.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """Initializes a Fourier neural layer object.

        Args:
            in_channels: An integer indicating the number of input channels.
            out_channels: An integer indicating the number of output channels.
            modes: An integer indicating the number of Fourier modes to update.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # A scalar factor for initializing the Fourier neural layer weights.
        scale = 1.0 / (in_channels * out_channels)

        # Initialize weights of the layer.
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(scale * torch.rand(in_channels,
                                                  out_channels,
                                                  self.modes,
                                                  self.modes,
                                                  dtype=torch.cfloat)),
            torch.nn.Parameter(scale * torch.rand(in_channels,
                                                  out_channels,
                                                  self.modes,
                                                  self.modes,
                                                  dtype=torch.cfloat))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier neural layer.

        Args:
            x: A torch.Tensor of size `(batch_size, in_channels, height,
                width)`.

        Returns:
            A torch.Tensor of size `(batch_size, out_channels, height, width)`,
            which is The outcome of the `FourierNeuralLayer2D`.
        """
        # Compute 2D Fourier transform of the real values input, and return the
        # one-sided representation, i.e., only the positive frequencies with
        # shape `(batch_size, in_channels, height, width // 2 + 1)`.
        x_fft = torch.fft.rfft2(x)

        # Placeholder for the one-sided representation of the output Tensor in
        # the Fourier domain with shape `(batch_size, out_channels, height,
        # width // 2 + 1)`
        out = torch.zeros(
            [x.shape[0], self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1],
            dtype=torch.cfloat,
            device=x.device)

        # Elementwise multiplication in Fourier domain. This operation involves
        # an elementwise multiplication followed by sum of `x_fft` and
        # `self.weights[:, o, :, :]` where `o` is the index of the output
        # channel.
        out[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy, ioxy -> boxy", x_fft[..., :self.modes, :self.modes],
            self.weights[0])
        out[:, :, -self.modes:, :self.modes] = torch.einsum(
            "bixy, ioxy -> boxy", x_fft[..., -self.modes:, :self.modes],
            self.weights[1])

        # Inverse 2D Fourier transform to get a real valued signal.
        return torch.fft.irfft2(out, s=(x.shape[-2], x.shape[-1]))


class FourierNeuralOperator1D(torch.nn.Module):
    """A Fourier neural operator object.

    The main components of Fourier neural operators are: (1) a linear lifting
    operator, (2) a Fourier layer, involving a Conv1d skip connection; and (3) a
    linear dimensionality reduction operator.

    Attributes:
        lifted_dim: An integer indicating the lifted dimensions.
        num_fourier_layers: An integer indicating the number of Fourier
            Layers.
        linear_layers: A list of torch.nn.Linear layers.
        fourier_layers: A list of FourierNeuralLayer1D layers.
        conv1d_layers: A list of torch.nn.Conv1d layers.
        batchnorm_layers: A list of torch.nn.BatchNorm layers.
    """

    def __init__(self,
                 modes: int,
                 lifted_dim: int,
                 in_length: int,
                 out_length: int,
                 num_fourier_layers: Optional[int] = 4):
        """Initializes a Fourier neural operator object.

        Args:
            modes: An integer indicating the number of Fourier modes to update.
            lifted_dim: An integer indicating the lifted dimensions.
            num_fourier_layers: An optional integer indicating the number of
                Fourier layers.
        """
        super().__init__()
        self.lifted_dim = lifted_dim
        self.num_fourier_layers = num_fourier_layers

        # Initialize linear lifting and the linear dimensionality reduction
        # operators.
        self.linear_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_length, self.lifted_dim),
            torch.nn.Linear(self.lifted_dim, 64),
            torch.nn.Linear(64, out_length)
        ])

        # Initialize Fourier neural layers.
        self.fourier_layers = torch.nn.ModuleList([
            FourierNeuralLayer1D(self.lifted_dim, self.lifted_dim, modes)
            for _ in range(num_fourier_layers)
        ])

        # Initialize Conv1d skip layers.
        self.conv1d_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(self.lifted_dim, self.lifted_dim, 1)
            for _ in range(num_fourier_layers)
        ])

        # Initialize batch normalization layers.
        self.batchnorm_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(self.lifted_dim)
            for _ in range(num_fourier_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier neural operator.

        Args:
            x: A torch.Tensor of size `(batch_size, length, 3)`.

        Returns:
            A torch.Tensor of size `(batch_size, length, 1)`, which is
            the outcome of the `FourierNeuralOperator1D`.
        """
        # from IPython import embed; embed()
        # Extract the `batch_size`, and `length` values.
        batch_size, length = x.shape[0:2]

        # Linear lifting layer, which is applied to the last dimension.
        x = self.linear_layers[0](x)

        # Permutation resulting in the shape `(batch_size, lifted_dim,
        # length)`.
        x = x.permute(0, 2, 1)

        # A loop over the Fourier neural layers.
        for j in range(self.num_fourier_layers):
            # Forward pass of the Fourier layer.
            x1 = self.fourier_layers[j](x)

            # Forward pass of the Conv1d skip connection. This is equivalent to
            # applying a linear layer to the lifted dimension.
            x2 = self.conv1d_layers[j](x.view(batch_size, self.lifted_dim, -1))
            x2 = x2.view(batch_size, self.lifted_dim, length)

            # Batch normalization
            x = self.batchnorm_layers[j](x1 + x2)

            # Apply ReLU at the end of the Fourier neural layers, except for
            # the last one, which instead we undo the permutation that was
            # applied before Fourier neural layers.
            if j < self.num_fourier_layers - 1:
                x = torch.nn.functional.relu(x)
            else:
                # Back to the shape `(batch_size, length, lifted_dim)`.
                x = x.permute(0, 2, 1)

        # A linear layer applied to the lifted dimension, altering the shape to
        # `(batch_size, length, 128)`.
        x = self.linear_layers[1](x)
        x = torch.nn.functional.relu(x)

        # A linear layer applied to the last dimension, altering the shape to
        # `(batch_size, length, 1)`.
        x = self.linear_layers[2](x)

        return x[..., 0]


class FourierNeuralOperator2D(torch.nn.Module):
    """A Fourier neural operator object.

    The main components of Fourier neural operators are: (1) a linear lifting
    operator, (2) a Fourier layer, involving a Conv1d skip connection; and (3) a
    linear dimensionality reduction operator.

    Attributes:
        lifted_dim: An integer indicating the lifted dimensions.
        num_fourier_layers: An integer indicating the number of Fourier
            Layers.
        linear_layers: A list of torch.nn.Linear layers.
        fourier_layers: A list of FourierNeuralLayer2D layers.
        conv1d_layers: A list of torch.nn.Conv1d layers.
        batchnorm_layers: A list of torch.nn.BatchNorm layers.
    """

    def __init__(self,
                 modes: int,
                 lifted_dim: int,
                 in_length: int,
                 out_length: int,
                 num_fourier_layers: Optional[int] = 4):
        """Initializes a Fourier neural operator object.

        Args:
            modes: An integer indicating the number of Fourier modes to update.
            lifted_dim: An integer indicating the lifted dimensions.
            num_fourier_layers: An optional integer indicating the number of
                Fourier layers.
        """
        super().__init__()
        self.lifted_dim = lifted_dim
        self.num_fourier_layers = num_fourier_layers

        # Initialize linear lifting and the linear dimensionality reduction
        # operators.
        self.linear_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_length, self.lifted_dim),
            torch.nn.Linear(self.lifted_dim, 128),
            torch.nn.Linear(128, out_length)
        ])

        # Initialize Fourier neural layers.
        self.fourier_layers = torch.nn.ModuleList([
            FourierNeuralLayer2D(self.lifted_dim, self.lifted_dim, modes)
            for _ in range(num_fourier_layers)
        ])

        # Initialize Conv1d skip layers.
        self.conv1d_layers = torch.nn.ModuleList([
            torch.nn.Conv1d(self.lifted_dim, self.lifted_dim, 1)
            for _ in range(num_fourier_layers)
        ])

        # Initialize batch normalization layers.
        self.batchnorm_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(self.lifted_dim)
            for _ in range(num_fourier_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier neural operator.

        Args:
            x: A torch.Tensor of size `(batch_size, height, width,
                4)`, where the last dimensions involves the two input
                coefficients plus two dimensions for the height and width
                coordinates.

        Returns:
            A torch.Tensor of size `(batch_size, height, width, 1)`, which is
            the outcome of the `FourierNeuralOperator2D`.
        """
        # Extract the `batch_size`, `height`, and `width` values.
        batch_size, height, width = x.shape[0:3]

        # Linear lifting layer, which is applied to the last dimension.
        x = self.linear_layers[0](x)

        # Permutation resulting in the shape `(batch_size, lifted_dim, height,
        # width)`.
        x = x.permute(0, 3, 1, 2)

        # A loop over the Fourier neural layers.
        for j in range(self.num_fourier_layers):
            # Forward pass of the Fourier layer.
            x1 = self.fourier_layers[j](x)

            # Forward pass of the Conv1d skip connection. This is equivalent to
            # applying a linear layer to the lifted dimension.
            x2 = self.conv1d_layers[j](x.view(batch_size, self.lifted_dim, -1))
            x2 = x2.view(batch_size, self.lifted_dim, height, width)

            # Batch normalization
            x = self.batchnorm_layers[j](x1 + x2)

            # Apply ReLU at the end of the Fourier neural layers, except for the
            # last one, which instead we undo the permutation that was applied
            # before Fourier neural layers.
            if j < self.num_fourier_layers - 1:
                x = torch.nn.functional.relu(x)
            else:
                # Back to the shape `(batch_size, height, width, lifted_dim)`.
                x = x.permute(0, 2, 3, 1)

        # A linear layer applied to the lifted dimension, altering the shape to
        # `(batch_size, height, width, 128)`.
        x = self.linear_layers[1](x)
        x = torch.nn.functional.relu(x)

        # A linear layer applied to the last dimension, altering the shape to
        # `(batch_size, height, width, 1)`.
        x = self.linear_layers[2](x)

        return x