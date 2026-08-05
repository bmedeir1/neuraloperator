import math

import torch
from torch import nn

from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.spectral_convolution import SpectralConv, get_contract_fun


class MLPAdaptableSpectralConv(BaseSpectralConv):
    """Wrapper for SpectralConv that adapts weights for coarsening by applying an MLP to the Fourier weights.
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    The MLP is applied independently to the real and imaginary parts of the complex weights.
    Note that the same MLP is applied to all channels; only mode weights are affected by coarsening.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(device=kwargs.get("device"))
        self.spectral_conv = SpectralConv(*args, **kwargs)
        total_modes = math.prod(self.spectral_conv.max_n_modes)
        hidden_dim = 2 * total_modes
        self.mlp = nn.Sequential(
            nn.Linear(total_modes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_modes),
        )
        self.base_coarsening_level = 0

    def transform(self, x, output_shape=None):
        return self.spectral_conv.transform(x, output_shape=output_shape)

    @property
    def n_modes(self):
        return self.spectral_conv.n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.spectral_conv.n_modes = n_modes

    def _flatten_weight(self, W0: torch.Tensor) -> torch.Tensor:
        """Reshape weight tensor to (..., total_modes) for MLP input."""
        if self.spectral_conv.separable:
            return W0.view(W0.shape[0], -1)
        return W0.view(W0.shape[0], W0.shape[1], -1)

    def _apply_mlp(self, W_flat: torch.Tensor, coarsening_level: int) -> torch.Tensor:
        """Apply MLP with skip connection `coarsening_level` times.

        The MLP operates on real-valued tensors, so real and imaginary parts
        of the complex weight are adapted independently with shared parameters.
        """
        for _ in range(coarsening_level):
            W_real = self.mlp(W_flat.real) + W_flat.real
            W_imag = self.mlp(W_flat.imag) + W_flat.imag
            W_flat = torch.complex(W_real, W_imag)
        return W_flat

    def forward(self, x: torch.Tensor, coarsening_level: int = 0, output_shape=None):
        if coarsening_level == 0:
            return self.spectral_conv.forward(x, output_shape=output_shape)

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_mlp(W_flat, coarsening_level).view_as(W0)

        return self.spectral_conv.forward(x, output_shape=output_shape, override_weight=W_adapted)

    def merge_mlp_weights(self, coarsening_level: int):
        """Merge the MLP-adapted weights into the SpectralConv weight tensor for a given coarsening level.
        This allows for precomputing the adapted weights and removing the MLP from the forward pass.
        """
        if coarsening_level == 0:
            return

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_mlp(W_flat, coarsening_level).view_as(W0)

        self.spectral_conv.weight = nn.Parameter(W_adapted.detach())
        self.spectral_conv._contract = get_contract_fun(
            self.spectral_conv.weight,
            implementation=self.spectral_conv.implementation,
            separable=self.spectral_conv.separable,
        )
        self.base_coarsening_level += coarsening_level
