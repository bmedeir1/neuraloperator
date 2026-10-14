import math

import torch
from torch import nn

from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.spectral_convolution import SpectralConv, get_contract_fun

class AdaptableSpectralConv(BaseSpectralConv):
    """Base class for wrapping SpectralConv that adapts Fourier weights for coarsening 
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    
    Once trained, network weights can be precomputed for any desired coarsening level.

    Parameters
    ----------
    hidden_scale : int, optional
        Expansion factor of the weight transform's hidden dimension relative to its input
        dimension, by default 2. All other arguments are passed to SpectralConv.
    """

    def __init__(self, *args, hidden_scale: int = 2, **kwargs):
        super().__init__(device=kwargs.get("device"))
        self.spectral_conv = SpectralConv(*args, **kwargs)
        self.total_modes = math.prod(self.spectral_conv.max_n_modes)
        self.hidden_scale = hidden_scale
        self.hidden_dim = hidden_scale * self.total_modes
        self.weight_transform = self.default_weight_transform
        self.base_coarsening_level = 0

    def default_weight_transform(self, W_flat: torch.Tensor):
        raise NotImplementedError("Subclasses must define a weight transform method.")

    def transform(self, x, output_shape=None):
        return self.spectral_conv.transform(x, output_shape=output_shape)
    
    def _flatten_weight(self, W0: torch.Tensor) -> torch.Tensor:
        """Reshape weight tensor to (..., total_modes) for weight transform input."""
        if self.spectral_conv.separable:
            return W0.view(W0.shape[0], -1)
        return W0.view(W0.shape[0], W0.shape[1], -1)
    
    def _apply_weight_transform(self, W_flat: torch.Tensor, coarsening_level: int) -> torch.Tensor:
        """Apply weight_transform with skip connection `coarsening_level` times.

        Operates on real-valued tensors, so real and imaginary parts
        of the complex weight are adapted independently with shared parameters.

        Residual block formulation to match LoRA style adaptation, where the original weight is added to the transformed weight.
        """
        for _ in range(coarsening_level):
            W_real = self.weight_transform(W_flat.real) + W_flat.real
            W_imag = self.weight_transform(W_flat.imag) + W_flat.imag
            W_flat = torch.complex(W_real, W_imag)
        return W_flat
    
    def forward(self, x: torch.Tensor, coarsening_level: int = 0, output_shape=None):
        if coarsening_level == 0:
            return self.spectral_conv.forward(x, output_shape=output_shape)

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_weight_transform(W_flat, coarsening_level).view_as(W0)

        return self.spectral_conv.forward(x, output_shape=output_shape, override_weight=W_adapted)

    def merge_weights(self, coarsening_level: int):
        """Merge the MLP-adapted weights into the SpectralConv weight tensor for a given coarsening level.
        This allows for precomputing the adapted weights and removing the MLP from the forward pass.
        """
        if coarsening_level == 0:
            return

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_weight_transform(W_flat, coarsening_level).view_as(W0)

        self.spectral_conv.weight = nn.Parameter(W_adapted.detach())
        self.spectral_conv._contract = get_contract_fun(
            self.spectral_conv.weight,
            implementation=self.spectral_conv.implementation,
            separable=self.spectral_conv.separable,
        )
        self.base_coarsening_level += coarsening_level

    @property
    def n_modes(self):
        return self.spectral_conv.n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.spectral_conv.n_modes = n_modes

def coarsen_model(model: nn.Module, coarsening_level: int) -> nn.Module:
    """Merge MLP-adapted weights into SpectralConv weights in a model to coarsen it.

    Recursively walks the module tree and calls merge_mlp_weights(coarsening_level) on each
    AdaptableSpectralConv layer in-place. This effectively updates the underlying
    SpectralConv weights to reflect the coarsened representation.

    Works with nested modules including ``nn.ModuleList``.

    Parameters
    ----------
    model : nn.Module
        Model whose AdaptableSpectralConv layers should be merged into SpectralConv.
    coarsening_level : int
        Number of levels of coarsening to apply to each AdaptableSpectralConv-wrapped SpectralConv layer.

    Returns
    -------
    nn.Module
        The same model object with coarsened SpectralConv weights.
    """
    for _, module in model.named_children():
        if isinstance(module, AdaptableSpectralConv):
            module.merge_weights(coarsening_level)
        else:
            coarsen_model(module, coarsening_level)
    return model

class MLPAdaptableSpectralConv(AdaptableSpectralConv):
    """Wrapper for SpectralConv that adapts weights for coarsening by applying an MLP to the Fourier weights.
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    The MLP is applied independently to the real and imaginary parts of the complex weights.
    Note that the same MLP is applied to all channels; only mode weights are affected by coarsening.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_transform = nn.Sequential(
            nn.Linear(self.total_modes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.total_modes),
        )


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W1 = nn.Linear(d_in, d_out)
        self.W2 = nn.Linear(d_in, d_out)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.W1(x) * self.silu(self.W2(x))

class GLUAdaptableSpectralConv(AdaptableSpectralConv):
    """Wrapper for SpectralConv that adapts weights for coarsening by applying a Gated Linear Unit (GLU)-based 
    transformation to the Fourier weights.
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    The GLU is applied independently to the real and imaginary parts of the complex weights.
    Note that the same GLU is applied to all channels; only mode weights are affected by coarsening.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_transform = nn.Sequential(
            nn.Linear(self.total_modes, self.hidden_dim),
            nn.GELU(),
            SwiGLU(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.total_modes),
        )


class FactorizedChannelAdaptableSpectralConv(AdaptableSpectralConv):
    """Base class for adapters that mix only along the channel dimensions of the Fourier weights,
    factorized into one transform per channel axis.

    For a dense weight of shape (in, out, *modes), the update is applied sequentially:
    first along the input-channel axis (shared across output channels and modes), then along
    the output-channel axis (shared across input channels and modes), each with a skip connection.
    For a separable weight of shape (channels, *modes), there is a single channel axis,
    so only the input-channel transform is used.

    Compared to transforming the flattened (in * out) channel matrix at each mode jointly,
    this reduces the parameter count from O((in * out)^2) to O(in^2 + out^2).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sc = self.spectral_conv
        self.in_transform = self._make_transform(sc.in_channels)
        self.out_transform = None if sc.separable else self._make_transform(sc.out_channels)
        self.weight_transform = self.factorized_weight_transform

    def _make_transform(self, n_channels: int) -> nn.Module:
        raise NotImplementedError("Subclasses must define the per-axis channel transform.")

    def factorized_weight_transform(self, W_flat: torch.Tensor) -> torch.Tensor:
        """Return the total update to W_flat; the outer skip connection adds W_flat back."""
        if self.spectral_conv.separable:
            # (channels, modes) -> (modes, channels)
            return self.in_transform(W_flat.transpose(0, 1)).transpose(0, 1)

        # (in, out, modes) -> (out, modes, in): mix along input channels
        dW_in = self.in_transform(W_flat.permute(1, 2, 0)).permute(2, 0, 1)
        W_mid = W_flat + dW_in
        # (in, out, modes) -> (in, modes, out): mix along output channels
        dW_out = self.out_transform(W_mid.permute(0, 2, 1)).permute(0, 2, 1)
        return dW_in + dW_out


class MLPFactorizedChannelAdaptableSpectralConv(FactorizedChannelAdaptableSpectralConv):
    """Factorized channel adapter where each per-axis transform is an MLP.
    The MLPs are applied independently to the real and imaginary parts of the complex weights.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def _make_transform(self, n_channels: int) -> nn.Module:
        hidden_dim = self.hidden_scale * n_channels
        return nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels),
        )


class GLUFactorizedChannelAdaptableSpectralConv(FactorizedChannelAdaptableSpectralConv):
    """Factorized channel adapter where each per-axis transform is a Gated Linear Unit (GLU)-based network.
    The GLUs are applied independently to the real and imaginary parts of the complex weights.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def _make_transform(self, n_channels: int) -> nn.Module:
        hidden_dim = self.hidden_scale * n_channels
        return nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            SwiGLU(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, n_channels),
        )
