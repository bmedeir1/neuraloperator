import torch
from torch import nn

from neuralop.layers.spectral_convolution import SpectralConv

class MLPAdaptableSpectralConv(nn.Module):
    """Wrapper for SpectralConv that adapts weights for coarsening by applying an MLP to the Fourier weights. 
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    Note that the same MLP is applied to all channels, only modes weights are affected in coarsening. 

    Once trained, network weights can be precomputed for any desired coarsening level. 
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.spectral_conv = SpectralConv(**kwargs)
        total_modes = self.spectral_conv.max_n_modes[0] * self.spectral_conv.max_n_modes[1]
        hidden_dim = 2 * total_modes
        self.mlp = nn.Sequential(
            nn.Linear(total_modes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_modes)
        )
        self.base_coarsening_level = 0  # Default to no coarsening

    def forward(self, x: torch.Tensor, coarsening_level: int, output_shape=None):
        if coarsening_level == 0:
            return self.spectral_conv.forward(x, output_shape=output_shape)

        # Get the original weight tensor
        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        # Flatten the mode dimensions for MLP input
        W_flat = W0.view(W0.shape[0], W0.shape[1], -1)  # (in_channels, out_channels, modes_flat)
        
        # Apply MLP with a skip connection to adapt weights
        for _ in range(coarsening_level):
            W_flat = self.mlp(W_flat) + W_flat  # (in_channels, out_channels, modes_flat)
        
        # Reshape back to original weight shape
        W_adapted = W_flat.view_as(W0)

        return self.spectral_conv.forward(x, output_shape=output_shape, override_weight=W_adapted)
    
    def merge_mlp_weights(self, coarsening_level: int):
        """Merge the MLP-adapted weights into the SpectralConv weight tensor for a given coarsening level.
        This allows for precomputing the adapted weights and removing the MLP from the forward pass.
        """
        if coarsening_level == 0:
            return  # No adaptation needed

        # Get the original weight tensor
        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        # Flatten the mode dimensions for MLP input
        W_flat = W0.view(W0.shape[0], W0.shape[1], -1)  # (in_channels, out_channels, modes_flat)
        
        # Apply MLP with a skip connection to adapt weights
        for _ in range(coarsening_level):
            W_flat = self.mlp(W_flat) + W_flat  # (in_channels, out_channels, modes_flat)
        
        # Reshape back to original weight shape
        W_adapted = W_flat.view_as(W0)

        # Update the SpectralConv weight with adapted weights
        self.spectral_conv.weight = nn.Parameter(W_adapted)

        self.base_coarsening_level += coarsening_level