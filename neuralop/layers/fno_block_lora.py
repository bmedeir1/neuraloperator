import math
from typing import Optional

import torch
from torch import nn

from neuralop.layers.fno_block import FNOBlocks


class LoRAFNOBlocks(nn.Module):
    """Wrapper for FNOBlocks that adapts the output of each Fourier layer with a low-rank update.

    Each layer ``index`` has its own pair of low-rank matrices A (rank, channels) and
    B (channels, rank), applied pointwise along the channel dimension of the layer output y:

        y <- y + B A y

    The update is applied ``coarsening_level`` times, i.e. y <- (I + BA)^n y at level n, so that
    one application corresponds to one level of coarsening, matching AdaptableSpectralConv.
    At ``coarsening_level=0`` the wrapped FNOBlocks is used unchanged.

    If ``rank`` is None, each layer instead has a single full-rank matrix W (channels, channels)
    and the update is y <- y + W y.

    B (or W) is initialized to zero so that the adapted model initially matches the wrapped one.

    Parameters
    ----------
    fno_blocks : FNOBlocks
        FNOBlocks module to adapt.
    rank : int or None, optional
        Rank of the low-rank update, by default None (full-rank update).
    freeze_base : bool, optional
        Whether to freeze the parameters of the wrapped FNOBlocks, by default True.
    """

    def __init__(self, fno_blocks: FNOBlocks, rank: Optional[int] = None, freeze_base: bool = True):
        super().__init__()
        if fno_blocks.complex_data:
            raise NotImplementedError("LoRAFNOBlocks does not support complex_data yet.")

        self.fno_blocks = fno_blocks
        self.rank = rank
        if freeze_base:
            for p in self.fno_blocks.parameters():
                p.requires_grad_(False)

        channels = fno_blocks.out_channels
        n_layers = fno_blocks.n_layers
        device = next(fno_blocks.parameters()).device
        if rank is None:
            self.lora_W = nn.ParameterList(
                [nn.Parameter(torch.zeros(channels, channels, device=device)) for _ in range(n_layers)]
            )
            return

        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, channels, device=device)) for _ in range(n_layers)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(channels, rank, device=device)) for _ in range(n_layers)]
        )
        for A in self.lora_A:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

    def _lora_update(self, y: torch.Tensor, index: int) -> torch.Tensor:
        """Compute B A y (or W y if full-rank) pointwise along the channel dimension of y of shape (batch, channels, *spatial)."""
        if self.rank is None:
            return torch.einsum("dc,bc...->bd...", self.lora_W[index], y)
        Ay = torch.einsum("rc,bc...->br...", self.lora_A[index], y)
        return torch.einsum("cr,br...->bc...", self.lora_B[index], Ay)

    def forward(self, x, index=0, output_shape=None, coarsening_level=0):
        y = self.fno_blocks(x, index, output_shape=output_shape, coarsening_level=coarsening_level)
        for _ in range(coarsening_level):
            y = y + self._lora_update(y, index)
        return y

    @property
    def n_layers(self):
        return self.fno_blocks.n_layers

    @property
    def n_modes(self):
        return self.fno_blocks.n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes

    def set_ada_in_embeddings(self, *embeddings):
        self.fno_blocks.set_ada_in_embeddings(*embeddings)


def inject_lora_fno_blocks(model: nn.Module, rank: Optional[int] = None, freeze_base: bool = True) -> nn.Module:
    """Replace all FNOBlocks in a model with LoRAFNOBlocks wrappers.

    Recursively walks the module tree and replaces each FNOBlocks in-place with a
    LoRAFNOBlocks of the given rank.

    Works with nested modules including ``nn.ModuleList``.

    Parameters
    ----------
    model : nn.Module
        Model whose FNOBlocks should be adapted.
    rank : int or None, optional
        Rank of the low-rank update, by default None (full-rank update).
    freeze_base : bool, optional
        Whether to freeze the parameters of the wrapped FNOBlocks, by default True.

    Returns
    -------
    nn.Module
        The same model object with FNOBlocks replaced in-place.
    """
    for name, module in model.named_children():
        if isinstance(module, FNOBlocks):
            setattr(model, name, LoRAFNOBlocks(module, rank, freeze_base=freeze_base))
        else:
            inject_lora_fno_blocks(module, rank, freeze_base=freeze_base)
    return model
