import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (AdaIN) layer for style transfer in neural operators.

    AdaIN performs instance normalization followed by adaptive scaling and shifting
    based on an external embedding vector. This allows for style transfer by
    modulating the output characteristics based on a conditioning signal.

    The layer first normalizes the input using instance normalization, then applies
    learned scaling (weight) and shifting (bias) parameters derived from an embedding
    vector through a multi-layer perceptron.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vector used for style conditioning
    in_channels : int
        Number of input channels to normalize
    mlp : nn.Module, optional
        Multi-layer perceptron that maps embedding to (weight, bias) parameters.
        Should output 2*in_channels values. If None, a default MLP is created
        with architecture: Linear(embed_dim, 512) -> GELU -> Linear(512, 2*in_channels)
    eps : float, optional
        Small value added to the denominator for numerical stability in normalization.
        Default is 1e-5.
    """

    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp

        self.embedding = None

    def set_embedding(self, x):
        """Set the embedding vector for style conditioning."""
        self.embedding = x.reshape(
            self.embed_dim,
        )

    def forward(self, x):
        """Apply adaptive instance normalization to the input tensor."""
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)


class BatchAdaIN(nn.Module):
    """Per-sample Adaptive Instance Normalization for neural operators.

    Like AdaIN, BatchAdaIN performs instance normalization followed by adaptive
    scaling and shifting derived from an embedding through an MLP. Unlike AdaIN,
    which applies a single embedding's scale/shift to every sample in the batch,
    BatchAdaIN accepts one embedding per sample and computes each sample's own
    scale/shift independently. This is appropriate when the conditioning signal
    (e.g. a physical parameter) varies across samples in the same batch.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vector used for conditioning
    in_channels : int
        Number of input channels to normalize
    mlp : nn.Module, optional
        Multi-layer perceptron that maps embeddings to (weight, bias) parameters.
        Should output 2*in_channels values per sample. If None, a default MLP is
        created with architecture: Linear(embed_dim, 512) -> GELU -> Linear(512, 2*in_channels)
    eps : float, optional
        Small value added to the denominator for numerical stability in normalization.
        Default is 1e-5.
    """

    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp

        self.embedding = None

    def set_embedding(self, x):
        """Set the per-sample embeddings used for conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Batch of embeddings of shape (batch, embed_dim), one per sample.
        """
        assert x.ndim == 2, (
            "BatchAdaIN: expected embeddings of shape (batch, embed_dim), "
            f"got shape {tuple(x.shape)}"
        )
        self.embedding = x.reshape(x.shape[0], self.embed_dim)

    def forward(self, x):
        """Apply per-sample adaptive instance normalization to the input tensor."""
        assert (
            self.embedding is not None
        ), "BatchAdaIN: update embeddding before running forward"
        assert self.embedding.shape[0] == x.shape[0], (
            f"BatchAdaIN: batch size of embeddings ({self.embedding.shape[0]}) "
            f"must match batch size of input ({x.shape[0]})"
        )

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=-1)

        normalized = nn.functional.instance_norm(x, eps=self.eps)
        broadcast_shape = (x.shape[0], self.in_channels) + (1,) * (x.ndim - 2)
        return normalized * weight.view(broadcast_shape) + bias.view(broadcast_shape)

class AdaLNModulation(nn.Module):
    """Per-sample conditioned modulation (adaLN-Zero) for neural operators.

    Follows the adaLN-Zero conditioning of DiT [1]_: the input is normalized,
    then modulated as ``norm(x) * (1 + scale) + shift``, and a per-channel
    ``gate`` is returned alongside to scale the output of the residual branch
    the modulated tensor is fed into. ``scale``, ``shift`` and ``gate`` are
    produced per sample from a conditioning embedding by an MLP.

    Unlike DiT, which uses LayerNorm over channels for each token, normalization
    here is instance normalization (over the spatial dimensions, per sample and
    channel), the natural analogue for function-space inputs.

    When the default MLP is used, its last layer is zero-initialized, so that at
    initialization ``scale = shift = gate = 0``: the modulation is the identity
    on the normalized input and the gated branch contributes nothing.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding vector used for conditioning
    in_channels : int
        Number of input channels to normalize
    mlp : nn.Module, optional
        Multi-layer perceptron that maps embeddings to scale, shift, and gate parameters.
        Should output 3*in_channels values per sample. If None, a default MLP is
        created with architecture: Linear(embed_dim, 512) -> GELU -> Linear(512, 3*in_channels),
        with the last layer zero-initialized.
    eps : float, optional
        Small value added to the denominator for numerical stability in normalization.
        Default is 1e-5.

    References
    ----------
    .. [1] Peebles, W., Xie, S. "Scalable Diffusion Models with Transformers" (2023).
           ICCV 2023, https://arxiv.org/abs/2212.09748.
    """

    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 3 * in_channels)
            )
            # adaLN-Zero: start with scale = shift = gate = 0
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        self.mlp = mlp

        self.embedding = None

    def set_embedding(self, x):
        """Set the per-sample embeddings used for conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Batch of embeddings of shape (batch, embed_dim), one per sample.
        """
        assert x.ndim == 2, (
            "AdaLNModulation: expected embeddings of shape (batch, embed_dim), "
            f"got shape {tuple(x.shape)}"
        )
        self.embedding = x.reshape(x.shape[0], self.embed_dim)

    def forward(self, x):
        """Normalize and modulate the input; also return the gate.

        Returns
        -------
        modulated : torch.Tensor
            ``instance_norm(x) * (1 + scale) + shift``, same shape as ``x``.
        gate : torch.Tensor
            Per-sample, per-channel gate of shape (batch, in_channels, 1, ..., 1),
            to multiply the output of the residual branch.
        """
        assert (
            self.embedding is not None
        ), "AdaLNModulation: update embeddding before running forward"
        assert self.embedding.shape[0] == x.shape[0], (
            f"AdaLNModulation: batch size of embeddings ({self.embedding.shape[0]}) "
            f"must match batch size of input ({x.shape[0]})"
        )

        scale, shift, gate = torch.split(
            self.mlp(self.embedding), self.in_channels, dim=-1
        )
        normalized = nn.functional.instance_norm(x, eps=self.eps)
        broadcast_shape = (x.shape[0], self.in_channels) + (1,) * (x.ndim - 2)
        modulated = normalized * (1 + scale.view(broadcast_shape)) + shift.view(
            broadcast_shape
        )
        return modulated, gate.view(broadcast_shape)


class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators.

    InstanceNorm normalizes each sample in the batch independently, computing
    mean and variance across spatial dimensions for each sample and channel
    separately. This is useful when the statistical properties of each sample
    are distinct and should be treated separately.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional parameters to pass to torch.nn.functional.instance_norm().
        Common parameters include:
        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
        - momentum : float, optional
            Value used for the running_mean and running_var computation.
            Default is 0.1.
        - use_input_stats : bool, optional
            If True, use input statistics. Default is True.
        - weight : torch.Tensor, optional
            Weight tensor for affine transformation. If None, no scaling applied.
        - bias : torch.Tensor, optional
            Bias tensor for affine transformation. If None, no bias applied.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        """Apply instance normalization to the input tensor."""
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class BatchNorm(nn.Module):
    """Dimension-agnostic batch normalization layer for neural operators.

    BatchNorm normalizes data across the entire batch, computing a single mean
    and standard deviation for all samples combined. This is the most common
    form of normalization and is effective when batch statistics are a good
    approximation of the overall data distribution.

    For dimensions > 3, the layer automatically flattens spatial dimensions
    and uses BatchNorm1d, as PyTorch doesn't implement batch norm for 4D+ tensors.

    Parameters
    ----------
    n_dim : int
        Spatial dimension of input data (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
        Determined by FNOBlocks.n_dim. If n_dim > 3, spatial dimensions are
        flattened and BatchNorm1d is used.
    num_features : int
        Number of channels in the input tensor to be normalized
    **kwargs : dict, optional
        Additional parameters to pass to the underlying batch normalization layer.
        Common parameters include:
        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
        - momentum : float, optional
            Value used for the running_mean and running_var computation.
            Default is 0.1.
        - affine : bool, optional
            If True, apply learnable affine transformation. Default is True.
        - track_running_stats : bool, optional
            If True, track running statistics. Default is True.
    """

    def __init__(self, n_dim: int, num_features: int, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.num_features = num_features
        self.kwargs = kwargs

        if self.n_dim <= 3:
            self.norm = getattr(torch.nn, f"BatchNorm{n_dim}d")(
                num_features=num_features, **kwargs
            )
        else:
            print(
                "Warning: torch does not implement batch norm for dimensions higher than 3.\
                  We manually flatten the spatial dimension of 4+D tensors to apply batch norm. "
            )
            self.norm = torch.nn.BatchNorm1d(num_features=num_features, **kwargs)

    def forward(self, x):
        """Apply batch normalization to the input tensor."""
        size = x.shape
        num_channels = size[1]

        # in 4+D, we flatten and use batchnorm1d.
        if self.n_dim >= 4:
            x = x.reshape(size[0], size[1], -1)
        x = self.norm(x)

        # if flattening occurred, unflatten
        if self.n_dim >= 4:
            x = x.reshape(size)

        assert x.shape == size
        return x
