# LoRA adapters for spectral convolution weights

Implemented in `neuralop/layers/conv_adapter.py`.

## Idea

In LLMs, LoRA adds a low-rank update to a weight matrix, `W' = W₀ + BA`. A Q/K/V projection is
one matrix applied to every token independently, so a single `BA` is shared by all tokens.

A spectral convolution is also pointwise, but in Fourier space: at each retained mode `k` it
applies a channel-mixing matrix `W_k ∈ ℂ^{in × out}`, with no mixing between modes. Unlike the
token case, every mode has its own `W_k`, so the update naturally has one slice `ΔW_k` per mode,
and how parameters are shared across modes is a design choice. Each `SpectralConv` in a model
gets its own independent adapter, just as `W_q`, `W_k` and `W_v` each get their own LoRA.

The number of stored modes is `M = k^{N-1} · (k//2 + 1)` for N-D real data, since the real FFT
keeps only half of the last dimension. For `n_modes=(16, 16)`, `M = 16 × 9 = 144`.

## Coarsening

Coarsening level `n` uses

```
W = W₀ + n · ΔW
```

i.e. the additive update applied recursively `n` times. Because `ΔW` does not depend on `W`,
the effect grows linearly with the level (unlike the MLP/GLU adapters, where `W ← W + f(W)`
compounds). Level 0 is the unchanged base layer.

## Variants

All three subclass `LoRAAdaptableSpectralConv(AdaptableSpectralConv)` and only define
`delta_weight()`, which returns `ΔW` of shape `(in, out, M)`. `forward`, `merge_weights` and
`coarsen_model` are inherited.

| Class | `ΔW_k` | Sharing across modes | Parameters (complex) |
|---|---|---|---|
| `PerModeLoRAAdaptableSpectralConv` (A) | `B_k A_k` | None: independent update per mode | `r · (in + out) · M` |
| `CPLoRAAdaptableSpectralConv` (B) | `B diag(s_k) A` | Shared channel factors, per-mode scale | `r · (in + out + M)` |
| `SharedLoRAAdaptableSpectralConv` (C) | `B A` | Fully shared, same update at every mode | `r · (in + out)` |

Shapes: `A: (r, in)`, `B: (out, r)`, `s: (M, r)`; the per-mode variant has one `A_k`, `B_k`
per mode.

- **Per-mode (A):** the literal translation of LoRA to every `W_k`. Most expressive, but
  nothing is shared between neighbouring frequencies, and it is expensive at large `M`.
- **CP (B):** treats `ΔW` as a 3-way tensor `(out, in, modes)` and writes it as a rank-`r` CP
  decomposition, `ΔW = Σ_r b_r ⊗ a_r ⊗ s_r`. Each component is a channel coupling (read input
  channels along `a_r`, write output channels along `b_r`) with its own learned spectral filter
  `s_r`. In physical space: project onto `a_r` to get a scalar field, convolve it with the filter
  whose Fourier coefficients are `s_r`, and write it back along `b_r`.
- **Shared (C):** one update applied identically at every mode. Equivalent to adding a
  low-pass-filtered low-rank 1×1 convolution, and the closest analogue to LLM LoRA.

In every variant, each mode's update has rank at most `r`; they differ only in how much is
shared across modes. C is B with `s = 1`.

## Implementation notes

- **Rank argument:** `lora_rank` (default 8). It cannot be called `rank`, because `FNOBlocks`
  already passes `rank=` to every conv module for tensor factorization. Use it in an FNO with:
  ```python
  from neuralop.models.fno import partialclass
  conv_module = partialclass("CPLoRA", CPLoRAAdaptableSpectralConv, lora_rank=8)
  ```
- **Initialization:** standard LoRA. `A` is random with std `1/√in`, `B = 0`, `s = 1`, so
  `ΔW = 0` and the adapted layer initially matches the base layer at every level. The CP
  variant starts out equivalent to the shared variant and learns frequency dependence.
- **Complex factors:** all factors use the base weight's complex dtype, so each component can
  carry a phase.
- **Merging is exact:** `merge_weights(n)` sets `W₀ ← W₀ + n·ΔW`, after which level `m` gives
  the unmerged level `n + m`.
- **Base weights are not frozen**, as with the other `AdaptableSpectralConv` adapters. Freeze
  `spectral_conv` in the training script for LoRA-only fine-tuning.
- **Separable weights** raise `NotImplementedError`: each mode's weight is then a vector of
  channels, so there is no matrix to make low-rank.
- `hidden_scale` and `checkpoint_weight_transform` are accepted (inherited) but have no effect.

## Parameter counts

### 32 channels, `n_modes=(16, 16)` (M = 144), dense weights

Base layer: `32 × 32 × 144 = 147,456` complex weights. Counts are complex; double them for real
parameters. Percentages are relative to the base layer.

| `lora_rank` | Per-mode (A) | CP (B) | Shared (C) |
|---|---|---|---|
| 4 | 36,864 (25%) | 832 (0.6%) | 256 (0.2%) |
| 8 | 73,728 (50%) | 1,664 (1.1%) | 512 (0.3%) |
| 16 | 147,456 (100%) | 3,328 (2.3%) | 1,024 (0.7%) |
| 32 (full) | 294,912 (200%) | 6,656 (4.5%) | 2,048 (1.4%) |

Formulas at this size: per-mode `9,216 · r`, CP `208 · r`, shared `64 · r`.

- **CP and shared** stay tiny at every rank, at most 4.5% of the base layer even at full rank.
- **Per-mode** is as large as the base layer by rank 16. At full rank (`r = 32`) it is twice
  the base layer's size and gains nothing over fine-tuning `W` directly, since a full-rank
  `B_k A_k` can represent any `ΔW_k`. At 32 channels, ranks 4–8 are its useful range.

### Reference: 256 channels, `n_modes=(16, 16)`, `lora_rank=8`

| Per-mode (A) | CP (B) | Shared (C) |
|---|---|---|
| 589,824 | 5,248 | 4,096 |
