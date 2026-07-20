1. SpectralConv.forward() — add optional weight override (one line change)

Add override_weight: Optional[torch.Tensor] = None to the signature. At the slicing line, replace weight = self.weight[slices_w] with:
weight = (override_weight if override_weight is not None else self.weight)[slices_w]
This lets the wrapper inject an adapted weight without duplicating any forward logic.

---
2. LoRASpectralConv(nn.Module) — new class in spectral_convolution.py

- Takes an existing SpectralConv and freezes its parameters.
- Trainable V of shape (max_n_modes[0], rank), complex-valued.
- _adapted_weight(): computes W_new by applying V@V^T to the x-mode dimension and V_rfft@V_rfft^T to the y-mode dimension (where V_rfft = V[:max_n_modes[-1], :] handles the rfft slice automatically — same V, sliced). Applied sequentially for efficiency:
W_temp = einsum('kx,ijxy->ijky', V@V^T, W0)
W_new  = einsum('ly,ijky->ijkl', V_rfft@V_rfft^T, W_temp)
- This is O(in_ch × out_ch × n_modes³) rather than O(n_modes⁴).
- forward(): calls spectral_conv.forward(x, output_shape, override_weight=W_new).
- Delegates transform() and the n_modes property/setter to the inner SpectralConv so it's a drop-in replacement inside FNOBlocks.

---
3. inject_lora_spectral_conv(model, rank) — new function in spectral_convolution.py

Recursively walks the module tree via named_children(). When a SpectralConv is found, replaces it in-place with LoRASpectralConv(spectral_conv, rank) via setattr. Works for both regular module attributes and nn.ModuleList entries (since setattr on an nn.Module with a Module value routes through _modules).

---
No changes needed to fno_block.py or fno.py — the wrapper delegates everything FNOBlocks needs.