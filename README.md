# evo-hidden

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Hidden-state extraction API for the [Evo](https://github.com/evo-design/evo) genomic foundation model (StripedHyena architecture).

Evo is a powerful DNA sequence model — but its original implementation only exposes next-token logits. **evo-hidden** adds a backward-compatible hidden-state extraction interface so you can use Evo as an encoder for representation learning, probing, and downstream classification tasks.

---

## Why

When using Evo for tasks like antimicrobial resistance prediction, protein function classification, or any sequence-level labeling, you need the intermediate representations — not just the output logits. Different layers capture different levels of biological abstraction, and layer selection is itself a research question.

This package lets you do:

```python
# Extract from a specific layer
out = model.extract_layer_hidden_states(input_ids, target_layer=10)
embedding = out["hidden_states"].mean(dim=1)   # (batch, hidden_size)

# Sweep all 32 layers in one forward pass for diagnostic analysis
states = model.extract_multiple_layers(input_ids, list(range(32)))
```

All without modifying any existing generation logic — the original `model(x)` interface is completely unchanged.

---

## Installation

```bash
pip install evo-hidden
```

For GPU inference with flash attention support:

```bash
pip install evo-hidden[flash-attn]
```

Install from source:

```bash
git clone https://github.com/haleyyy2001/evo-hidden
cd evo-hidden
pip install -e .
```

---

## Quick Start

### Load the model

Download Evo weights from [HuggingFace](https://huggingface.co/evo-design/evo-1-8k-base) first.

```python
import torch
from evo_hidden import StripedHyena, StripedHyenaConfig, ByteTokenizer

# Load config
config = StripedHyenaConfig.from_original_config("path/to/evo-1-8k-base/config.json")

# Build model and load weights
model = StripedHyena(config)
model.load_from_split_converted_state_dict("path/to/evo-1-8k-base/")

# Recommended inference setup
model.eval()
model.to_bfloat16_except_poles_residues()
model.precompute_filters(L=8192, device="cuda")
model = model.to("cuda")
```

### Tokenize a DNA sequence

```python
tokenizer = ByteTokenizer()
sequence = "ATGCTTGACCGAATGCTTGACCGA"
input_ids = tokenizer(sequence, return_tensors="pt").input_ids.to("cuda")
```

### Extract hidden states from a single layer

```python
out = model.extract_layer_hidden_states(input_ids, target_layer=10)

hidden = out["hidden_states"]  # (1, seq_len, hidden_size)
pooled = hidden.mean(dim=1)    # (1, hidden_size) — global mean pool
```

### Extract from all layers (layer diagnostic sweep)

```python
layer_states = model.extract_multiple_layers(input_ids, list(range(32)))

for layer_idx, state in layer_states.items():
    print(f"Layer {layer_idx}: {state.shape}")
    # Layer 0: torch.Size([1, 24, 4096])
    # Layer 1: torch.Size([1, 24, 4096])
    # ...
```

### HuggingFace-style dict return

```python
# Full output dict
outputs = model(input_ids, return_dict=True)

outputs["logits"]            # (batch, seq_len, vocab_size) — same as original
outputs["hidden_states"]     # list of 32 tensors, one per block
outputs["last_hidden_state"] # (batch, seq_len, hidden_size) — post-norm
outputs["inference_params"]  # None for stateless forward
```

### Original generation — unchanged

```python
# Works exactly as before, no overhead
logits, _ = model(input_ids)
```

---

## API Reference

### `StripedHyena`

Drop-in replacement for the original `StripedHyena` class. All original methods are preserved.

**`forward(x, inference_params_dict=None, padding_mask=None, output_hidden_states=False, return_dict=False)`**

| Parameter | Default | Description |
|---|---|---|
| `output_hidden_states` | `False` | If True, return `(logits, list[hidden_states])` |
| `return_dict` | `False` | If True, return dict with all outputs |

**`extract_layer_hidden_states(input_ids, target_layer, padding_mask=None)`**

Extract from a single block layer (0-indexed). Returns:
```python
{
    "hidden_states": Tensor,   # (batch, seq_len, hidden_size)
    "layer_index":  int,
    "shape":        tuple,
}
```

**`extract_multiple_layers(input_ids, layer_list, padding_mask=None)`**

Extract from multiple layers in one pass. Returns `dict[int, Tensor]`.

---

## Layer Selection

For Evo-1-8k-base, Layer 10 was identified as the optimal extraction point via systematic layer-diagnostic sweeps across all 32 blocks.

The sweep measures per-layer:
- activation magnitude (mean L2 norm)
- isotropic angular diversity
- effective rank of the activation covariance
- token-norm concentration
- cross-seed stability

A sharp stability boundary appears at Layer 11. Layer 10 is the deepest jointly stable layer, making it the recommended default for downstream tasks.

See the [layer sweep example](examples/layer_sweep.py) to reproduce this analysis on your own data.

---

## Architecture

StripedHyena is a hybrid SSM + attention architecture (Hyena operators interleaved with multi-head attention). Key properties relevant to extraction:

- 32 blocks (Hyena or attention)
- `hidden_states[k]` = activations **after** block `k` (0-indexed)
- `last_hidden_state` = post-RMSNorm, used for next-token prediction
- Tokenizes raw DNA at single-nucleotide resolution (no protein translation)
- 8 192-token context window for `evo-1-8k-base`

---

## Examples

| Script | Description |
|---|---|
| [`examples/extract_layer10.py`](examples/extract_layer10.py) | Extract Layer 10 embeddings for a list of sequences |
| [`examples/layer_sweep.py`](examples/layer_sweep.py) | Sweep all 32 layers and compute activation statistics |

---

## Related Work

This package was developed for:

**Cross-Species Antimicrobial Resistance Prediction from Genomic Foundation Models**  
Huilin Tai, Columbia University, 2025

The Evo-AMR project used Layer 10 embeddings with MiniRocket aggregation for cross-species AMR classification.  
→ [Evo-AMR repository](https://github.com/haleyyy2001/Evo-Amr)

The original Evo model:  
→ [evo-design/evo](https://github.com/evo-design/evo)  
→ Nguyen et al., *Science* 2024: [doi:10.1126/science.ado9336](https://www.science.org/doi/10.1126/science.ado9336)

---

## License

Apache 2.0 — same as the original Evo/StripedHyena codebase.

The original model architecture and utility files are Copyright (c) Together Computer,
distributed under the Apache License 2.0.  
The hidden-state extraction extension (`model.py` additions) is Copyright (c) Huilin Tai, 2025.

---

## Citation

If you use this package, please cite both the original Evo paper and this work:

```bibtex
@article{nguyen2024sequence,
  author  = {Eric Nguyen and Michael Poli and Matthew G. Durrant and others},
  title   = {Sequence modeling and design from molecular to genome scale with Evo},
  journal = {Science},
  volume  = {386},
  number  = {6723},
  year    = {2024},
  doi     = {10.1126/science.ado9336},
}

@misc{tai2025evohidden,
  author = {Huilin Tai},
  title  = {evo-hidden: Hidden-State Extraction for the Evo Genomic Foundation Model},
  year   = {2025},
  url    = {https://github.com/haleyyy2001/evo-hidden},
}
```
