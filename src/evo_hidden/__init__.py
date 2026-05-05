"""evo-hidden: hidden-state extraction for the Evo genomic foundation model.

This package extends StripedHyena (the architecture behind Evo) with a
backward-compatible hidden-state extraction API, enabling representation
learning and layer-diagnostic sweeps without modifying any existing
generation or inference functionality.

Quick start
-----------
>>> from evo_hidden import StripedHyena, StripedHyenaConfig, ByteTokenizer
>>>
>>> config = StripedHyenaConfig.from_original_config("path/to/config.json")
>>> model = StripedHyena(config)
>>> model.to_bfloat16_except_poles_residues()
>>>
>>> tokenizer = ByteTokenizer()
>>> ids = tokenizer("ATGCATGC...", return_tensors="pt").input_ids
>>>
>>> # Single layer
>>> out = model.extract_layer_hidden_states(ids, target_layer=10)
>>> embedding = out["hidden_states"].mean(dim=1)   # (1, hidden_size)
>>>
>>> # All layers (diagnostic sweep)
>>> layer_states = model.extract_multiple_layers(ids, list(range(32)))
"""

from .model import StripedHyena
from .configuration_hyena import StripedHyenaConfig
from .tokenizer import ByteTokenizer

__version__ = "0.1.0"
__author__ = "Huilin Tai"
__license__ = "Apache-2.0"

__all__ = [
    "StripedHyena",
    "StripedHyenaConfig",
    "ByteTokenizer",
]
