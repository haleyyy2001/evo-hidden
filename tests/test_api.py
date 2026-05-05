"""Dry-run tests for evo-hidden.

These tests validate the API contract without requiring GPU, model weights,
or flash-attention. They use a minimal randomly-initialized model.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

def make_config():
    """Build a tiny StripedHyenaConfig-like dotdict for testing."""
    from evo_hidden.utils import dotdict
    cfg = dotdict({
        "vocab_size": 512,
        "hidden_size": 64,
        "num_filters": 64,
        "inner_mlp_size": 128,
        "attn_layer_idxs": [],
        "hyena_layer_idxs": list(range(2)),
        "num_layers": 2,
        "tie_embeddings": True,
        "short_filter_length": 3,
        "num_attention_heads": 4,
        "proj_groups": 1,
        "hyena_filter_groups": 1,
        "short_filter_bias": True,
        "mha_out_proj_bias": False,
        "qkv_proj_bias": False,
        "final_norm": True,
        "use_cache": True,
        "use_flash_attention_2": False,
        "use_flash_rmsnorm": False,
        "use_flash_depthwise": False,
        "use_flashfft": False,
        "inference_mode": False,
        "prefill_style": "fft",
        "max_seqlen": 128,
        "eps": 1e-5,
        "state_size": 2,
        "rotary_emb_base": 500000,
        "smeared_gqa": False,
        "model_parallel_size": 1,
        "pipe_parallel_size": 1,
        "column_split_hyena": True,
        "use_flash_attn": False,
        "make_vocab_size_divisible_by": 8,
        "log_intermediate_values": False,
        "mlp_dtype": "float32",
    })
    return cfg


# ---------------------------------------------------------------------------
# Import checks (no GPU needed)
# ---------------------------------------------------------------------------

def test_package_imports():
    from evo_hidden import StripedHyena, StripedHyenaConfig, ByteTokenizer  # noqa: F401


def test_version_exists():
    import evo_hidden
    assert hasattr(evo_hidden, "__version__")
    assert evo_hidden.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def test_stripedhyena_instantiates_with_minimal_config():
    from evo_hidden.model import StripedHyena, ParallelGatedConvBlock
    config = make_config()
    model = StripedHyena(config)
    assert len(model.blocks) == 2


def test_stripedhyena_has_extraction_methods():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config)
    assert callable(model.extract_layer_hidden_states)
    assert callable(model.extract_multiple_layers)


# ---------------------------------------------------------------------------
# Forward pass — output_hidden_states flag
# ---------------------------------------------------------------------------

def test_forward_default_returns_tuple(tmp_path):
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    out = model(x)
    assert isinstance(out, tuple)
    assert len(out) == 2
    logits, params = out
    assert logits.shape == (1, 8, 512)
    assert params is None


def test_forward_output_hidden_states_returns_list():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    logits, hidden_states = model(x, output_hidden_states=True)
    assert isinstance(hidden_states, list)
    assert len(hidden_states) == 2   # 2-block toy model
    assert hidden_states[0].shape == (1, 8, 64)


def test_forward_return_dict_keys():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    out = model(x, return_dict=True)
    assert isinstance(out, dict)
    for key in ("logits", "hidden_states", "last_hidden_state", "inference_params"):
        assert key in out
    assert out["logits"].shape == (1, 8, 512)
    assert len(out["hidden_states"]) == 2
    assert out["last_hidden_state"].shape == (1, 8, 64)


def test_forward_default_matches_without_flags():
    """Logits with and without flags should be identical (no side effects)."""
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    logits_plain, _ = model(x)
    logits_dict = model(x, return_dict=True)["logits"]
    assert torch.allclose(logits_plain, logits_dict)


# ---------------------------------------------------------------------------
# Extraction convenience methods
# ---------------------------------------------------------------------------

def test_extract_layer_hidden_states_layer_0():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    out = model.extract_layer_hidden_states(x, target_layer=0)
    assert "hidden_states" in out
    assert out["layer_index"] == 0
    assert out["hidden_states"].shape == (1, 8, 64)
    assert out["shape"] == (1, 8, 64)


def test_extract_layer_hidden_states_out_of_range_raises():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    with pytest.raises(ValueError, match="out of range"):
        model.extract_layer_hidden_states(x, target_layer=99)


def test_extract_multiple_layers_returns_all():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    states = model.extract_multiple_layers(x, layer_list=[0, 1])
    assert set(states.keys()) == {0, 1}
    assert states[0].shape == (1, 8, 64)
    assert states[1].shape == (1, 8, 64)


def test_extract_multiple_layers_bad_index_raises():
    from evo_hidden.model import StripedHyena
    config = make_config()
    model = StripedHyena(config).eval()
    x = torch.randint(0, 512, (1, 8))
    with pytest.raises(ValueError, match="out of range"):
        model.extract_multiple_layers(x, layer_list=[0, 999])


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_byte_tokenizer_encodes_dna():
    from evo_hidden import ByteTokenizer
    tok = ByteTokenizer()
    ids = tok("ATGC", return_tensors="pt").input_ids
    assert ids.shape[1] == 4
    assert ids.dtype == torch.long


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_stripedhyena_config_defaults():
    from evo_hidden import StripedHyenaConfig
    cfg = StripedHyenaConfig()
    assert cfg.num_layers == 32
    assert cfg.hidden_size == 4096
