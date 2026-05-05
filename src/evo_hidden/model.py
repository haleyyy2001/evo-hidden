# Copyright (c) Together Computer
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli
#
# Hidden-state extraction extension by Huilin Tai (Columbia University, 2025)
# Developed for cross-species AMR prediction research using Evo genomic embeddings.
# Source: https://github.com/haleyyy2001/evo-hidden

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import InferenceParams, RecurrentInferenceParams
from .engine import HyenaInferenceEngine
from .layers import ParallelGatedMLP, RMSNorm, VocabParallelEmbedding
from .utils import column_split, print_rank_0

try:
    from flash_attn.modules.mha import MHA
except ImportError:
    MHA = None

try:
    from .positional_embeddings import swap_mha_rope
except ImportError:
    swap_mha_rope = None

from .tokenizer import ByteTokenizer  # bundled so HuggingFace serialises the tokenizer


class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.pre_norm, self.post_norm = RMSNorm(config), RMSNorm(config)
        self.layer_idx = layer_idx
        self.proj_groups = config.get("proj_groups", 1)
        dtype = config.get("attn_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads

        self.counter = 0
        self.inner_mha_cls = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_attention_heads // self.proj_groups,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            qkv_proj_bias=config.get("qkv_proj_bias", True),
            rotary_emb_base=config.get("rotary_emb_base", 10000),
            causal=True,
            layer_idx=layer_idx,
            out_proj_bias=config.get("mha_out_proj_bias", True),
            use_flash_attn=self.config.use_flash_attn,
        ).to(dtype=dtype)

        if config.get("use_interpolated_rotary_pos_emb", False) and swap_mha_rope is not None:
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={"scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)},
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        self.inner_mha_cls.rotary_emb.register_buffer(
            "inv_freq", self.inner_mha_cls.rotary_emb.inv_freq
        )

        self.mlp = ParallelGatedMLP(config).to(dtype=mlp_dtype)

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]
        u = self.inner_mha_cls(self.pre_norm(u), inference_params=inference_params) + u
        if type(padding_mask) == torch.Tensor:
            u = u * padding_mask[..., None]
        u = self.mlp(self.post_norm(u)) + u
        return u, None


class ParallelHyenaFilter(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hyena_filter_groups = config.get("hyena_filter_groups", self.config.hidden_size)

        self.use_flashfft = config.get("use_flashfft", False)
        self.state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.inference_mode = config.get("inference_mode", True)
        self.counter = 0
        self.column_split_hyena = config.get("column_split_hyena", True)

        assert self.hidden_size % self.num_filters == 0 and self.num_filters <= self.hidden_size

        self.D = nn.Parameter(torch.zeros(self.hidden_size))

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads

        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = nn.Parameter(
            torch.randn(3 * config.hidden_size, 1, config.short_filter_length)
        )
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * config.hidden_size)) if config.short_filter_bias else None
        )

        self.engine = HyenaInferenceEngine(layer_idx=layer_idx)
        self.use_flash_depthwise = config.get("use_flash_depthwise", False)
        self.data_dtype = None

        if self.use_flash_depthwise:
            self.fir_fn = FlashDepthwiseConv1d(
                channels=3 * self.hidden_size,
                kernel_size=self.short_filter_length,
                padding=self.short_filter_length - 1,
                weights=self.short_filter_weight,
                bias=self.short_filter_bias,
                device=None,
                dtype=self.config.get("depthwise_dtype", torch.bfloat16),
            )
        else:
            self.fir_fn = F.conv1d

        self.fftconv_fn = None
        self.long_fir_threshold = config.get("long_fir_threshold", None)
        if self.long_fir_threshold is not None:
            assert self.use_flashfft is False

        self.num_systems = self.hidden_size // self.hyena_filter_groups

        poles = torch.randn(self.num_systems, self.state_size, 1, 2)
        poles[..., 0] = 1e-2 * torch.randn(self.num_systems, self.state_size, 1)
        poles[..., 1] = 1e-3 * torch.randn(self.num_systems, self.state_size, 1)

        self.poles = nn.Parameter(poles)
        self.residues = nn.Parameter(torch.randn(self.num_systems, self.state_size, 1, 2))
        self.h = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
            return self.sequential_forward(u, inference_params)
        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
        )
        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.h is None:
            h, filter_dtype, poles, residues = self.compute_filter(L, u.device)
        else:
            h = self.h
            filter_dtype = self.h.dtype

        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 1)

        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )
        y = self.engine.parallel_iir(
            z_pre,
            h,
            self.D,
            L,
            t=self.t,
            poles=self.poles,
            residues=self.residues,
            dims=dims,
            inference_params=inference_params,
            layer_idx=self.layer_idx,
            prefill_style=self.config.get("prefill_style", "fft"),
            use_flashfft=self.use_flashfft,
            fftconv_fn=self.fftconv_fn,
            column_split_hyena=self.column_split_hyena,
            long_fir_threshold=self.long_fir_threshold,
            padding_mask=padding_mask,
        )
        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if self.data_dtype is None:
            self.data_dtype = u.dtype
        if len(u.shape) > 2:
            u = u[:, -1]

        fir_state, iir_state = (
            inference_params.fir_state_dict[self.layer_idx],
            inference_params.state_dict[self.layer_idx],
        )

        z_pre, fir_state = self.engine.step_fir(
            u, fir_state, weight=self.short_filter_weight, bias=self.short_filter_bias
        )
        x2, x1, v = (
            column_split(z_pre, self.num_attention_heads, self.hidden_size_per_attention_head)
            if self.column_split_hyena
            else z_pre.split([self.hidden_size, self.hidden_size, self.hidden_size], dim=1)
        )

        y, iir_state = self.engine.step_iir(
            x2, x1, v, self.D, self.residues, self.poles, iir_state,
            iir_groups=self.hyena_filter_groups,
        )

        inference_params.fir_state_dict[self.layer_idx] = fir_state
        inference_params.state_dict[self.layer_idx] = iir_state
        y = y.to(dtype=self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device):
        if not hasattr(self, "t"):
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues, log_poles = (
            torch.view_as_complex(self.residues.to(filter_dtype)),
            torch.view_as_complex(self.poles.to(filter_dtype)).log(),
        )
        h = (residues * (log_poles * self.t).exp()).real.sum(1)[None]
        return h, filter_dtype, log_poles, residues


class ParallelGatedConvBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.low_mem_mode = config.get("low_mem_mode", False)
        dtype_str = config.get("hyena_block_dtype", "float32")
        dtype = getattr(torch, dtype_str) if isinstance(dtype_str, str) else dtype_str
        mlp_dtype_str = config.get("mlp_dtype", "bfloat16")
        mlp_dtype = getattr(torch, mlp_dtype_str) if isinstance(mlp_dtype_str, str) else mlp_dtype_str
        self.pre_norm = RMSNorm(config).to(dtype=dtype)
        self.post_norm = RMSNorm(config).to(dtype=dtype)
        self.filter = ParallelHyenaFilter(config, layer_idx).to(dtype=dtype)
        self.projections = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_filter_dense = nn.Linear(config.hidden_size, config.hidden_size).to(dtype)
        self.mlp = ParallelGatedMLP(config).to(dtype=mlp_dtype)

        self.proj_norm_fn = self.proj_norm
        self.res_mlp_norm_fn = self.res_mlp_norm

        if self.config.get("compile", False):
            self.proj_norm_fn = torch.compile(
                self.proj_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
            self.res_mlp_norm_fn = torch.compile(
                self.res_mlp_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )

    def proj_norm(self, x):
        return self.projections(self.pre_norm(x))

    def res_mlp_norm(self, x):
        return self.mlp(self.post_norm(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm_fn(u)
        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[..., None]
        z, inference_params = self.filter(z, inference_params=inference_params, padding_mask=padding_mask)
        z_in = self.out_filter_dense(z) + u
        if type(padding_mask) == torch.Tensor:
            z_in = z_in * padding_mask[..., None]
        y = self.res_mlp_norm_fn(z_in)
        return y, inference_params


def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hyena_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", "False"):
            block.filter.fftconv_fn = flash_fft
        return block
    else:
        raise NotImplementedError


class StripedHyena(nn.Module):
    """StripedHyena with hidden-state extraction support.

    This is a backward-compatible extension of the original StripedHyena model.
    All original functionality is preserved — calling ``model(x)`` without the
    new keyword arguments returns the same ``(logits, inference_params)`` tuple
    as the original implementation.

    New capabilities
    ----------------
    ``output_hidden_states : bool``
        When True, ``forward()`` returns ``(logits, list_of_hidden_states)``
        where ``list_of_hidden_states[k]`` is the output of block ``k``,
        shape ``(batch, seq_len, hidden_size)``.

    ``return_dict : bool``
        When True, ``forward()`` returns a dict with keys:
        ``logits``, ``hidden_states``, ``last_hidden_state`` (post-norm),
        and ``inference_params``.

    ``extract_layer_hidden_states(input_ids, target_layer)``
        Convenience method for single-layer extraction.

    ``extract_multiple_layers(input_ids, layer_list)``
        Convenience method for multi-layer extraction in one forward pass.
        Used for systematic layer-diagnostic sweeps.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = VocabParallelEmbedding(config)
        self.norm = RMSNorm(config) if config.get("final_norm", True) else None
        self.unembed = (
            self.embedding_layer if config.tie_embeddings else VocabParallelEmbedding(config)
        )

        if config.get("use_flashfft", "False"):
            try:
                from flashfftconv import FlashFFTConv
            except ImportError:
                raise ImportError("flashfftconv not installed. Run: pip install flashfftconv")
            self.flash_fft = FlashFFTConv(2 * config.seqlen, dtype=torch.bfloat16)
        else:
            self.flash_fft = None

        self.blocks = nn.ModuleList(
            get_block(config, layer_idx, flash_fft=self.flash_fft)
            for layer_idx in range(config.num_layers)
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        inference_params_dict=None,
        padding_mask=None,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ):
        """Run a forward pass with optional hidden-state collection.

        Parameters
        ----------
        x:
            Token IDs, shape ``(batch, seq_len)``.
        inference_params_dict:
            Stateful KV-cache / recurrent state dicts for autoregressive
            generation (same as original).
        padding_mask:
            Boolean mask applied to embeddings and block outputs.
        output_hidden_states:
            If True, collect block outputs and return them alongside logits.
        return_dict:
            If True, return a dict instead of a tuple.  Implies
            ``output_hidden_states=True``.

        Returns
        -------
        Default (both flags False):
            ``(logits, inference_params_dict_out)``  — identical to original.
        ``output_hidden_states=True``:
            ``(logits, list[Tensor])`` — list indexed 0 … num_layers-1.
        ``return_dict=True``:
            ``dict`` with keys ``logits``, ``hidden_states``,
            ``last_hidden_state``, ``inference_params``.
        """
        x = self.embedding_layer.embed(x)

        collect = output_hidden_states or return_dict
        all_hidden_states = [] if collect else None

        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
                all_hidden_states=all_hidden_states,
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(
                x,
                padding_mask=padding_mask,
                all_hidden_states=all_hidden_states,
            )

        x = self.norm(x)
        last_hidden_state = x.clone() if return_dict else None
        x = self.unembed.unembed(x)

        if return_dict:
            return {
                "logits": x,
                "hidden_states": all_hidden_states,
                "last_hidden_state": last_hidden_state,
                "inference_params": inference_params_dict_out,
            }
        if output_hidden_states:
            return x, all_hidden_states
        return x, inference_params_dict_out

    def stateful_forward(self, x, inference_params_dict=None, all_hidden_states=None):
        for block_idx, block in enumerate(self.blocks):
            block_name = "mha" if block_idx in self.config.attn_layer_idxs else "hyena"
            inference_params = inference_params_dict[block_name]
            x, _ = block(x, inference_params=inference_params)
            if all_hidden_states is not None:
                all_hidden_states.append(x.clone())
        return x, inference_params_dict

    def stateless_forward(self, x, padding_mask=None, all_hidden_states=None):
        if type(padding_mask) == torch.Tensor:
            x = x * padding_mask[..., None]
        for _, block in enumerate(self.blocks):
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)
            if all_hidden_states is not None:
                all_hidden_states.append(x.clone())
        return x, None

    # ------------------------------------------------------------------
    # Extraction convenience methods
    # ------------------------------------------------------------------

    def extract_layer_hidden_states(
        self,
        input_ids: torch.Tensor,
        target_layer: int,
        padding_mask=None,
    ) -> dict:
        """Extract hidden states from a specific block layer.

        Parameters
        ----------
        input_ids:
            Tokenised input, shape ``(batch, seq_len)``.
        target_layer:
            0-based block index.  For Evo-1-8k-base (32 blocks), valid range
            is 0–31.  Layer 10 was identified as the optimal extraction depth
            for AMR downstream tasks via layer-diagnostic sweeps.
        padding_mask:
            Optional boolean mask.

        Returns
        -------
        dict:
            ``hidden_states`` — tensor ``(batch, seq_len, hidden_size)``
            ``layer_index``   — the requested index
            ``shape``         — tuple of the tensor shape
        """
        if not (0 <= target_layer < len(self.blocks)):
            raise ValueError(
                f"target_layer {target_layer} is out of range "
                f"[0, {len(self.blocks) - 1}]"
            )
        with torch.no_grad():
            out = self.forward(input_ids, padding_mask=padding_mask, return_dict=True)
        hidden = out["hidden_states"][target_layer]
        return {"hidden_states": hidden, "layer_index": target_layer, "shape": tuple(hidden.shape)}

    def extract_multiple_layers(
        self,
        input_ids: torch.Tensor,
        layer_list: list[int],
        padding_mask=None,
    ) -> dict:
        """Extract hidden states from multiple layers in a single forward pass.

        Parameters
        ----------
        input_ids:
            Tokenised input, shape ``(batch, seq_len)``.
        layer_list:
            Iterable of 0-based block indices.
        padding_mask:
            Optional boolean mask.

        Returns
        -------
        dict mapping layer index → tensor ``(batch, seq_len, hidden_size)``.
        Returns ``None`` for any index that is out of range.
        """
        for layer in layer_list:
            if not (0 <= layer < len(self.blocks)):
                raise ValueError(
                    f"Layer {layer} is out of range [0, {len(self.blocks) - 1}]"
                )
        with torch.no_grad():
            out = self.forward(input_ids, padding_mask=padding_mask, return_dict=True)
        return {
            layer: out["hidden_states"][layer]
            if layer < len(out["hidden_states"])
            else None
            for layer in layer_list
        }

    # ------------------------------------------------------------------
    # Generation / loading utilities (unchanged from original)
    # ------------------------------------------------------------------

    def initialize_inference_params(self):
        print_rank_0("Initializing inference params...")
        return {
            "mha": InferenceParams(
                max_seqlen=self.config.get("max_seqlen", 8192),
                max_batch_size=self.config.get("max_batch_size", 1),
                seqlen_offset=0,
            ),
            "hyena": RecurrentInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                state_dim=self.config.state_size,
                seqlen_offset=0,
            ),
        }

    def precompute_filters(self, L, device):
        for block_idx, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == ParallelHyenaFilter:
                    L = block.filter.long_fir_threshold or L
                    print_rank_0(f"Precomputing filters, L={L}...")
                    filter_dtype = torch.float16 if L >= 2048 else torch.float32
                    block.filter._set_time(L, device)
                    residues, poles = (
                        torch.view_as_complex(block.filter.residues.to(torch.float16)),
                        torch.view_as_complex(block.filter.poles.to(torch.float16)),
                    )
                    block.filter.h = (residues * poles ** block.filter.t).real.sum(1)[None]
                    block.filter.h = block.filter.h.to(dtype=filter_dtype)

    def load_poles_residues(self, path):
        for block_idx, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == ParallelHyenaFilter:
                    print(f"Loading poles and residues for block {block_idx}")
                    poles = torch.load(
                        path + f"/approx_poles_{block_idx+1}.pt", map_location="cpu"
                    )
                    poles = torch.view_as_real(poles)
                    residues = torch.load(
                        path + f"/approx_residues_{block_idx+1}.pt", map_location="cpu"
                    )
                    residues = torch.view_as_real(residues)
                    poles = poles.permute(1, 0, 2).unsqueeze(-2)
                    residues = residues.permute(1, 0, 2).unsqueeze(-2)
                    block.filter.poles = nn.Parameter(poles)
                    block.filter.residues = nn.Parameter(residues)

    def to_bfloat16_except_poles_residues(self):
        """Convert all parameters to bfloat16 except poles and residues.

        Particularly important for longer-context inference.
        """
        for k, p in self.named_parameters():
            if "poles" not in k and "residues" not in k:
                p.data = p.data.to(torch.bfloat16)

    def load_from_split_converted_state_dict(self, path):
        print("Loading from split converted state dict")
        embedding_weight = torch.load(path + "/layer_00.pt")["word_embeddings.weight"]
        self.embedding_layer.weight = nn.Parameter(
            embedding_weight.to(self.embedding_layer.weight.dtype)
        )
        print("Loading embedding weight ok")

        if self.config.get("final_norm", False) is not None:
            idx = len(self.blocks) + 1
            final_norm_scale = torch.load(path + f"/layer_{idx:02d}.pt")["norm.scale"]
            self.norm.scale = nn.Parameter(final_norm_scale.to(self.norm.scale.dtype))
            print("Loading final norm ok")

        if not self.config.get("tie_embeddings", True):
            idx = len(self.blocks) + 2
            embedding_weight = torch.load(path + f"/layer_{idx:02d}.pt")["word_embeddings.weight"]
            self.unembed.weight = nn.Parameter(
                embedding_weight.to(self.unembed.weight.dtype)
            )
            print("Loading unembed weight ok")

        for block_idx, block in enumerate(self.blocks):
            print(f"Loading block {block_idx}...")
            loaded_dict = torch.load(path + f"/layer_{block_idx + 1:02d}.pt")
            block.load_state_dict(loaded_dict, strict=True)
