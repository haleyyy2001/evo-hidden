"""Sweep all 32 Evo layers and compute activation statistics.

This reproduces the layer-diagnostic methodology used to identify Layer 10 as
the optimal extraction depth for Evo-1-8k-base.

Metrics computed per layer
--------------------------
- mean_norm:      average L2 norm of hidden states across tokens
- std_norm:       standard deviation of per-token norms
- effective_rank: approximate rank of the activation covariance
- angular_div:    cosine diversity (1 = perfectly isotropic, 0 = collapsed)

Usage
-----
    python examples/layer_sweep.py \
        --model_dir path/to/evo-1-8k-base \
        --sequence ATGCTTGACCGAATGCTTGACCGAATGCTTGACCGA

Requires GPU and Evo model weights.
"""

import argparse
import torch
from evo_hidden import StripedHyena, StripedHyenaConfig, ByteTokenizer


def effective_rank(h: torch.Tensor) -> float:
    """Compute the effective rank of a (seq_len, hidden_size) activation matrix."""
    h_f32 = h.float()
    cov = h_f32.T @ h_f32 / h_f32.shape[0]
    try:
        sv = torch.linalg.svdvals(cov)
    except Exception:
        return float("nan")
    sv = sv[sv > 0]
    p = sv / sv.sum()
    return float((-p * p.log()).sum().exp())


def angular_diversity(h: torch.Tensor) -> float:
    """Fraction of cosine similarity pairs that are < 0.9 (rough isotropy proxy)."""
    h_f32 = h.float()
    norms = h_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    h_norm = h_f32 / norms
    n = min(h_norm.shape[0], 64)
    sample = h_norm[:n]
    sim = sample @ sample.T
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    diverse = (sim[mask].abs() < 0.9).float().mean()
    return float(diverse)


def main():
    parser = argparse.ArgumentParser(description="Evo layer diagnostic sweep")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sequence", required=True, help="A single DNA sequence to probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir} ...")
    config = StripedHyenaConfig.from_original_config(f"{args.model_dir}/config.json")
    model = StripedHyena(config)
    model.load_from_split_converted_state_dict(args.model_dir)
    model.eval()
    model.to_bfloat16_except_poles_residues()
    model.precompute_filters(L=8192, device=args.device)
    model = model.to(args.device)

    tokenizer = ByteTokenizer()
    ids = tokenizer(args.sequence, return_tensors="pt").input_ids.to(args.device)

    print(f"\nSequence length: {ids.shape[1]} tokens")
    print(f"Sweeping {len(model.blocks)} layers ...\n")
    print(f"{'Layer':>6}  {'mean_norm':>10}  {'std_norm':>9}  {'eff_rank':>9}  {'angular_div':>11}")
    print("-" * 55)

    layer_states = model.extract_multiple_layers(ids, list(range(len(model.blocks))))

    for layer_idx, state in layer_states.items():
        h = state.squeeze(0)  # (seq_len, hidden_size)
        norms = h.float().norm(dim=-1)
        mn = float(norms.mean())
        sn = float(norms.std())
        er = effective_rank(h)
        ad = angular_diversity(h)
        print(f"{layer_idx:>6}  {mn:>10.3f}  {sn:>9.3f}  {er:>9.1f}  {ad:>11.3f}")

    print("\nDone. Look for the stability boundary — layers with a sharp change")
    print("in effective_rank or angular_div mark where the representation shifts.")
    print("The deepest stable layer before that boundary is the recommended extraction point.")


if __name__ == "__main__":
    main()
