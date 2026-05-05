"""Extract Layer 10 embeddings from Evo for a list of DNA sequences.

Usage
-----
    python examples/extract_layer10.py \
        --model_dir path/to/evo-1-8k-base \
        --sequences ATGCTTGAC GCTAGCTAGC \
        --layer 10

Requires GPU and Evo model weights. See the README for download instructions.
"""

import argparse
import torch
from evo_hidden import StripedHyena, StripedHyenaConfig, ByteTokenizer


def load_model(model_dir: str, device: str = "cuda"):
    config = StripedHyenaConfig.from_original_config(f"{model_dir}/config.json")
    model = StripedHyena(config)
    model.load_from_split_converted_state_dict(model_dir)
    model.eval()
    model.to_bfloat16_except_poles_residues()
    model.precompute_filters(L=8192, device=device)
    return model.to(device)


def extract(model, tokenizer, sequences, layer, device):
    results = []
    for seq in sequences:
        ids = tokenizer(seq, return_tensors="pt").input_ids.to(device)
        out = model.extract_layer_hidden_states(ids, target_layer=layer)
        hidden = out["hidden_states"]           # (1, seq_len, hidden_size)
        pooled = hidden.mean(dim=1).squeeze(0)  # (hidden_size,)
        results.append({"sequence": seq[:20] + "...", "pooled_shape": tuple(pooled.shape), "pooled": pooled})
        print(f"  seq={seq[:20]}...  layer={layer}  shape={hidden.shape}  pooled={pooled.shape}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract Evo hidden states")
    parser.add_argument("--model_dir", required=True, help="Path to model directory")
    parser.add_argument("--sequences", nargs="+", required=True, help="DNA sequences")
    parser.add_argument("--layer", type=int, default=10, help="Block layer to extract (0-indexed)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir} ...")
    model = load_model(args.model_dir, args.device)

    tokenizer = ByteTokenizer()
    print(f"\nExtracting Layer {args.layer} hidden states:")
    extract(model, tokenizer, args.sequences, args.layer, args.device)


if __name__ == "__main__":
    main()
