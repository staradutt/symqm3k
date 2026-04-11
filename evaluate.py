# ═══════════════════════════════════════════════════════════════
# evaluate.py
# Load pretrained SymMACE weights and evaluate on the test set.
#
# Usage:
#   python evaluate.py --weights symmace_best.pt
#
# Download weights from https://huggingface.co/datasets/taradutt007/symqm3k or use your own
# trained checkpoint from train.py
# ═══════════════════════════════════════════════════════════════

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

# ── Import model and dataset from train.py ────────────────────
from train import (SymMACE, SymQMDataset, get_atomic_numbers,
                   evaluate, transition_allowed,
                   IRREP_MAP, PG_MAP, PG_IRREP_MASK,
                   summary, atoms_grouped, orbs_grouped,
                   device)

def main(weights_path):
    print(f"Loading weights from: {weights_path}")
    print(f"Device: {device}")

    # Build test loader
    test_dataset = SymQMDataset(
        summary[summary['split']=='test'],
        atoms_grouped, orbs_grouped)
    test_loader  = DataLoader(
        test_dataset, batch_size=32, shuffle=False)
    print(f"Test set: {len(test_dataset)} molecules")

    # Load model
    model = SymMACE().to(device)
    model.load_state_dict(torch.load(
        weights_path,
        map_location=device))
    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # Evaluate
    results = evaluate(model, test_loader, split_name='test')

    # Save
    out_path = 'eval_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print expected vs actual
    print("\n=== Expected Results (from paper) ===")
    expected = {
        'HOMO MAE'          : '0.1528 eV',
        'LUMO MAE'          : '0.1393 eV',
        'HOMO Irrep Acc'    : '76.0%',
        'LUMO Irrep Acc'    : '87.6%',
        'HOMO Irrep Idx MAE': '3.40 positions',
        'LUMO Irrep Idx MAE': '2.05 positions',
        'Transition Acc'    : '76.5%',
    }
    for k, v in expected.items():
        print(f"  {k:<22}: {v}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='symmace_best.pt',
                        help='Path to model weights (.pt file)')
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"Weights file not found: {args.weights}")
        print("Download from [HuggingFace URL] or train with:")
        print("  python train.py")
        exit(1)

    main(args.weights)
