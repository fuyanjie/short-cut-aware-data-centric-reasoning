#!/usr/bin/env python3
"""Extract raw data for Figure 3 panels and save as text for LaTeX tables."""
import os, sys, json, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config as C
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model
from src.trainer import train_our_method


def set_seed(seed=C.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    # Train our method on all datasets and collect scores
    merged = {'scores': [], 'is_shortcut': [], 'alignments': [], 'concentrations': []}

    for ds_name, ds in datasets.items():
        print(f'Training: {ds_name}...')
        set_seed()
        model = create_model()
        model, collected = train_our_method(
            model, ds, use_reweighting=True, use_gradient_surgery=True,
            collect_scores=True
        )
        for key in merged:
            merged[key].extend(collected[key])
        del model
        print(f'  Collected {len(collected["scores"])} samples')

    scores = np.array(merged['scores'])
    is_sc = np.array(merged['is_shortcut'])
    alignments = np.array(merged['alignments'])
    concentrations = np.array(merged['concentrations'])

    print(f'\nTotal samples: {len(scores)}')
    print(f'Score range: [{scores.min():.4f}, {scores.max():.4f}]')
    print(f'Alignment range: [{alignments.min():.4f}, {alignments.max():.4f}]')

    # ======= Panel (a): ShortcutScore bins vs Shortcut Rate =======
    n_bins = 10  # Use 10 bins for a clean table
    bins = np.linspace(scores.min(), scores.max(), n_bins + 1)
    panel_a_data = []
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (scores >= bins[i]) & (scores <= bins[i + 1])
        if mask.sum() > 0:
            panel_a_data.append({
                'bin_low': float(bins[i]),
                'bin_high': float(bins[i + 1]),
                'center': float((bins[i] + bins[i + 1]) / 2),
                'count': int(mask.sum()),
                'shortcut_rate': float(is_sc[mask].mean()),
                'mean_score': float(scores[mask].mean()),
            })

    # Correlation
    bin_centers = [d['center'] for d in panel_a_data]
    bin_rates = [d['shortcut_rate'] for d in panel_a_data]
    if len(bin_centers) > 1:
        r = np.corrcoef(bin_centers, bin_rates)[0, 1]
    else:
        r = 0.0

    print(f'\nPanel (a): Pearson r = {r:.2f}')
    print(f'{"Bin Range":<20} {"Count":>6} {"SC Rate":>10}')
    for d in panel_a_data:
        print(f'[{d["bin_low"]:.3f}, {d["bin_high"]:.3f}]  {d["count"]:>5}  {d["shortcut_rate"]:.3f}')

    # ======= Panel (b): Alignment statistics =======
    sc_align = alignments[is_sc > 0.5]
    nsc_align = alignments[is_sc < 0.5]

    panel_b_data = {
        'shortcut': {
            'count': int(len(sc_align)),
            'mean': float(sc_align.mean()),
            'std': float(sc_align.std()),
            'min': float(sc_align.min()),
            'q25': float(np.percentile(sc_align, 25)),
            'median': float(np.median(sc_align)),
            'q75': float(np.percentile(sc_align, 75)),
            'max': float(sc_align.max()),
        },
        'non_shortcut': {
            'count': int(len(nsc_align)),
            'mean': float(nsc_align.mean()),
            'std': float(nsc_align.std()),
            'min': float(nsc_align.min()),
            'q25': float(np.percentile(nsc_align, 25)),
            'median': float(np.median(nsc_align)),
            'q75': float(np.percentile(nsc_align, 75)),
            'max': float(nsc_align.max()),
        },
    }

    print(f'\nPanel (b): Gradient Alignment Statistics')
    for group, stats in panel_b_data.items():
        print(f'  {group}: n={stats["count"]}, mean={stats["mean"]:.4f}, '
              f'std={stats["std"]:.4f}, median={stats["median"]:.4f}')

    # Also compute histogram-like bin data for alignment
    align_bins = np.linspace(min(alignments.min(), -1.0), max(alignments.max(), 1.0), 11)
    panel_b_hist = []
    for i in range(len(align_bins) - 1):
        lo, hi = align_bins[i], align_bins[i + 1]
        if i == len(align_bins) - 2:
            sc_mask = (sc_align >= lo) & (sc_align <= hi)
            nsc_mask = (nsc_align >= lo) & (nsc_align <= hi)
        else:
            sc_mask = (sc_align >= lo) & (sc_align < hi)
            nsc_mask = (nsc_align >= lo) & (nsc_align < hi)
        sc_pct = sc_mask.sum() / len(sc_align) * 100 if len(sc_align) > 0 else 0
        nsc_pct = nsc_mask.sum() / len(nsc_align) * 100 if len(nsc_align) > 0 else 0
        panel_b_hist.append({
            'bin_low': float(lo),
            'bin_high': float(hi),
            'sc_count': int(sc_mask.sum()),
            'sc_pct': float(sc_pct),
            'nsc_count': int(nsc_mask.sum()),
            'nsc_pct': float(nsc_pct),
        })

    # ======= Panel (c): Reweighting function =======
    panel_c_data = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        w = np.exp(-C.lambda_ * s)
        panel_c_data.append({'score': s, 'weight': float(w)})

    print(f'\nPanel (c): Reweighting Function w(s) = exp(-{C.lambda_} * s)')
    for d in panel_c_data:
        print(f'  S(s)={d["score"]:.2f} -> w={d["weight"]:.4f}')

    # Save all data as JSON
    output = {
        'panel_a': {'bins': panel_a_data, 'pearson_r': r},
        'panel_b': {'stats': panel_b_data, 'histogram': panel_b_hist},
        'panel_c': {'values': panel_c_data, 'lambda': C.lambda_},
    }

    out_path = os.path.join('results', 'figure3_data.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nData saved to {out_path}')


if __name__ == '__main__':
    main()
