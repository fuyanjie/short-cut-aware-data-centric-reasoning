#!/usr/bin/env python3
"""
Main experiment runner for:
"Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in LLMs"

Supports:
  DATASET_TYPE=synthetic  python3 run_all.py   # 3 synthetic datasets (default)
  DATASET_TYPE=realworld  python3 run_all.py   # GSM8K + MATH
  DATASET_TYPE=all        python3 run_all.py   # all 5 datasets

All results are output as tables (no figures).
"""
import os
import sys
import time
import copy
import torch
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config as C, PROFILE, DATASET_TYPE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset, get_dataloader)
from src.model import create_model, create_model_nl, count_parameters
from src.trainer import train_standard, train_data_filtering, train_our_method
from src.evaluate import (run_full_evaluation, run_full_evaluation_nl,
                          evaluate_gradient_alignment)
from src.visualize import (generate_table1, generate_table2, generate_table3,
                           generate_table4)


def set_seed(seed=C.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _empty_cache():
    if C.device == 'cuda':
        torch.cuda.empty_cache()
    elif C.device == 'mps':
        torch.mps.empty_cache()


def run_synthetic_experiments(all_results, collected_data_all, dataset_names):
    """Run experiments on 3 synthetic datasets."""
    print('\n--- Generating Synthetic Datasets ---')
    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }
    for name, ds in datasets.items():
        dataset_names.append(name)
        print(f'  {name}: train={len(ds["train"])}, val={len(ds["val"])}, '
              f'test_clean={len(ds["test_clean"])}, test_perturbed={len(ds["test_perturbed"])}')

    # Check model size
    tmp_model = create_model('cpu')
    print(f'Synthetic model parameters: {count_parameters(tmp_model):,}')
    del tmp_model

    # ===================================================================
    # Experiment 1: Overall Method Performance
    # ===================================================================
    print('\n' + '=' * 70)
    print('EXPERIMENT 1: Overall Method Performance (Synthetic)')
    print('=' * 70)

    for ds_name, ds in datasets.items():
        print(f'\n--- Dataset: {ds_name} ---')

        # (a) Standard Fine-Tuning
        print('\n[1/4] Training: Standard Fine-Tuning...')
        set_seed()
        model_ft = create_model()
        t0 = time.time()
        train_standard(model_ft, ds)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_ft = run_full_evaluation(model_ft, ds, compute_f1=False)
        all_results[(ds_name, 'standard_ft')] = results_ft
        print(f'  Accuracy: {results_ft["accuracy_clean"]:.3f}, '
              f'Robustness: {results_ft["robustness"]:.3f}')

        # (b) Self-Consistency Decoding
        print('\n[2/4] Evaluating: Self-Consistency Decoding...')
        results_sc = run_full_evaluation(model_ft, ds, use_self_consistency=True)
        all_results[(ds_name, 'self_consistency')] = results_sc
        print(f'  Accuracy: {results_sc["accuracy_clean"]:.3f}, '
              f'Robustness: {results_sc["robustness"]:.3f}')

        # (c) Data Filtering
        print('\n[3/4] Training: Data Filtering...')
        set_seed()
        model_df = create_model()
        t0 = time.time()
        train_data_filtering(model_df, ds)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_df = run_full_evaluation(model_df, ds, compute_f1=True)
        all_results[(ds_name, 'data_filtering')] = results_df
        print(f'  Accuracy: {results_df["accuracy_clean"]:.3f}, '
              f'Robustness: {results_df["robustness"]:.3f}, '
              f'F1: {results_df["shortcut_f1"]:.3f}')

        # (d) Our Full Method
        print('\n[4/4] Training: Our Method (Reweighting + Gradient Surgery)...')
        set_seed()
        model_ours = create_model()
        t0 = time.time()
        model_ours, collected = train_our_method(
            model_ours, ds, use_reweighting=True, use_gradient_surgery=True,
            collect_scores=True
        )
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_ours = run_full_evaluation(
            model_ours, ds, compute_f1=True, compute_alignment=True
        )
        all_results[(ds_name, 'full_method')] = results_ours
        collected_data_all[ds_name] = collected
        print(f'  Accuracy: {results_ours["accuracy_clean"]:.3f}, '
              f'Robustness: {results_ours["robustness"]:.3f}, '
              f'F1: {results_ours["shortcut_f1"]:.3f}, '
              f'Alignment: {results_ours["gradient_alignment"]:.3f}')

        del model_ft, model_df, model_ours
        _empty_cache()

    # ===================================================================
    # Experiments 2 & 3: Ablation Studies
    # ===================================================================
    print('\n' + '=' * 70)
    print('EXPERIMENTS 2 & 3: Ablation Studies (Synthetic)')
    print('=' * 70)

    for ds_name, ds in datasets.items():
        print(f'\n--- Dataset: {ds_name} ---')

        print('\n[1/2] Training: Reweighting Only...')
        set_seed()
        model_rw = create_model()
        t0 = time.time()
        train_our_method(model_rw, ds, use_reweighting=True, use_gradient_surgery=False)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_rw = run_full_evaluation(
            model_rw, ds, compute_f1=True, compute_alignment=True
        )
        all_results[(ds_name, 'reweight_only')] = results_rw
        print(f'  Accuracy: {results_rw["accuracy_clean"]:.3f}, '
              f'Robustness: {results_rw["robustness"]:.3f}')

        print('\n[2/2] Training: Gradient Surgery Only...')
        set_seed()
        model_gs = create_model()
        t0 = time.time()
        train_our_method(model_gs, ds, use_reweighting=False, use_gradient_surgery=True)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_gs = run_full_evaluation(
            model_gs, ds, compute_f1=True, compute_alignment=True
        )
        all_results[(ds_name, 'gs_only')] = results_gs
        print(f'  Accuracy: {results_gs["accuracy_clean"]:.3f}, '
              f'Robustness: {results_gs["robustness"]:.3f}')

        del model_rw, model_gs
        _empty_cache()

    # Gradient alignment for standard_ft baseline
    print('\n--- Computing gradient alignment for Standard FT baseline ---')
    for ds_name, ds in datasets.items():
        set_seed()
        model_ft_align = create_model()
        train_standard(model_ft_align, ds, verbose=False)
        val_loader = get_dataloader(ds['val'], batch_size=C.batch_size, shuffle=False)
        alignment = evaluate_gradient_alignment(model_ft_align, ds['train'], val_loader)
        all_results[(ds_name, 'standard_ft')]['gradient_alignment'] = alignment
        print(f'  {ds_name} Standard FT alignment: {alignment:.3f}')
        del model_ft_align
        _empty_cache()


def run_realworld_experiments(all_results, collected_data_all, dataset_names):
    """Run experiments on GSM8K and MATH real-world datasets."""
    from transformers import AutoTokenizer
    from src.data_realworld import generate_gsm8k_dataset, generate_math_dataset_realworld

    print('\n--- Initializing NL Environment ---')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # NL training config
    nl_cfg = {
        'batch_size': C.NL.batch_size,
        'lr': C.NL.lr,
        'epochs': C.NL.epochs,
        'weight_decay': C.NL.weight_decay,
        'score_max_samples': C.NL.score_max_samples,
        'score_batch_size': C.NL.score_batch_size,
        'df_warmup_epochs': C.NL.df_warmup_epochs,
        'df_confidence_threshold': C.NL.df_confidence_threshold,
    }

    # Check NL model size
    tmp_model = create_model_nl('cpu')
    print(f'NL model parameters: {count_parameters(tmp_model):,}')
    print(f'NL config: d={C.NL.d_model}, layers={C.NL.num_layers}, '
          f'heads={C.NL.nhead}, ff={C.NL.d_ff}')
    print(f'NL training: bs={C.NL.batch_size}, lr={C.NL.lr}, epochs={C.NL.epochs}')
    del tmp_model

    print('\n--- Generating Real-World Datasets ---')
    nl_datasets = {
        'GSM8K': generate_gsm8k_dataset(tokenizer, seed=42),
        'MATH': generate_math_dataset_realworld(tokenizer, seed=43),
    }
    for name, ds in nl_datasets.items():
        dataset_names.append(name)
        print(f'  {name}: train={len(ds["train"])}, val={len(ds["val"])}, '
              f'test_clean={len(ds["test_clean"])}, test_perturbed={len(ds["test_perturbed"])}')

    # ===================================================================
    # Experiment 1: Overall Method Performance (Real-World)
    # ===================================================================
    print('\n' + '=' * 70)
    print('EXPERIMENT 1: Overall Method Performance (Real-World)')
    print('=' * 70)

    for ds_name, ds in nl_datasets.items():
        print(f'\n--- Dataset: {ds_name} ---')

        # (a) Standard Fine-Tuning
        print('\n[1/4] Training: Standard Fine-Tuning...')
        set_seed()
        model_ft = create_model_nl()
        t0 = time.time()
        train_standard(model_ft, ds, cfg=nl_cfg)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_ft = run_full_evaluation_nl(model_ft, ds, tokenizer, compute_f1=False)
        all_results[(ds_name, 'standard_ft')] = results_ft
        print(f'  Accuracy: {results_ft["accuracy_clean"]:.3f}, '
              f'Robustness: {results_ft["robustness"]:.3f}')

        # (b) Self-Consistency Decoding
        print('\n[2/4] Evaluating: Self-Consistency Decoding...')
        results_sc = run_full_evaluation_nl(model_ft, ds, tokenizer,
                                             use_self_consistency=True)
        all_results[(ds_name, 'self_consistency')] = results_sc
        print(f'  Accuracy: {results_sc["accuracy_clean"]:.3f}, '
              f'Robustness: {results_sc["robustness"]:.3f}')

        # (c) Data Filtering
        print('\n[3/4] Training: Data Filtering...')
        set_seed()
        model_df = create_model_nl()
        t0 = time.time()
        train_data_filtering(model_df, ds, cfg=nl_cfg)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_df = run_full_evaluation_nl(model_df, ds, tokenizer, compute_f1=True)
        all_results[(ds_name, 'data_filtering')] = results_df
        print(f'  Accuracy: {results_df["accuracy_clean"]:.3f}, '
              f'Robustness: {results_df["robustness"]:.3f}, '
              f'F1: {results_df["shortcut_f1"]:.3f}')

        # (d) Our Full Method
        print('\n[4/4] Training: Our Method (Reweighting + Gradient Surgery)...')
        set_seed()
        model_ours = create_model_nl()
        t0 = time.time()
        model_ours, collected = train_our_method(
            model_ours, ds, use_reweighting=True, use_gradient_surgery=True,
            collect_scores=True, cfg=nl_cfg
        )
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_ours = run_full_evaluation_nl(
            model_ours, ds, tokenizer, compute_f1=True, compute_alignment=True
        )
        all_results[(ds_name, 'full_method')] = results_ours
        collected_data_all[ds_name] = collected
        print(f'  Accuracy: {results_ours["accuracy_clean"]:.3f}, '
              f'Robustness: {results_ours["robustness"]:.3f}, '
              f'F1: {results_ours["shortcut_f1"]:.3f}, '
              f'Alignment: {results_ours["gradient_alignment"]:.3f}')

        del model_ft, model_df, model_ours
        _empty_cache()

    # ===================================================================
    # Experiments 2 & 3: Ablation Studies (Real-World)
    # ===================================================================
    print('\n' + '=' * 70)
    print('EXPERIMENTS 2 & 3: Ablation Studies (Real-World)')
    print('=' * 70)

    for ds_name, ds in nl_datasets.items():
        print(f'\n--- Dataset: {ds_name} ---')

        print('\n[1/2] Training: Reweighting Only...')
        set_seed()
        model_rw = create_model_nl()
        t0 = time.time()
        train_our_method(model_rw, ds, use_reweighting=True,
                         use_gradient_surgery=False, cfg=nl_cfg)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_rw = run_full_evaluation_nl(
            model_rw, ds, tokenizer, compute_f1=True, compute_alignment=True
        )
        all_results[(ds_name, 'reweight_only')] = results_rw
        print(f'  Accuracy: {results_rw["accuracy_clean"]:.3f}, '
              f'Robustness: {results_rw["robustness"]:.3f}')

        print('\n[2/2] Training: Gradient Surgery Only...')
        set_seed()
        model_gs = create_model_nl()
        t0 = time.time()
        train_our_method(model_gs, ds, use_reweighting=False,
                         use_gradient_surgery=True, cfg=nl_cfg)
        print(f'  Training time: {time.time()-t0:.1f}s')

        print('  Evaluating...')
        results_gs = run_full_evaluation_nl(
            model_gs, ds, tokenizer, compute_f1=True, compute_alignment=True
        )
        all_results[(ds_name, 'gs_only')] = results_gs
        print(f'  Accuracy: {results_gs["accuracy_clean"]:.3f}, '
              f'Robustness: {results_gs["robustness"]:.3f}')

        del model_rw, model_gs
        _empty_cache()

    # Gradient alignment for standard_ft baseline
    print('\n--- Computing gradient alignment for Standard FT baseline (NL) ---')
    for ds_name, ds in nl_datasets.items():
        set_seed()
        model_ft_align = create_model_nl()
        train_standard(model_ft_align, ds, verbose=False, cfg=nl_cfg)
        val_loader = get_dataloader(ds['val'], batch_size=C.NL.batch_size, shuffle=False)
        alignment = evaluate_gradient_alignment(model_ft_align, ds['train'], val_loader)
        all_results[(ds_name, 'standard_ft')]['gradient_alignment'] = alignment
        print(f'  {ds_name} Standard FT alignment: {alignment:.3f}')
        del model_ft_align
        _empty_cache()


def main():
    print('=' * 70)
    print('Gradient-Aware Shortcut Detection and Correction')
    print('Experimental Evaluation')
    print('=' * 70)
    print(f'Profile: {PROFILE}')
    print(f'Dataset type: {DATASET_TYPE}')
    print(f'Device: {C.device}')
    if C.device == 'cuda':
        n_gpus = torch.cuda.device_count()
        print(f'GPUs: {n_gpus}')
        for i in range(n_gpus):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)} '
                  f'({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
    print(f'Seed: {C.seed}')

    if DATASET_TYPE in ('synthetic', 'all'):
        print(f'\nSynthetic model: d={C.d_model}, layers={C.num_layers}, '
              f'heads={C.nhead}, ff={C.d_ff}')
        print(f'Synthetic data: train={C.n_train}, val={C.n_val}, test={C.n_test}')
        print(f'Synthetic training: bs={C.batch_size}, lr={C.lr}, epochs={C.epochs}')

    dataset_names = []
    all_results = {}
    collected_data_all = {}

    total_start = time.time()

    # ===================================================================
    # Synthetic Experiments
    # ===================================================================
    if DATASET_TYPE in ('synthetic', 'all'):
        run_synthetic_experiments(all_results, collected_data_all, dataset_names)

    # ===================================================================
    # Real-World Experiments
    # ===================================================================
    if DATASET_TYPE in ('realworld', 'all'):
        run_realworld_experiments(all_results, collected_data_all, dataset_names)

    # ===================================================================
    # Generate Results (Tables)
    # ===================================================================
    print('\n' + '=' * 70)
    print('GENERATING RESULTS')
    print('=' * 70)

    print('\n')
    generate_table1(all_results, dataset_names)
    print('\n')
    generate_table2(all_results, dataset_names)
    print('\n')
    generate_table3(all_results, dataset_names)
    print('\n')
    generate_table4(all_results, dataset_names)

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - total_start
    print('\n' + '=' * 70)
    print(f'ALL EXPERIMENTS COMPLETED in {total_time/60:.1f} minutes')
    print(f'Results saved to: {os.path.join(os.path.dirname(__file__), "results")}')
    print('=' * 70)

    # Print key findings
    print('\nKey Findings:')
    ft_accs = [all_results[(ds, 'standard_ft')]['accuracy_clean'] for ds in dataset_names]
    ours_accs = [all_results[(ds, 'full_method')]['accuracy_clean'] for ds in dataset_names]
    ft_robs = [all_results[(ds, 'standard_ft')]['robustness'] for ds in dataset_names]
    ours_robs = [all_results[(ds, 'full_method')]['robustness'] for ds in dataset_names]

    acc_improvement = (np.mean(ours_accs) - np.mean(ft_accs)) * 100
    rob_improvement = (np.mean(ours_robs) - np.mean(ft_robs)) * 100

    df_accs = [all_results[(ds, 'data_filtering')]['accuracy_clean'] for ds in dataset_names]
    acc_over_strong = (np.mean(ours_accs) - np.mean(df_accs)) * 100

    print(f'  Accuracy improvement over Standard FT: {acc_improvement:.1f}%')
    print(f'  Accuracy improvement over Data Filtering: {acc_over_strong:.1f}%')
    print(f'  Robustness improvement: {rob_improvement:.1f}%')

    ours_f1s = [all_results[(ds, 'full_method')]['shortcut_f1']
                for ds in dataset_names if all_results[(ds, 'full_method')]['shortcut_f1'] is not None]
    if ours_f1s:
        print(f'  Average Shortcut Detection F1: {np.mean(ours_f1s):.2f}')

    return all_results


if __name__ == '__main__':
    results = main()
