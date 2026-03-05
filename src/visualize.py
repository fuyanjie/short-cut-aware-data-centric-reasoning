"""Visualization: generate tables and figures for the paper."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import Config as C

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def format_pct(val):
    if val is None:
        return '-'
    return f'{val*100:.1f}%'


def format_f1(val):
    if val is None:
        return '-'
    return f'{val:.2f}'


def generate_table1(all_results, datasets):
    """Table 1: Overall Method Performance and Baseline Superiority.

    Rows: methods. Columns: Accuracy, Robustness, Reasoning Consistency, Shortcut F1.
    Values are averaged across datasets.
    """
    methods = ['standard_ft', 'self_consistency', 'data_filtering', 'full_method']
    method_names = {
        'standard_ft': 'Standard Fine-Tuning',
        'self_consistency': 'Self-Consistency Decoding',
        'data_filtering': 'Data Filtering',
        'full_method': 'Our Method (Full)',
    }

    lines = []
    lines.append('=' * 90)
    lines.append('Table 1: Overall Method Performance and Baseline Superiority')
    lines.append('=' * 90)
    lines.append(f'{"Method":<30} {"Accuracy":>10} {"Robustness":>12} {"Reasoning":>12} {"SC Det F1":>10}')
    lines.append('-' * 90)

    for method in methods:
        accs, robs, reas, f1s = [], [], [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                accs.append(r['accuracy_clean'])
                robs.append(r['robustness'])
                reas.append(r['reasoning_consistency'])
                if r['shortcut_f1'] is not None:
                    f1s.append(r['shortcut_f1'])

        avg_acc = np.mean(accs) if accs else 0
        avg_rob = np.mean(robs) if robs else 0
        avg_rea = np.mean(reas) if reas else 0
        avg_f1 = np.mean(f1s) if f1s else None

        lines.append(f'{method_names[method]:<30} {format_pct(avg_acc):>10} '
                      f'{format_pct(avg_rob):>12} {format_pct(avg_rea):>12} '
                      f'{format_f1(avg_f1):>10}')

    lines.append('=' * 90)

    # Per-dataset breakdown
    lines.append('\nPer-Dataset Breakdown:')
    for ds_name in datasets:
        lines.append(f'\n  Dataset: {ds_name}')
        lines.append(f'  {"Method":<30} {"Acc Clean":>10} {"Acc Perturb":>12} {"Reasoning":>12}')
        lines.append('  ' + '-' * 70)
        for method in methods:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                lines.append(f'  {method_names[method]:<30} '
                              f'{format_pct(r["accuracy_clean"]):>10} '
                              f'{format_pct(r["robustness"]):>12} '
                              f'{format_pct(r["reasoning_consistency"]):>12}')

    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table1.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_table2(all_results, datasets):
    """Table 2: Contribution of Shortcut-aware Reweighting (Ablation)."""
    methods = ['full_method', 'gs_only', 'reweight_only']
    method_names = {
        'full_method': 'Full Method (Both)',
        'gs_only': 'Gradient Surgery Only',
        'reweight_only': 'Reweighting Only',
    }

    lines = []
    lines.append('=' * 70)
    lines.append('Table 2: Contribution of Shortcut-aware Reweighting')
    lines.append('=' * 70)
    lines.append(f'{"Configuration":<30} {"Accuracy":>12} {"SC Det F1":>12}')
    lines.append('-' * 70)

    for method in methods:
        accs, f1s = [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                accs.append(r['accuracy_clean'])
                if r['shortcut_f1'] is not None:
                    f1s.append(r['shortcut_f1'])

        avg_acc = np.mean(accs) if accs else 0
        avg_f1 = np.mean(f1s) if f1s else None
        lines.append(f'{method_names[method]:<30} {format_pct(avg_acc):>12} {format_f1(avg_f1):>12}')

    lines.append('=' * 70)
    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table2.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_table3(all_results, datasets):
    """Table 3: Contribution of Gradient Surgery."""
    methods = ['standard_ft', 'gs_only', 'reweight_only', 'full_method']
    method_names = {
        'standard_ft': 'Standard FT (Baseline)',
        'gs_only': 'Gradient Surgery Only',
        'reweight_only': 'Reweighting Only',
        'full_method': 'Full Method (Both)',
    }

    lines = []
    lines.append('=' * 80)
    lines.append('Table 3: Contribution of Gradient Surgery')
    lines.append('=' * 80)
    lines.append(f'{"Configuration":<30} {"Accuracy":>10} {"Robustness":>12} {"Grad Align":>12}')
    lines.append('-' * 80)

    for method in methods:
        accs, robs, aligns = [], [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                accs.append(r['accuracy_clean'])
                robs.append(r['robustness'])
                if r.get('gradient_alignment') is not None:
                    aligns.append(r['gradient_alignment'])

        avg_acc = np.mean(accs) if accs else 0
        avg_rob = np.mean(robs) if robs else 0
        avg_align = np.mean(aligns) if aligns else None

        align_str = f'{avg_align:.2f}' if avg_align is not None else '-'
        lines.append(f'{method_names[method]:<30} {format_pct(avg_acc):>10} '
                      f'{format_pct(avg_rob):>12} {align_str:>12}')

    lines.append('=' * 80)
    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table3.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_table4(all_results, datasets):
    """Table 4: Ablation Studies - Component Contributions."""
    lines = []
    lines.append('=' * 70)
    lines.append('Table 4: Ablation Studies - Component Contributions')
    lines.append('=' * 70)

    # Compute drops relative to full method
    full_accs, full_robs = [], []
    gs_accs, gs_robs = [], []
    rw_accs, rw_robs = [], []

    for ds_name in datasets:
        if (ds_name, 'full_method') in all_results:
            full_accs.append(all_results[(ds_name, 'full_method')]['accuracy_clean'])
            full_robs.append(all_results[(ds_name, 'full_method')]['robustness'])
        if (ds_name, 'gs_only') in all_results:
            gs_accs.append(all_results[(ds_name, 'gs_only')]['accuracy_clean'])
            gs_robs.append(all_results[(ds_name, 'gs_only')]['robustness'])
        if (ds_name, 'reweight_only') in all_results:
            rw_accs.append(all_results[(ds_name, 'reweight_only')]['accuracy_clean'])
            rw_robs.append(all_results[(ds_name, 'reweight_only')]['robustness'])

    full_acc = np.mean(full_accs) if full_accs else 0
    full_rob = np.mean(full_robs) if full_robs else 0

    lines.append(f'{"Component Removed":<35} {"Acc Drop":>12} {"Rob Drop":>12}')
    lines.append('-' * 70)

    if gs_accs:
        acc_drop = full_acc - np.mean(gs_accs)
        rob_drop = full_rob - np.mean(gs_robs)
        lines.append(f'{"Remove Reweighting (GS only)":<35} {format_pct(acc_drop):>12} {format_pct(rob_drop):>12}')

    if rw_accs:
        acc_drop = full_acc - np.mean(rw_accs)
        rob_drop = full_rob - np.mean(rw_robs)
        lines.append(f'{"Remove Grad Surgery (RW only)":<35} {format_pct(acc_drop):>12} {format_pct(rob_drop):>12}')

    lines.append(f'\nFull Method Accuracy: {format_pct(full_acc)}')
    lines.append(f'Full Method Robustness: {format_pct(full_rob)}')
    lines.append('=' * 70)

    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table4.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_figure3(collected_data, all_results, datasets):
    """Figure 3: Empirical Validation of Theoretical Claims.

    Panel (a): ShortcutScore vs Performance Degradation (correlation)
    Panel (b): Average batch ShortcutScore vs Gradient Alignment change
    Panel (c): Reweighting function visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel (a): ShortcutScore vs Is-Shortcut (proxy for performance degradation)
    if collected_data.get('scores') and collected_data.get('is_shortcut'):
        scores = np.array(collected_data['scores'])
        is_sc = np.array(collected_data['is_shortcut'])

        # Group by ShortcutScore bins
        n_bins = 20
        bins = np.linspace(scores.min(), scores.max(), n_bins + 1)
        bin_centers = []
        bin_shortcut_rates = []
        for i in range(n_bins):
            mask = (scores >= bins[i]) & (scores < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_shortcut_rates.append(is_sc[mask].mean())

        axes[0].scatter(bin_centers, bin_shortcut_rates, alpha=0.8, s=40, color='steelblue')
        if len(bin_centers) > 1:
            z = np.polyfit(bin_centers, bin_shortcut_rates, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(bin_centers), max(bin_centers), 100)
            axes[0].plot(x_line, p(x_line), 'r--', alpha=0.7)
            # Compute correlation
            r = np.corrcoef(bin_centers, bin_shortcut_rates)[0, 1]
            axes[0].set_title(f'(a) ShortcutScore vs Shortcut Rate\n(r={r:.2f})')
        else:
            axes[0].set_title('(a) ShortcutScore vs Shortcut Rate')
        axes[0].set_xlabel('ShortcutScore S(s)')
        axes[0].set_ylabel('Shortcut Sample Rate')

    # Panel (b): Alignment distribution for shortcut vs non-shortcut
    if collected_data.get('alignments') and collected_data.get('is_shortcut'):
        alignments = np.array(collected_data['alignments'])
        is_sc = np.array(collected_data['is_shortcut'])

        sc_align = alignments[is_sc > 0.5]
        nsc_align = alignments[is_sc < 0.5]

        axes[1].hist(sc_align, bins=30, alpha=0.6, label='Shortcut', color='red', density=True)
        axes[1].hist(nsc_align, bins=30, alpha=0.6, label='Non-shortcut', color='green', density=True)
        axes[1].set_xlabel('Gradient Alignment A(s)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('(b) Alignment Distribution')
        axes[1].legend()

    # Panel (c): Reweighting function
    S_range = np.linspace(0, 3, 100)
    weights = np.exp(-C.lambda_ * S_range)
    axes[2].plot(S_range, weights, 'b-', linewidth=2)
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(x=C.tau_A, color='orange', linestyle='--', alpha=0.5, label=f'τ_A={C.tau_A}')
    axes[2].set_xlabel('ShortcutScore S(s)')
    axes[2].set_ylabel('Sample Weight w(s)')
    axes[2].set_title(f'(c) Reweighting Function (λ={C.lambda_})')
    axes[2].legend()

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figure3.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 3 saved to {fig_path}')
    return fig_path


def generate_training_curves(training_logs, save_name='training_curves.png'):
    """Plot training loss curves for different methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, losses in training_logs.items():
        ax.plot(losses, label=method_name, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {fig_path}')
    return fig_path


def generate_summary_bar_chart(all_results, datasets):
    """Bar chart comparing methods across accuracy and robustness."""
    methods = ['standard_ft', 'self_consistency', 'data_filtering', 'full_method']
    method_labels = ['Standard FT', 'Self-Consistency', 'Data Filtering', 'Our Method']

    avg_clean = []
    avg_perturbed = []
    for method in methods:
        cleans, perturbs = [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                cleans.append(all_results[key]['accuracy_clean'])
                perturbs.append(all_results[key]['robustness'])
        avg_clean.append(np.mean(cleans) if cleans else 0)
        avg_perturbed.append(np.mean(perturbs) if perturbs else 0)

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, [v*100 for v in avg_clean], width, label='Clean Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, [v*100 for v in avg_perturbed], width, label='Perturbed Accuracy (Robustness)', color='coral')

    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Method Comparison: Clean vs Perturbed Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    fig_path = os.path.join(RESULTS_DIR, 'comparison_bar_chart.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved to {fig_path}')
    return fig_path
