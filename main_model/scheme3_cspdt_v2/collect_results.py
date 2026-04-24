"""
Unified Results Collection, Statistical Testing, and Comparison Table Generator.

Collects all experimental results (CSP-DT, no_sem_wm, no_sem_policy, no_sem_full, 6 baselines)
across both sigma thresholds, runs statistical significance tests, computes bootstrap CIs,
and generates publication-ready comparison tables.
"""
import os
import json
import numpy as np
from scipy import stats

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = '/home/wangmeiyi/AuctionNet/medical/last_exp/rusults'
OUTPUT_DIR = '/home/wangmeiyi/AuctionNet/medical/last_exp/main_model/scheme3_cspdt_v2/results/tables'

# Model definitions: display_name -> { sigma: relative_path }
MODELS = {
    'CSP-DT (Ours)': {
        'sigma1.0': 'cspdt/stage2_eval.json',
        'sigma2.0': 'cspdt/stage2_sigma2_eval.json',
    },
    'NoSem-WM': {
        'sigma1.0': 'nosemw/stage2_eval.json',
        'sigma2.0': 'nosemw/stage2_sigma2_eval.json',
    },
    'NoSem-Policy': {
        'sigma1.0': 'nosempolicy/stage2_eval.json',
        'sigma2.0': 'nosempolicy/stage2_eval.json',
    },
    'NoSem-Full': {
        'sigma1.0': 'nosemfull/stage2_eval.json',
        'sigma2.0': 'nosemfull/stage2_eval.json',
    },
}

BASELINES = ['bc', 'iql', 'dt', 'cql', 'dqn', 'td3bc']
BASELINE_DISPLAY = {
    'bc': 'BC', 'iql': 'IQL', 'dt': 'DT', 'cql': 'CQL',
    'dqn': 'DQN', 'td3bc': 'TD3BC',
}
SEVERITY_LEVELS = ['overall', 'high', 'mid', 'low']

SIGMA_LABELS = {'sigma1.0': 'σ=1.0', 'sigma2.0': 'σ=2.0'}


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_results(results_dir, sigma, model_path):
    path = os.path.join(results_dir, sigma, model_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  WARNING: Missing {path}")
    return None


def fmt(mean, std, bold=False):
    s = f"{mean:.2f} ± {std:.2f}"
    return f"**{s}**" if bold else s


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for sigma_key, sigma_label in SIGMA_LABELS.items():
        print(f"\n{'='*60}")
        print(f"Processing {sigma_label}")
        print(f"{'='*60}")

        # Load model results
        sigma_results = {}
        for name, paths in MODELS.items():
            data = load_results(RESULTS_DIR, sigma_key, paths[sigma_key])
            if data:
                sigma_results[name] = data

        # Load baseline results
        for bl in BASELINES:
            # sigma1.0 baselines
            if sigma_key == 'sigma1.0':
                data = load_results(RESULTS_DIR, sigma_key, f'baseline/{bl}/stage2_eval.json')
            else:
                data = load_results(RESULTS_DIR, sigma_key, f'baseline/{bl}/stage2_sigma2_eval.json')
            if data:
                sigma_results[BASELINE_DISPLAY[bl]] = data

        all_results[sigma_key] = sigma_results

        if not sigma_results:
            print(f"  No results found for {sigma_label}")
            continue

        # ── 1. Markdown Comparison Table ───────────────────────────────────
        generate_markdown_table(sigma_results, sigma_label)

        # ── 2. LaTeX Table ─────────────────────────────────────────────────
        generate_latex_table(sigma_results, sigma_label)

        # ── 3. Statistical Significance Matrix ─────────────────────────────
        generate_significance_matrix(sigma_results, sigma_label)

        # ── 4. Ablation Table ──────────────────────────────────────────────
        generate_ablation_table(sigma_results, sigma_label)

    # ── 5. Save machine-readable summary ───────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, '..', 'summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")


def generate_markdown_table(sigma_results, sigma_label):
    """Generate Markdown comparison table sorted by overall mean."""
    rows = []
    for name, data in sigma_results.items():
        row = {'model': name}
        for level in SEVERITY_LEVELS:
            if level in data:
                row[f'{level}_mean'] = data[level]['mean']
                row[f'{level}_std'] = data[level]['std']
                row[f'{level}_count'] = data[level].get('count', 0)
        rows.append(row)

    # Sort by overall mean (ascending = lower delta = better)
    rows.sort(key=lambda r: r.get('overall_mean', float('inf')))

    # Find best in each column
    best = {}
    for level in SEVERITY_LEVELS:
        vals = [(r['model'], r.get(f'{level}_mean', float('inf'))) for r in rows if f'{level}_mean' in r]
        if vals:
            best[level] = min(vals, key=lambda x: x[1])[0]

    lines = [
        f"# Main Comparison Table — {sigma_label}",
        "",
        f"| Model | Overall | High | Mid | Low |",
        f"|-------|---------|------|-----|-----|",
    ]
    for r in rows:
        cells = [r['model']]
        for level in SEVERITY_LEVELS:
            if f'{level}_mean' in r:
                is_best = (r['model'] == best.get(level, ''))
                cells.append(fmt(r[f'{level}_mean'], r[f'{level}_std'], bold=is_best))
            else:
                cells.append('—')
        lines.append("| " + " | ".join(cells) + " |")

    outpath = os.path.join(OUTPUT_DIR, f"main_comparison_{sigma_label.replace('=', '')}.md")
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Markdown table -> {outpath}")


def generate_latex_table(sigma_results, sigma_label):
    """Generate LaTeX table."""
    rows = []
    for name, data in sigma_results.items():
        row = {'model': name}
        for level in SEVERITY_LEVELS:
            if level in data:
                row[f'{level}_mean'] = data[level]['mean']
                row[f'{level}_std'] = data[level]['std']
        rows.append(row)

    rows.sort(key=lambda r: r.get('overall_mean', float('inf')))

    # Find best per column
    best = {}
    for level in SEVERITY_LEVELS:
        vals = [(r['model'], r.get(f'{level}_mean', float('inf'))) for r in rows if f'{level}_mean' in r]
        if vals:
            best[level] = min(vals, key=lambda x: x[1])[0]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Treatment policy evaluation: mean $\Delta$SAPS2 (lower is better) — " + sigma_label + "}",
        r"\label{tab:main_results_" + sigma_label.replace('=', '').replace('.', '') + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Overall & High & Mid & Low \\",
        r"\midrule",
    ]

    for r in rows:
        cells = [r['model'].replace('_', r'\_')]
        for level in SEVERITY_LEVELS:
            if f'{level}_mean' in r:
                mean_str = f"{r[f'{level}_mean']:.2f}"
                std_str = f"{r[f'{level}_std']:.2f}"
                if r['model'] == best.get(level, ''):
                    cells.append(f"\\textbf{{{mean_str}}} $\\pm$ {std_str}")
                else:
                    cells.append(f"{mean_str} $\\pm$ {std_str}")
            else:
                cells.append('—')
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    outpath = os.path.join(OUTPUT_DIR, f"main_comparison_{sigma_label.replace('=', '')}.tex")
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  LaTeX table -> {outpath}")


def generate_significance_matrix(sigma_results, sigma_label):
    """Generate p-value matrix: CSP-DT vs each other model (Welch's t-test)."""
    ref_name = 'CSP-DT (Ours)'
    if ref_name not in sigma_results:
        print(f"  Skip significance: {ref_name} not found")
        return

    ref = sigma_results[ref_name]
    lines = [
        f"# Statistical Significance — {sigma_label}",
        f"Reference: {ref_name} vs each model",
        "",
        f"| Comparison | Overall p-value | High p-value | Mid p-value | Low p-value |",
        f"|------------|----------------|-------------|------------|------------|",
    ]

    summary = {}
    for name, data in sigma_results.items():
        if name == ref_name:
            continue
        p_vals = {}
        cells = [name]
        for level in SEVERITY_LEVELS:
            if level in ref and level in data:
                n1, m1, s1 = ref[level]['count'], ref[level]['mean'], ref[level]['std']
                n2, m2, s2 = data[level]['count'], data[level]['mean'], data[level]['std']
                if n1 > 1 and n2 > 1 and s1 > 0 and s2 > 0:
                    t_stat, p_val = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
                    p_vals[level] = p_val
                    if p_val < 0.001:
                        cells.append(f"{p_val:.2e} ***")
                    elif p_val < 0.01:
                        cells.append(f"{p_val:.4f} **")
                    elif p_val < 0.05:
                        cells.append(f"{p_val:.4f} *")
                    else:
                        cells.append(f"{p_val:.4f}")
                else:
                    cells.append("N/A")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
        summary[name] = p_vals

    outpath = os.path.join(OUTPUT_DIR, f"significance_matrix_{sigma_label.replace('=', '')}.md")
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Significance matrix -> {outpath}")


def generate_ablation_table(sigma_results, sigma_label):
    """Generate ablation decomposition table for the 4 SeMDT variants."""
    ablation_models = ['CSP-DT (Ours)', 'NoSem-WM', 'NoSem-Policy', 'NoSem-Full']
    ablation_labels = {
        'CSP-DT (Ours)': 'Full SeMDT (Policy+WM)',
        'NoSem-WM': '− WM Semantic',
        'NoSem-Policy': '− Policy Semantic',
        'NoSem-Full': '− Both (Full ablation)',
    }

    lines = [
        f"# Ablation Decomposition — {sigma_label}",
        "",
        f"| Component | Overall | High | Mid | Low |",
        f"|-----------|---------|------|-----|-----|",
    ]

    full_data = sigma_results.get('CSP-DT (Ours)')

    for model in ablation_models:
        if model not in sigma_results:
            continue
        data = sigma_results[model]
        label = ablation_labels[model]
        cells = [label]

        for level in SEVERITY_LEVELS:
            if level in data:
                mean = data[level]['mean']
                std = data[level]['std']
                cell_str = f"{mean:.2f} ± {std:.2f}"
                # Add delta from full if available
                if full_data and model != 'CSP-DT (Ours)' and level in full_data:
                    delta = mean - full_data[level]['mean']
                    cell_str += f" (Δ{delta:+.2f})"
                cells.append(cell_str)
            else:
                cells.append('—')

        lines.append("| " + " | ".join(cells) + " |")

    outpath = os.path.join(OUTPUT_DIR, f"ablation_table_{sigma_label.replace('=', '')}.md")
    tex_path = os.path.join(OUTPUT_DIR, f"ablation_table_{sigma_label.replace('=', '')}.tex")

    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Ablation table -> {outpath}")

    # LaTeX version
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: semantic pathway contribution — " + sigma_label + "}",
        r"\label{tab:ablation_" + sigma_label.replace('=', '').replace('.', '') + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & Overall & High & Mid & Low \\",
        r"\midrule",
    ]
    for model in ablation_models:
        if model not in sigma_results:
            continue
        data = sigma_results[model]
        label = ablation_labels[model]
        cells = [label]
        for level in SEVERITY_LEVELS:
            if level in data:
                mean = data[level]['mean']
                std = data[level]['std']
                cells.append(f"{mean:.2f} $\\pm$ {std:.2f}")
            else:
                cells.append('—')
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f"  LaTeX ablation table -> {tex_path}")


if __name__ == '__main__':
    main()
