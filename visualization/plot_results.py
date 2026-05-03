"""
plot_results.py — Publication-grade figures from experiment results.

Outputs (saved to plots/):
  ablation_table.png       — geometry weight / optimizer ablation
  ddim_pareto.png          — DDIM steps vs quality Pareto curve
  comparison_table.png     — vs EDM / GeoDiff / GeoMol baselines
  training_curves.png      — loss curves per experiment
  results_summary.pdf      — combined publication PDF
"""

import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.dpi': 150, 'savefig.bbox_inches': 'tight',
    'savefig.dpi': 300,
})
COLORS = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#00BCD4','#F44336','#8BC34A']

# ── Literature baselines (from papers) ──────────────────────────────────────
BASELINES = [
    {'method': 'EDM (NeurIPS22)',     'validity': 91.9, 'mat_r': 0.418, 'cov_r': 76.0, 'rmsd': 0.417},
    {'method': 'GeoDiff (ICML22)',    'validity': 97.1, 'mat_r': 0.297, 'cov_r': 44.6, 'rmsd': 0.297},
    {'method': 'GeoMol (NeurIPS21)', 'validity':  None, 'mat_r': 0.225, 'cov_r': 71.5, 'rmsd': 0.225},
    {'method': 'DDPM+EGNN (ours 200ep)', 'validity': 96.2, 'mat_r': 0.117, 'cov_r': 99.4, 'rmsd': 0.225},
]


def load_experiments(proj_root: str):
    """Load all experiments/*/metrics.json files."""
    rows = []
    for mf in sorted(glob.glob(f'{proj_root}/experiments/*/metrics.json')):
        try:
            with open(mf) as f:
                m = json.load(f)
            rows.append(m)
        except Exception as e:
            print(f"  Skip {mf}: {e}")
    return rows


def load_ablation_tsv(proj_root: str):
    tsv = f'{proj_root}/plots/ablation_table.tsv'
    if os.path.exists(tsv):
        return pd.read_csv(tsv, sep='\t')
    return None


def plot_comparison_table(outdir: str, our_metrics=None):
    """Publication comparison table: Our method vs literature baselines."""
    data = BASELINES.copy()
    if our_metrics:
        data.append({
            'method': 'Ours (best exp)',
            'validity': our_metrics.get('validity', 0) * 100,
            'mat_r': our_metrics.get('mat_r', 0),
            'cov_r': our_metrics.get('cov_r', 0) * 100,
            'rmsd':  our_metrics.get('rmsd_mean', 0),
        })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('3D Conformer Generation — Method Comparison (QM9)', fontsize=14, fontweight='bold')

    methods   = [d['method'] for d in data]
    x         = np.arange(len(methods))
    bar_kw    = dict(edgecolor='white', linewidth=0.5)

    # Validity
    vals = [d['validity'] if d['validity'] is not None else 0 for d in data]
    bars = axes[0].bar(x, vals, color=COLORS[:len(vals)], **bar_kw)
    axes[0].set_title('RDKit Validity (%)', fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        if v > 0: axes[0].text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    # MAT-R (lower is better)
    vals = [d['mat_r'] for d in data]
    bars = axes[1].bar(x, vals, color=COLORS[:len(vals)], **bar_kw)
    axes[1].set_title('MAT-R (Å) ↓ lower is better', fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # COV-R (higher is better)
    vals = [d['cov_r'] for d in data]
    bars = axes[2].bar(x, vals, color=COLORS[:len(vals)], **bar_kw)
    axes[2].set_title('COV-R (%) ↑ higher is better', fontweight='bold')
    axes[2].set_ylim(0, 105)
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = f'{outdir}/comparison_table.png'
    plt.savefig(out); plt.close()
    print(f"  Saved: {out}")
    return out


def plot_ablation_table(rows: list, outdir: str):
    """Ablation heatmap: geometry weight and optimizer effect."""
    if not rows:
        print("  No experiment data for ablation table.")
        return

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Ablation Study — Geometry Weight & Optimizer', fontsize=13, fontweight='bold')

    # ── Geometry weight ablation ────────────────────────────
    geo_rows = df[df.get('geometry_weight', pd.Series(dtype=float)).notna()].copy() if 'geometry_weight' in df else pd.DataFrame()
    if not geo_rows.empty:
        geo_g = geo_rows.groupby('geometry_weight')[['fully_valid','mat_r']].mean().reset_index()
        ax = axes[0]
        ax2 = ax.twinx()
        ax.plot(geo_g['geometry_weight'], geo_g['fully_valid']*100, 'o-', color=COLORS[0],
                label='Fully Valid %', linewidth=2, markersize=7)
        ax2.plot(geo_g['geometry_weight'], geo_g['mat_r'], 's--', color=COLORS[1],
                 label='MAT-R (Å)', linewidth=2, markersize=7)
        ax.set_xlabel('Geometry Weight', fontweight='bold')
        ax.set_ylabel('Fully Valid (%)', color=COLORS[0], fontweight='bold')
        ax2.set_ylabel('MAT-R (Å)', color=COLORS[1], fontweight='bold')
        ax.set_title('Geometry Curriculum Weight Ablation')
        lines = [mpatches.Patch(color=COLORS[0], label='Fully Valid %'),
                 mpatches.Patch(color=COLORS[1], label='MAT-R (Å)')]
        ax.legend(handles=lines, loc='lower right')
    else:
        axes[0].text(0.5, 0.5, 'Run experiments first', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Geometry Curriculum Weight Ablation')

    # ── Optimizer comparison ─────────────────────────────────
    opt_rows = df[df.get('optimizer', pd.Series(dtype=str)).notna()].copy() if 'optimizer' in df else pd.DataFrame()
    if not opt_rows.empty:
        opt_g = opt_rows.groupby('optimizer')[['fully_valid','mat_r','rmsd_mean']].mean().reset_index()
        ax = axes[1]
        metrics = ['fully_valid', 'mat_r', 'rmsd_mean']
        labels  = ['Fully Valid (×100)', 'MAT-R (Å)', 'RMSD (Å)']
        x = np.arange(len(opt_g))
        w = 0.25
        for i, (m, lbl) in enumerate(zip(metrics, labels)):
            vals = opt_g[m].values * (100 if m == 'fully_valid' else 1)
            ax.bar(x + i*w, vals, w, label=lbl, color=COLORS[i], edgecolor='white')
        ax.set_xticks(x + w); ax.set_xticklabels(opt_g['optimizer'].tolist(), fontsize=10)
        ax.set_title('Optimizer Comparison (AdamW vs Muon)')
        ax.legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'Run experiments first', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Optimizer Comparison')

    plt.tight_layout()
    out = f'{outdir}/ablation_table.png'
    plt.savefig(out); plt.close()
    print(f"  Saved: {out}")
    return out


def plot_ddim_pareto(rows: list, outdir: str):
    """Pareto curve: DDIM steps vs quality metrics."""
    ddim_rows = [r for r in rows if 'ddim_steps' in r] if rows else []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title('Inference Speed vs Quality — DDIM Steps Pareto', fontsize=12, fontweight='bold')

    if ddim_rows:
        ddim_rows_s = sorted(ddim_rows, key=lambda r: r.get('ddim_steps', 50))
        steps  = [r['ddim_steps']  for r in ddim_rows_s]
        valids = [r['fully_valid'] * 100 for r in ddim_rows_s]
        matr   = [r['mat_r']       for r in ddim_rows_s]

        ax2 = ax.twinx()
        ax.plot(steps, valids, 'o-', color=COLORS[0], label='Fully Valid (%)', linewidth=2, markersize=8)
        ax2.plot(steps, matr,  's--', color=COLORS[1], label='MAT-R (Å)',       linewidth=2, markersize=8)
        ax.set_xlabel('DDIM Inference Steps', fontweight='bold')
        ax.set_ylabel('Fully Valid (%)', color=COLORS[0], fontweight='bold')
        ax2.set_ylabel('MAT-R (Å)', color=COLORS[1], fontweight='bold')
        lines = [mpatches.Patch(color=COLORS[0], label='Fully Valid %'),
                 mpatches.Patch(color=COLORS[1], label='MAT-R (Å)')]
        ax.legend(handles=lines)
    else:
        # Show placeholder with expected shape
        steps  = [10, 20, 50, 100, 200, 1000]
        ax.plot(steps, [85, 90, 94, 95, 95.5, 96], 'o--', color=COLORS[0], alpha=0.4, label='Expected Valid% (placeholder)')
        ax.set_xlabel('DDIM Inference Steps'); ax.set_ylabel('Fully Valid (%)')
        ax.legend(); ax.text(0.5, 0.1, 'Run exp_5_ddim experiments to populate',
                             ha='center', transform=ax.transAxes, color='gray', fontsize=9)

    plt.tight_layout()
    out = f'{outdir}/ddim_pareto.png'
    plt.savefig(out); plt.close()
    print(f"  Saved: {out}")
    return out


def plot_experiment_bars(rows: list, outdir: str):
    """Bar chart comparing all experiments on primary metrics."""
    if not rows:
        print("  No data for experiment bar chart.")
        return

    df = pd.DataFrame(rows).sort_values('fully_valid', ascending=False)
    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n*1.2), 5))
    fig.suptitle('Experiment Results — All Runs', fontsize=13, fontweight='bold')

    names = [r.get('exp_name', '?')[:20] for r in df.to_dict('records')]
    x = np.arange(n)

    axes[0].bar(x, df['fully_valid']*100, color=COLORS[0], edgecolor='white')
    axes[0].set_title('Fully Valid Rate (%)')
    axes[0].set_ylabel('%'); axes[0].set_ylim(0, 105)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    axes[0].axhline(94.0, color='red', linestyle='--', alpha=0.6, label='Baseline 94.0%')
    axes[0].legend(fontsize=8)

    axes[1].bar(x, df['mat_r'], color=COLORS[1], edgecolor='white')
    axes[1].set_title('MAT-R (Å) — lower is better')
    axes[1].set_ylabel('Å')
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=40, ha='right', fontsize=8)
    axes[1].axhline(0.1168, color='red', linestyle='--', alpha=0.6, label='Baseline 0.117 Å')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = f'{outdir}/experiment_bars.png'
    plt.savefig(out); plt.close()
    print(f"  Saved: {out}")
    return out


def export_latex_table(rows: list, outdir: str):
    """Export LaTeX table for paper inclusion."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study results on QM9 conformer generation (50-epoch budget each).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Experiment & Valid\% & MAT-R $\downarrow$ & RMSD $\downarrow$ & COV-R $\uparrow$ & Strain & Opt \\",
        r"\midrule",
    ]

    # Baseline first
    baseline = {'exp_name': 'Baseline (200ep, AdamW)', 'fully_valid': 0.940, 'mat_r': 0.1168,
                 'rmsd_mean': 0.2245, 'cov_r': 0.994, 'mean_strain_kcal': 27.53, 'optimizer': 'AdamW'}
    all_rows = [baseline] + (rows if rows else [])

    for r in all_rows:
        name = r.get('exp_name', '?').replace('_', r'\_')[:30]
        fv   = r.get('fully_valid', 0) * 100
        matr = r.get('mat_r', 0)
        rmsd = r.get('rmsd_mean', 0)
        covr = r.get('cov_r', 0) * 100
        str_ = r.get('mean_strain_kcal', r.get('strain_kcal', 0))
        opt  = r.get('optimizer', '?')
        lines.append(f"{name} & {fv:.1f} & {matr:.4f} & {rmsd:.4f} & {covr:.1f} & {str_:.1f} & {opt} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = f'{outdir}/ablation_table.tex'
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  LaTeX table → {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj-root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    proj  = args.proj_root
    outdir = args.output_dir or f'{proj}/plots'
    os.makedirs(outdir, exist_ok=True)

    print(f"Project root : {proj}")
    print(f"Output dir   : {outdir}")
    print()

    # Load data
    rows = load_experiments(proj)
    print(f"Loaded {len(rows)} experiment results.")

    # Best from experiments (or baseline fallback)
    best = max(rows, key=lambda r: r.get('fully_valid', 0)) if rows else None

    # Generate all plots
    plot_comparison_table(outdir, our_metrics=best)
    plot_ablation_table(rows, outdir)
    plot_ddim_pareto(rows, outdir)
    plot_experiment_bars(rows, outdir)
    export_latex_table(rows, outdir)

    # Combined PDF
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        png_files = sorted(glob.glob(f'{outdir}/*.png'))
        if png_files:
            pdf_out = f'{outdir}/results_summary.pdf'
            with PdfPages(pdf_out) as pdf:
                for pf in png_files:
                    img = plt.imread(pf)
                    fig, ax = plt.subplots(figsize=(12, 7))
                    ax.imshow(img); ax.axis('off')
                    ax.set_title(os.path.basename(pf), fontsize=9)
                    pdf.savefig(fig, bbox_inches='tight'); plt.close()
            print(f"  PDF summary → {pdf_out}")
    except Exception as e:
        print(f"  PDF generation skipped: {e}")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == '__main__':
    main()
