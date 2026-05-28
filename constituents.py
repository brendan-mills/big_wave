"""Harmonic analysis of the invsnr water-level record vs the Gr1kmTM model.

Reads a cached `state.parquet`, solves for the tidal constituents directly from
the observed water level (pyTMD least-squares harmonic analysis), and compares
them against the model's stored constituents at the station — both as a printed
table and as a polar phasor plot.

Each constituent is a complex phasor `A·e^{iθ}` (amplitude A, Greenwich phase θ).
On the polar plot, radius = amplitude (cm), angle = phase, color = constituent.
The dashed arrow is the model, the solid arrow is the observed GNSS-IR estimate,
and the dotted connector between their tips is the discrepancy vector.

Edit the constants in RUN CONFIG, then click Run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD

import config as c
from tide import GreenlandTideModel, _to_days_since_epoch


# =============================================================================
# RUN CONFIG — edit then Run
# =============================================================================

# Range tag to analyze. None = auto-pick the cached range with the most rows
# (longest record → best constituent resolution).
TAG: str | None = None

# Detrend polynomial degree folded into the LS fit (0 = constant/mean only,
# 1 = also remove a linear drift over the record).
POLY_ORDER = 1

# Least-squares solver passed to pyTMD.solve.constants ('lstsq' is fine here).
SOLVER = 'lstsq'


# =============================================================================
# Data / solve
# =============================================================================

def pick_longest_tag() -> str:
    """Cached range with the most state rows."""
    range_root = c.RESULTS_DIR / 'range'
    best, best_n = None, -1
    for d in sorted(range_root.iterdir()):
        sp = d / 'state.parquet'
        if not sp.exists():
            continue
        n = len(pd.read_parquet(sp, columns=['t_utc']))
        if n > best_n:
            best, best_n = d.name, n
    if best is None:
        raise SystemExit(f'No state.parquet found under {range_root}')
    return best


def solve_observed(state: pd.DataFrame, constituents: list[str], *,
                   order: int = POLY_ORDER,
                   solver: str = SOLVER) -> dict[str, complex]:
    """Least-squares harmonic constants from the observed water level.

    Uses `corrections='OTIS'` so the nodal corrections / phase convention
    match the Gr1kmTM (OTIS-format) model — making the comparison apples-to-apples.
    """
    s = state.dropna(subset=['water_level_m'])
    t = _to_days_since_epoch(s['t_utc'])
    ht = s['water_level_m'].to_numpy()
    ds = pyTMD.solve.constants(t, ht - ht.mean(), constituents,
                               corrections='OTIS', order=order, solver=solver)
    return {nm: complex(ds[nm].item()) for nm in constituents}


def compare_table(obs: dict[str, complex],
                  model: dict[str, complex]) -> pd.DataFrame:
    """Per-constituent amplitude (cm), Greenwich phase (deg), and the
    amplitude / phase / complex-vector discrepancies."""
    rows = []
    for nm in model:
        zo, zm = obs[nm], model[nm]
        po = np.degrees(np.angle(zo)) % 360
        pm = np.degrees(np.angle(zm)) % 360
        rows.append({
            'constituent':   nm.upper(),
            'obs_amp_cm':    abs(zo) * 100,
            'mod_amp_cm':    abs(zm) * 100,
            'obs_phase_deg': po,
            'mod_phase_deg': pm,
            'dphase_deg':    ((po - pm + 180) % 360) - 180,
            'vec_error_cm':  abs(zo - zm) * 100,
        })
    return (pd.DataFrame(rows)
            .sort_values('mod_amp_cm', ascending=False)
            .reset_index(drop=True))


# =============================================================================
# Plot
# =============================================================================

def plot_phasors(obs: dict[str, complex],
                 model: dict[str, complex], *,
                 title: str = ''):
    """Polar phasor diagram: observed (solid) vs model (dashed) per constituent."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    names = sorted(model, key=lambda n: -abs(model[n]))   # largest first
    cmap = plt.get_cmap('tab10')
    colors = {nm: cmap(i % 10) for i, nm in enumerate(names)}

    rmax = max(max(abs(obs[n]), abs(model[n])) for n in names) * 100 * 1.18

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw={'projection': 'polar'})

    for nm in names:
        col = colors[nm]
        zo, zm = obs[nm], model[nm]
        to, ro = np.angle(zo), abs(zo) * 100
        tm_, rm = np.angle(zm), abs(zm) * 100

        # model arrow (dashed, faded), observed arrow (solid)
        ax.annotate('', xy=(tm_, rm), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col,
                                    lw=1.5, ls='--', alpha=0.55))
        ax.annotate('', xy=(to, ro), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2))
        # discrepancy connector (tip to tip)
        ax.plot([tm_, to], [rm, ro], color=col, lw=0.9, ls=':', alpha=0.8)
        # constituent label just beyond the observed tip
        ax.text(to, ro + rmax * 0.04, nm.upper(), color=col,
                ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_rmax(rmax)
    ax.set_rlabel_position(112.5)
    ax.set_theta_zero_location('E')
    ax.grid(True, alpha=0.35)
    ax.set_title(title, pad=24, fontsize=12)

    handles = [
        Line2D([0], [0], color='0.3', lw=2.2, label='Observed (GNSS-IR / invsnr)'),
        Line2D([0], [0], color='0.3', lw=1.5, ls='--', alpha=0.7, label='Model (Gr1kmTM)'),
        Line2D([0], [0], color='0.3', lw=0.9, ls=':', label='Discrepancy'),
    ]
    ax.legend(handles=handles, loc='lower left',
              bbox_to_anchor=(-0.05, -0.08), fontsize=9, framealpha=0.9)
    fig.text(0.5, 0.02,
             'angle = Greenwich phase (deg)  ·  radius = amplitude (cm)  ·  color = constituent',
             ha='center', fontsize=9, color='0.4')
    fig.tight_layout()
    return fig, ax


# =============================================================================
# Run
# =============================================================================

def run(tag: str | None = TAG):
    tag = tag or pick_longest_tag()
    sp = c.RESULTS_DIR / 'range' / tag / 'state.parquet'
    if not sp.exists():
        raise SystemExit(f'No state.parquet for tag {tag} at {sp}')

    state = pd.read_parquet(sp, columns=['t_utc', 'water_level_m'])
    span = state['t_utc'].max() - state['t_utc'].min()
    n_valid = state['water_level_m'].notna().sum()
    print(f'Tag {tag}: {n_valid:,} valid samples, '
          f'{state.t_utc.min().date()} -> {state.t_utc.max().date()} '
          f'({span.days}d span)')

    tm = GreenlandTideModel(c.LAT, c.LON)
    cons = tm.constituent_names
    obs = solve_observed(state, cons)
    model = tm.constituents()

    table = compare_table(obs, model)
    print('\nObserved (GNSS-IR) vs Gr1kmTM model constituents:')
    print(table.round(2).to_string(index=False))
    rms_amp = np.sqrt((table['obs_amp_cm'] - table['mod_amp_cm']).pow(2).mean())
    rms_vec = np.sqrt(table['vec_error_cm'].pow(2).mean())
    print(f'\n  RMS amplitude diff: {rms_amp:.2f} cm   '
          f'RMS vector error: {rms_vec:.2f} cm')

    import matplotlib
    if matplotlib.get_backend().lower() not in (
            'module://matplotlib_inline.backend_inline', 'qtagg', 'macosx'):
        matplotlib.use('Agg')

    fig, _ = plot_phasors(
        obs, model,
        title=f'{c.STATION.upper()} tidal constituents — GNSS-IR vs Gr1kmTM\n'
              f'{tag}  ({span.days} d, {n_valid:,} samples)')
    out_dir = c.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'constituents_phasor_{tag}.png'
    fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
    print(f'\nPhasor plot written to {out.relative_to(c.PROJECT_DIR)}')
    return table


if __name__ == '__main__':
    run()
