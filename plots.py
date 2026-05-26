"""Standard visualization library for SSiSLS.

Conventions used by every function:
- Return `(fig, ax)`; accept `ax=None` to create a new figure.
- `ConciseDateFormatter` for any time x-axis.
- Project color schemes from `config` (`SNR_COL_COLOR`, `CONSTELLATION_MARKER`).
- Grids at alpha=0.3, reference lines at alpha=0.5.

The headline function is `plot_water_level`, which overlays the predicted
tide, the Kalman state estimate, and the raw per-arc GNSS-IR observations
on one axis — the diagnostic that tells you at a glance whether the filter
is tracking the tide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as c


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def style_time_axis(ax) -> None:
    """Apply project conventions to a time-x-axis (call after plotting)."""
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax.grid(True, alpha=0.3)


def melt_obs_to_long(arcs_df: pd.DataFrame,
                     antenna_msl: float = c.ANTENNA_MSL_M) -> pd.DataFrame:
    """Wide per-arc DataFrame -> long per-observation DataFrame.

    One row per (arc, signal) with finite RH. Columns:
        t_utc, sat, constellation, signal, snr_col, rh, sigma, water_level

    `water_level = antenna_msl − RH` (m above MSL, before any tide subtraction).
    """
    rows = []
    for _, arc in arcs_df.iterrows():
        for sig in c.ENABLED_SIGNALS:
            rh_col, sig_col = f'RH_{sig.name}', f'sigma_{sig.name}'
            if rh_col not in arc.index:
                continue
            rh, sigma = arc[rh_col], arc[sig_col]
            if not (np.isfinite(rh) and np.isfinite(sigma)):
                continue
            rows.append({
                't_utc':         arc['t_mid_utc'],
                'sat':           int(arc['sat']),
                'constellation': arc['constellation'],
                'signal':        sig.name,
                'snr_col':       sig.snr_col,
                'rh':            float(rh),
                'sigma':         float(sigma),
                'water_level':   antenna_msl - float(rh),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main plot: tide prior + KF state + raw obs
# ---------------------------------------------------------------------------

def plot_water_level(state_df: pd.DataFrame,
                     arcs_df: pd.DataFrame | None = None,
                     tide_series: pd.Series | None = None,
                     *,
                     ax=None,
                     antenna_msl: float = c.ANTENNA_MSL_M,
                     show_obs: bool = True,
                     show_sigma_band: bool = True,
                     show_gated: pd.DataFrame | None = None,
                     obs_alpha: float = 0.45,
                     color_by: str = 'snr_col'):
    """Overlay the predicted tide, KF state estimate (±σ), and raw obs.

    Parameters
    ----------
    state_df : DataFrame from `estimate.run_batch`.
        Expects columns `t_utc`, `water_level_m`, `eta_sigma_m`.
    arcs_df : DataFrame from `pipeline.load_results`. Set None to skip.
    tide_series : pd.Series indexed by tz-aware UTC times, values m above MSL.
        Obtain via `tide_model.predict_range(...)`. Set None to skip.
    show_gated : optional `gated_df` from `estimate.run_batch`; if provided,
        rejected obs are drawn as gray X markers for inspection.
    color_by : 'snr_col' (color by carrier band, default — signals on the
        same wavelength share a color), 'constellation', or 'signal'.

    Returns (fig, ax). All y values are in meters above MSL.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    # 1) predicted tide — drawn first so points/lines overlay
    if tide_series is not None and len(tide_series):
        ax.plot(tide_series.index, tide_series.values,
                color='0.5', alpha=0.7, lw=1.0,
                label='Predicted tide (Gr1kmTM)')

    # 2) raw observations as scatter
    if show_obs and arcs_df is not None and len(arcs_df):
        long = melt_obs_to_long(arcs_df, antenna_msl=antenna_msl)
        color_map = _resolve_color_map(color_by, long)
        for key, group in long.groupby(color_by):
            ax.scatter(group.t_utc, group.water_level,
                       c=color_map.get(key, 'k'),
                       s=14, alpha=obs_alpha, edgecolors='none',
                       label=f'{key} (n={len(group)})')

    # 3) gated observations (optional)
    if show_gated is not None and len(show_gated):
        # Reconstruct water level from gated y = antenna_msl − tide − RH
        # but y in gated_df is already (antenna_msl − tide − RH). So we add
        # back the local tide to get water_level. We need the tide at each
        # gated obs time — interpolate from the tide_series if provided,
        # otherwise plot just the (η = obs.y) markers at a horizontal line.
        if tide_series is not None:
            tide_at_g = np.interp(
                show_gated.t_utc.astype('int64').values,
                tide_series.index.astype('int64').values,
                tide_series.values,
            )
            wl_gated = show_gated.y + tide_at_g
            ax.scatter(show_gated.t_utc, wl_gated,
                       marker='x', s=30, c='0.3', alpha=0.6,
                       label=f'gated (n={len(show_gated)})')

    # 4) KF state estimate (on top so it's visible above scatter)
    if len(state_df):
        if show_sigma_band:
            ax.fill_between(
                state_df.t_utc,
                state_df.water_level_m - state_df.eta_sigma_m,
                state_df.water_level_m + state_df.eta_sigma_m,
                color='C3', alpha=0.25, lw=0,
                label='KF ± 1σ',
            )
        ax.plot(state_df.t_utc, state_df.water_level_m,
                color='C3', lw=1.5, label='KF water-level estimate')

    ax.set_ylabel('Water level (m, rel. MSL)')
    ax.set_xlabel('Time (UTC)')
    style_time_axis(ax)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    fig.tight_layout()
    return fig, ax


def _resolve_color_map(color_by: str, long_df: pd.DataFrame) -> dict:
    """Map distinct values of `color_by` column to a project color."""
    if color_by == 'snr_col':
        return dict(c.SNR_COL_COLOR)
    keys = sorted(long_df[color_by].dropna().unique())
    return {k: f'C{i}' for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# Residual diagnostic
# ---------------------------------------------------------------------------

def plot_residual(state_df: pd.DataFrame, *, ax=None):
    """KF η (= water level − predicted tide − offset) over time.

    Diagnostic for filter tracking quality: a clean η near zero with small
    σ band means the KF + tide model together explain the observations.
    Large slow drifts mean the offset / antenna calibration is off; large
    fast excursions mean either bad obs or real events the filter didn't
    absorb.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 4))
    else:
        fig = ax.figure

    mean_eta = float(state_df.eta_m.mean())
    ax.fill_between(state_df.t_utc,
                    state_df.eta_m - state_df.eta_sigma_m,
                    state_df.eta_m + state_df.eta_sigma_m,
                    color='C0', alpha=0.25, lw=0, label='η ± 1σ')
    ax.plot(state_df.t_utc, state_df.eta_m,
            color='C0', lw=1.2, label='η = water level − predicted tide')
    ax.axhline(mean_eta, color='k', lw=0.8, alpha=0.5, linestyle='--',
               label=f'mean: {mean_eta:+.3f} m')
    ax.set_ylabel('η (m)')
    ax.set_xlabel('Time (UTC)')
    style_time_axis(ax)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Smoke test — load doys 1-3, run filter, save the headline plot
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # headless

    from tide import GreenlandTideModel

    YEAR, DOYS = 2026, [1, 2, 3]

    # Load the state, binned obs (the actual KF input), and raw obs (faint
    # scatter behind for context). estimate.py is the canonical producer.
    state_path  = (c.RESULTS_DIR / f'{YEAR}' / 'state' /
                   f'{DOYS[0]:03d}-{DOYS[-1]:03d}_state.parquet')
    raw_path    = (c.RESULTS_DIR / f'{YEAR}' / 'windowed' /
                   f'{DOYS[0]:03d}-{DOYS[-1]:03d}_obs.parquet')
    binned_path = (c.RESULTS_DIR / f'{YEAR}' / 'windowed' /
                   f'{DOYS[0]:03d}-{DOYS[-1]:03d}_binned.parquet')
    if not state_path.exists():
        raise SystemExit(f'No KF state at {state_path} — run estimate.py first.')

    state = pd.read_parquet(state_path)
    raw_obs = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
    binned = pd.read_parquet(binned_path) if binned_path.exists() else pd.DataFrame()
    obs = binned if not binned.empty else raw_obs    # what the KF actually saw
    print(f'Loaded state: {len(state):,} rows; '
          f'raw obs: {len(raw_obs):,}; binned obs: {len(binned):,}')

    tm = GreenlandTideModel(c.LAT, c.LON)
    t_min, t_max = state.t_utc.min(), state.t_utc.max()
    tide = tm.predict_range(t_min, t_max, step_sec=300)

    out_dir = c.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_water_level(state, arcs_df=None, tide_series=tide)

    # Faint background: raw per-window obs across all sats/bands (so you can
    # see the underlying scatter the binning compressed)
    if not raw_obs.empty:
        ax.scatter(raw_obs.t_center_utc,
                   c.ANTENNA_MSL_M - raw_obs.rh,
                   c='0.5', s=3, alpha=0.15, edgecolors='none', zorder=1,
                   label=f'raw windowed obs (n={len(raw_obs)})')

    # Foreground: binned obs (what the KF actually consumed)
    if not binned.empty:
        ax.errorbar(binned.t_center_utc,
                    c.ANTENNA_MSL_M - binned.rh,
                    yerr=binned.sigma,
                    fmt='o', ms=3, mfc='C0', mec='C0',
                    ecolor='C0', elinewidth=0.5, alpha=0.7, zorder=2,
                    label=f'binned obs (n={len(binned)})')

    ax.legend(loc='upper right', fontsize=8, ncol=2)
    fig.suptitle(f'{c.STATION.upper()} water level (windowed KF, 120s bins) — '
                 f'doys {DOYS[0]:03d}–{DOYS[-1]:03d} of {YEAR}', y=1.02)
    out = out_dir / f'water_level_{YEAR}_{DOYS[0]:03d}-{DOYS[-1]:03d}.png'
    fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
    print(f'wrote {out.relative_to(c.PROJECT_DIR)}')

    fig, _ = plot_residual(state)
    fig.suptitle(f'{c.STATION.upper()} KF residual η (windowed) — '
                 f'doys {DOYS[0]:03d}–{DOYS[-1]:03d} of {YEAR}', y=1.02)
    out = out_dir / f'residual_{YEAR}_{DOYS[0]:03d}-{DOYS[-1]:03d}.png'
    fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
    print(f'wrote {out.relative_to(c.PROJECT_DIR)}')
