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

def plot_event_window(event_row: pd.Series,
                       state_df: pd.DataFrame,
                       binned_obs: pd.DataFrame,
                       raw_obs: pd.DataFrame | None,
                       tide_series: pd.Series,
                       *,
                       hours_each_side: float = 18.0,
                       ax=None):
    """Zoomed water-level plot around one detected event.

    Shows: predicted tide, faint raw windowed obs, binned obs (with σ),
    KF state line, and a vertical marker at the event peak.

    `event_row` is one row from `detect.detect_events()`'s `events_df`,
    expected columns: t_peak_utc, t_start_utc, t_end_utc, amplitude_m,
    direction, n_sats_peak, confidence.

    Returns (fig, ax). Time axis spans ±`hours_each_side` of `t_peak_utc`.
    """
    import matplotlib.pyplot as plt

    t_peak = pd.Timestamp(event_row['t_peak_utc'])
    half   = pd.Timedelta(hours=hours_each_side)
    t_lo, t_hi = t_peak - half, t_peak + half

    state_w  = state_df[(state_df.t_utc >= t_lo) & (state_df.t_utc <= t_hi)]
    binned_w = binned_obs[(binned_obs.t_center_utc >= t_lo)
                          & (binned_obs.t_center_utc <= t_hi)] \
               if binned_obs is not None and len(binned_obs) else pd.DataFrame()
    raw_w    = raw_obs[(raw_obs.t_center_utc >= t_lo)
                       & (raw_obs.t_center_utc <= t_hi)] \
               if raw_obs is not None and len(raw_obs) else pd.DataFrame()
    tide_w   = tide_series[(tide_series.index >= t_lo)
                            & (tide_series.index <= t_hi)] \
               if tide_series is not None and len(tide_series) else pd.Series(dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    if len(tide_w):
        ax.plot(tide_w.index, tide_w.values,
                color='0.5', alpha=0.7, lw=1.0,
                label='Predicted tide')

    if len(raw_w):
        ax.scatter(raw_w.t_center_utc, c.ANTENNA_MSL_M - raw_w.rh,
                   c='0.5', s=4, alpha=0.2, edgecolors='none', zorder=1,
                   label=f'raw obs (n={len(raw_w)})')

    if len(binned_w):
        ax.errorbar(binned_w.t_center_utc, c.ANTENNA_MSL_M - binned_w.rh,
                    yerr=binned_w.sigma,
                    fmt='o', ms=4, mfc='C0', mec='C0',
                    ecolor='C0', elinewidth=0.5, alpha=0.75, zorder=2,
                    label=f'binned obs (n={len(binned_w)})')

    if len(state_w):
        ax.fill_between(state_w.t_utc,
                        state_w.water_level_m - state_w.eta_sigma_m,
                        state_w.water_level_m + state_w.eta_sigma_m,
                        color='C3', alpha=0.20, lw=0,
                        label='KF ± 1σ', zorder=4)
        ax.plot(state_w.t_utc, state_w.water_level_m,
                color='C3', lw=1.8, label='KF water-level estimate',
                zorder=5)

    # Event marker
    ax.axvspan(pd.Timestamp(event_row['t_start_utc']),
               pd.Timestamp(event_row['t_end_utc']),
               color='C1', alpha=0.25, zorder=0,
               label='event window')
    ax.axvline(t_peak, color='C1', lw=1.2, ls='--', zorder=3)

    ax.set_xlim(t_lo, t_hi)
    ax.set_ylabel('Water level (m, rel. MSL)')
    ax.set_xlabel('Time (UTC)')
    style_time_axis(ax)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    # Annotation box
    txt = (f"peak: {t_peak.strftime('%Y-%m-%d %H:%M UTC')}\n"
           f"amplitude: {event_row['amplitude_m']:+.2f} m  ({event_row['direction']})\n"
           f"n_sats: {int(event_row['n_sats_peak'])}  "
           f"n_bins: {int(event_row['n_bins'])}\n"
           f"confidence: {event_row['confidence']:.2f}")
    ax.text(0.01, 0.99, txt, transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='0.7', alpha=0.85))
    fig.tight_layout()
    return fig, ax


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

    # ---- run config (edit to pick which range to plot) ----
    # By default, auto-pick the most recent range directory under data/results/range/.
    # Override TAG manually if you want a specific range.
    TAG = None                     # None = auto-detect newest
    TOP_N_EVENTS = 5               # number of per-event window plots
    HOURS_EACH_SIDE = 18.0         # half-width of event windows

    if TAG is None:
        range_root = c.RESULTS_DIR / 'range'
        if not range_root.exists():
            raise SystemExit(f'No {range_root}/ — run main.py first.')
        cand = sorted([p for p in range_root.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime)
        if not cand:
            raise SystemExit(f'No range subdirs in {range_root}.')
        TAG = cand[-1].name
    print(f'plots.py: loading range {TAG}')

    rdir = c.RESULTS_DIR / 'range' / TAG
    state_path   = rdir / 'state.parquet'
    binned_path  = rdir / 'binned.parquet'
    events_path  = rdir / 'events.parquet'
    if not state_path.exists():
        raise SystemExit(f'No KF state at {state_path} — run main.py first.')

    # Raw windowed obs aren't concatenated into the range dir; load from
    # the per-day caches under each year folder that the range spans.
    raw_obs_frames = []
    for ydir in (c.RESULTS_DIR.iterdir() if c.RESULTS_DIR.exists() else []):
        wdir = ydir / 'windowed'
        if not wdir.is_dir():
            continue
        for per_day in sorted(wdir.glob('[0-9][0-9][0-9]_obs.parquet')):
            raw_obs_frames.append(pd.read_parquet(per_day))
    raw_obs = (pd.concat(raw_obs_frames, ignore_index=True)
               if raw_obs_frames else pd.DataFrame())

    state = pd.read_parquet(state_path)
    binned = pd.read_parquet(binned_path) if binned_path.exists() else pd.DataFrame()
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    obs = binned if not binned.empty else raw_obs    # what the KF actually saw
    print(f'Loaded state: {len(state):,} rows; '
          f'raw obs: {len(raw_obs):,}; binned obs: {len(binned):,}; '
          f'events: {len(events):,}')

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

    # Mid layer: binned obs (what the KF actually consumed)
    if not binned.empty:
        ax.errorbar(binned.t_center_utc,
                    c.ANTENNA_MSL_M - binned.rh,
                    yerr=binned.sigma,
                    fmt='o', ms=3, mfc='C0', mec='C0',
                    ecolor='C0', elinewidth=0.5, alpha=0.7, zorder=2,
                    label=f'binned obs (n={len(binned)})')

    # Foreground: re-plot the KF state on top so the line is visible above
    # the obs scatter (plot_water_level draws it underneath by default)
    ax.plot(state.t_utc, state.water_level_m,
            color='C3', lw=1.8, zorder=5, label='_nolegend_')

    ax.legend(loc='upper right', fontsize=8, ncol=2)
    fig.suptitle(f'{c.STATION.upper()} water level (windowed KF, 120s bins) — '
                 f'range {TAG}', y=1.02)
    out = out_dir / f'water_level_{TAG}.png'
    fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
    print(f'wrote {out.relative_to(c.PROJECT_DIR)}')

    fig, _ = plot_residual(state)
    fig.suptitle(f'{c.STATION.upper()} KF residual η (windowed) — '
                 f'range {TAG}', y=1.02)
    out = out_dir / f'residual_{TAG}.png'
    fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
    print(f'wrote {out.relative_to(c.PROJECT_DIR)}')

    # ----- per-event window plots (top N by confidence) -----
    if not events.empty:
        events_sorted = events.sort_values('confidence', ascending=False)\
                              .reset_index(drop=True)
        topn = events_sorted.head(TOP_N_EVENTS)
        events_dir = out_dir / 'events'
        events_dir.mkdir(parents=True, exist_ok=True)
        print(f'\nGenerating top-{TOP_N_EVENTS} event window plots '
              f'(±{HOURS_EACH_SIDE:g} h around peak):')
        for rank, ev in enumerate(topn.itertuples(index=False), start=1):
            ev_series = pd.Series(ev._asdict())
            fig, _ = plot_event_window(
                ev_series, state, binned, raw_obs, tide,
                hours_each_side=HOURS_EACH_SIDE,
            )
            t_peak = pd.Timestamp(ev_series['t_peak_utc'])
            fig.suptitle(
                f'{c.STATION.upper()} event #{rank} '
                f'({t_peak.strftime("%Y-%m-%d %H:%M")} UTC, '
                f'{ev_series["amplitude_m"]:+.2f} m {ev_series["direction"]})',
                y=1.02,
            )
            tag = t_peak.strftime('%Y%m%dT%H%M')
            out = events_dir / f'event_{rank:02d}_{tag}.png'
            fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
            print(f'  #{rank}: {out.relative_to(c.PROJECT_DIR)}')
    else:
        print('No events found — skipping per-event plots.')
