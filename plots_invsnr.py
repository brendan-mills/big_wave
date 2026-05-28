"""Polished plotting for invsnr-based pipeline results.

Layout (back → front):
  1. raw windowed obs as a faded cloud, each satellite a distinct color,
     legend grouped by constellation (per-sat legend would be unreadable)
  2. invsnr water-level line (with ±1σ band)
  3. predicted tide overlaid on top as the reference signal

Three product plots from `generate()`:
  - `invsnr_overview_{tag}.png`         clean overview (no markers)
  - `invsnr_overview_marked_{tag}.png`  same + event windows shaded (color
                                        by trigger type)
  - per-event zoom: `events/event_NN_<trigger>_<peak>.png`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as c


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

CONSTELLATION_BASE = {
    'GPS':     ('C0', 'Blues'),
    'Galileo': ('C1', 'Oranges'),
    'GLONASS': ('C2', 'Greens'),
    'BeiDou':  ('C4', 'Purples'),
    'other':   ('0.5', 'Greys'),
}

# Event shading color by trigger type
TRIGGER_COLOR = {
    'jump':     'C1',   # orange
    'variance': 'C4',   # purple
    'both':     'C3',   # red
}


def _constellation_of(prn: int) -> str:
    prn = int(prn)
    if 1 <= prn <= 99:        return 'GPS'
    if 101 <= prn <= 199:     return 'GLONASS'
    if 201 <= prn <= 299:     return 'Galileo'
    if 301 <= prn <= 399:     return 'BeiDou'
    return 'other'


def _style_time_axis(ax) -> None:
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax.grid(True, alpha=0.3)


def _load_raw_obs_in_range(t_lo, t_hi) -> pd.DataFrame:
    """Concat per-day windowed obs parquets, clipped to [t_lo, t_hi]."""
    frames = []
    if not c.RESULTS_DIR.exists():
        return pd.DataFrame()
    for ydir in c.RESULTS_DIR.iterdir():
        wdir = ydir / 'windowed'
        if not wdir.is_dir():
            continue
        for p in sorted(wdir.glob('[0-9][0-9][0-9]_obs.parquet')):
            df = pd.read_parquet(p, columns=['t_center_utc', 'sat', 'rh', 'sigma'])
            df = df[df.t_center_utc.between(t_lo, t_hi)]
            if len(df):
                frames.append(df)
    return (pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame())


def _date_to_utc(year: int, doy: int, end_of_day: bool = False) -> pd.Timestamp:
    t = (pd.Timestamp(f'{year}-01-01', tz='UTC')
         + pd.Timedelta(days=doy - 1))
    if end_of_day:
        t += pd.Timedelta(hours=23, minutes=59, seconds=59)
    return t


def _clip_to_window(df, time_col, t_lo, t_hi):
    if df is None or df.empty:
        return df
    return df[df[time_col].between(t_lo, t_hi)].reset_index(drop=True)


def _scatter_by_sat(ax, raw_obs, antenna_msl, *, alpha=0.18, ms=4):
    """Plot raw obs scatter, deterministically colored per satellite,
    grouped in legend by constellation with sat counts."""
    if raw_obs is None or raw_obs.empty:
        return

    raw_obs = raw_obs.copy()
    raw_obs['_constel'] = raw_obs.sat.map(_constellation_of)

    import matplotlib.pyplot as plt

    for constel, group in raw_obs.groupby('_constel', sort=True):
        base_color_name, cmap_name = CONSTELLATION_BASE.get(constel, ('0.5', 'Greys'))
        cmap = plt.get_cmap(cmap_name)
        sats_sorted = sorted(group.sat.unique())
        shades = {sat: cmap(0.30 + 0.65 * i / max(1, len(sats_sorted) - 1))
                  for i, sat in enumerate(sats_sorted)}
        colors = group.sat.map(shades).tolist()
        ax.scatter(group.t_center_utc, antenna_msl - group.rh,
                   c=colors, s=ms, alpha=alpha, edgecolors='none', zorder=1)
        ax.scatter([], [], c=base_color_name, s=20, alpha=0.9,
                   edgecolors='none',
                   label=f'{constel} obs (n={len(group):,}, {len(sats_sorted)} sats)')


# ---------------------------------------------------------------------------
# Main plotting functions
# ---------------------------------------------------------------------------

def plot_overview(state: pd.DataFrame,
                   raw_obs: pd.DataFrame,
                   tide_series: pd.Series,
                   *,
                   events: pd.DataFrame | None = None,
                   antenna_msl: float = c.ANTENNA_MSL_M,
                   ax=None):
    """Headline plot. Layer order: raw obs → invsnr state + σ → tide.
    If `events` is provided, shade each event's time interval colored
    by its trigger type."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5.5))
    else:
        fig = ax.figure

    _scatter_by_sat(ax, raw_obs, antenna_msl)

    if state is not None and len(state):
        ax.plot(state.t_utc, state.water_level_m,
                color='C3', lw=1.8, zorder=3,
                label='invsnr water level')
        if 'eta_sigma_m' in state.columns:
            ax.fill_between(state.t_utc,
                            state.water_level_m - state.eta_sigma_m,
                            state.water_level_m + state.eta_sigma_m,
                            color='C3', alpha=0.18, lw=0, zorder=4,
                            label='invsnr ± 1σ')

    if tide_series is not None and len(tide_series):
        ax.plot(tide_series.index, tide_series.values,
                color='0.15', lw=1.2, alpha=0.85, zorder=5,
                label='Predicted tide (Gr1kmTM)')

    if events is not None and len(events):
        seen_triggers = set()
        for _, ev in events.iterrows():
            trig = ev.get('trigger', 'jump')
            color = TRIGGER_COLOR.get(trig, 'C5')
            label = None
            if trig not in seen_triggers:
                n = int((events['trigger'] == trig).sum())
                label = f'{trig} events (n={n})'
                seen_triggers.add(trig)
            ax.axvspan(ev['t_start_utc'], ev['t_end_utc'],
                       color=color, alpha=0.22, zorder=0, label=label)

    ax.set_ylabel('Water level (m, rel. MSL)')
    ax.set_xlabel('Time (UTC)')
    _style_time_axis(ax)
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    return fig, ax


def plot_event_window(event_row: pd.Series,
                       state: pd.DataFrame,
                       raw_obs: pd.DataFrame,
                       tide_series: pd.Series,
                       *,
                       hours_each_side: float = 12.0,
                       ax=None):
    """Zoomed water-level plot around one detected event. Annotation
    adapts to the trigger type ('jump', 'variance', or 'both')."""
    t_peak = pd.Timestamp(event_row['t_peak_utc'])
    half = pd.Timedelta(hours=hours_each_side)
    t_lo, t_hi = t_peak - half, t_peak + half

    state_w = _clip_to_window(state, 't_utc', t_lo, t_hi)
    raw_w   = _clip_to_window(raw_obs, 't_center_utc', t_lo, t_hi)
    tide_w  = (tide_series[(tide_series.index >= t_lo) & (tide_series.index <= t_hi)]
               if tide_series is not None and len(tide_series) else pd.Series(dtype=float))

    fig, ax = plot_overview(state_w, raw_w, tide_w, ax=ax)

    trig = event_row.get('trigger', 'jump')
    color = TRIGGER_COLOR.get(trig, 'C5')
    ax.axvspan(event_row['t_start_utc'], event_row['t_end_utc'],
               color=color, alpha=0.35, zorder=2)
    ax.axvline(t_peak, color=color, lw=1.5, ls='--', zorder=3)
    ax.set_xlim(t_lo, t_hi)

    lines = [
        f"peak: {t_peak.strftime('%Y-%m-%d %H:%M UTC')}",
        f"trigger: {trig}",
        f"duration: {event_row['duration_sec']:.0f} s",
    ]
    delta = event_row.get('delta_m')
    if delta is not None and not pd.isna(delta):
        wl = event_row.get('water_level_at_peak_m')
        wl_txt = f"  (water level {wl:+.2f} m)" if wl is not None and not pd.isna(wl) else ""
        lines.append(f"Δwater level: {delta:+.2f} m{wl_txt}")
    pvs = event_row.get('peak_var_std_m')
    if pvs is not None and not pd.isna(pvs):
        n = event_row.get('n_obs_in_window')
        n_txt = f", n_obs={int(n)}" if n is not None and not pd.isna(n) else ""
        lines.append(f"peak innov σ: {pvs:.2f} m{n_txt}")
    if 'confidence' in event_row.index:
        lines.append(f"confidence: {event_row['confidence']:.2f}")

    ax.text(0.01, 0.99, '\n'.join(lines), transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.7', alpha=0.9))
    return fig, ax


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate(tag: str | None = None,
             *,
             start: tuple[int, int] | None = None,
             end: tuple[int, int] | None = None,
             out_dir: Path | None = None,
             top_n_events: int = 5) -> dict:
    """Load cached pipeline outputs and render the full plot set.

    Parameters
    ----------
    tag : str or None
        Range tag. If None, auto-picks the most recently modified directory
        under `data/results/range/`.
    start, end : (year, doy), optional
        Clip plots (and event selection) to this sub-window.
    out_dir : Path, optional
        Where to write PNGs. Defaults to `c.PLOTS_DIR`.
    top_n_events : int
        How many top-confidence events to render zoom plots for.

    Returns a dict mapping plot kind -> Path written.
    """
    import matplotlib
    if matplotlib.get_backend().lower() not in ('module://matplotlib_inline.backend_inline',
                                                 'qtagg', 'macosx'):
        matplotlib.use('Agg')
    from tide import GreenlandTideModel

    range_root = c.RESULTS_DIR / 'range'
    if tag is None:
        cands = sorted([p for p in range_root.iterdir() if p.is_dir()],
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f'No range subdirs in {range_root}.')
        tag = cands[-1].name
    print(f'plots_invsnr: loading tag {tag}')

    rdir = range_root / tag
    state = pd.read_parquet(rdir / 'state.parquet')
    events = (pd.read_parquet(rdir / 'events.parquet')
              if (rdir / 'events.parquet').exists() else pd.DataFrame())

    if start is not None and end is not None:
        t_lo_clip = _date_to_utc(*start)
        t_hi_clip = _date_to_utc(*end, end_of_day=True)
        print(f'  clipping to {t_lo_clip} → {t_hi_clip}')
        state  = _clip_to_window(state, 't_utc', t_lo_clip, t_hi_clip)
        events = _clip_to_window(events, 't_peak_utc', t_lo_clip, t_hi_clip)

    if state.empty:
        raise SystemExit('No state rows in the selected range — nothing to plot.')

    t_lo, t_hi = state.t_utc.min(), state.t_utc.max()
    raw_obs = _load_raw_obs_in_range(t_lo, t_hi)

    tm = GreenlandTideModel(c.LAT, c.LON)
    tide = tm.predict_range(t_lo, t_hi, step_sec=300)

    out_dir = Path(out_dir) if out_dir else c.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    sub_tag = (tag if (start is None or end is None)
               else f'{tag}_{start[0]}{start[1]:03d}-{end[0]}{end[1]:03d}')

    # 1. Overview (no markers)
    fig, _ = plot_overview(state, raw_obs, tide)
    fig.suptitle(f'{c.STATION.upper()} water level (invsnr) — {sub_tag}', y=1.01)
    out = out_dir / f'invsnr_overview_{sub_tag}.png'
    fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
    written['overview'] = out
    print(f'  wrote {out.relative_to(c.PROJECT_DIR)}')

    # 2. Overview with events marked
    if not events.empty:
        fig, _ = plot_overview(state, raw_obs, tide, events=events)
        fig.suptitle(f'{c.STATION.upper()} water level + events — {sub_tag}',
                     y=1.01)
        out = out_dir / f'invsnr_overview_marked_{sub_tag}.png'
        fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
        written['overview_marked'] = out
        print(f'  wrote {out.relative_to(c.PROJECT_DIR)}')

    # 3. Per-event zoom plots
    if not events.empty:
        ev_dir = out_dir / 'events'
        ev_dir.mkdir(parents=True, exist_ok=True)
        top = events.head(top_n_events)
        for rank, (_, ev) in enumerate(top.iterrows(), start=1):
            fig, _ = plot_event_window(ev, state, raw_obs, tide)
            t = pd.Timestamp(ev['t_peak_utc'])
            trig = ev.get('trigger', 'event')
            fig.suptitle(f'{c.STATION.upper()} event #{rank} ({trig}) — '
                          f'{t.strftime("%Y-%m-%d %H:%M UTC")}', y=1.02)
            fname = f'event_{rank:02d}_{trig}_{t.strftime("%Y%m%dT%H%M")}.png'
            out = ev_dir / fname
            fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
            print(f'  wrote events/{fname}')
        written['events'] = ev_dir

    return written


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    TAG: str | None = None
    START: tuple[int, int] | None = None
    END:   tuple[int, int] | None = None
    TOP_N_EVENTS = 5

    generate(tag=TAG, start=START, end=END, top_n_events=TOP_N_EVENTS)
