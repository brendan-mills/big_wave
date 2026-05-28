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
    'surge':    'C5',   # brown
}


def _trigger_color(trigger: str) -> str:
    """Color for an event by trigger label. '+'-joined multi-trigger events
    (the most interesting) get red; single triggers get their own hue."""
    if '+' in str(trigger):
        return 'C3'     # red — multi-pathway event
    return TRIGGER_COLOR.get(trigger, 'C7')


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
            color = _trigger_color(trig)
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
    adapts to the trigger type ('jump', 'variance', 'surge', or combos)."""
    t_peak = pd.Timestamp(event_row['t_peak_utc'])
    half = pd.Timedelta(hours=hours_each_side)
    t_lo, t_hi = t_peak - half, t_peak + half

    state_w = _clip_to_window(state, 't_utc', t_lo, t_hi)
    raw_w   = _clip_to_window(raw_obs, 't_center_utc', t_lo, t_hi)
    tide_w  = (tide_series[(tide_series.index >= t_lo) & (tide_series.index <= t_hi)]
               if tide_series is not None and len(tide_series) else pd.Series(dtype=float))

    fig, ax = plot_overview(state_w, raw_w, tide_w, ax=ax)

    trig = event_row.get('trigger', 'jump')
    color = _trigger_color(trig)
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
    pvs = event_row.get('peak_burst_amp_m')
    if pvs is not None and not pd.isna(pvs):
        n = event_row.get('n_obs_in_window')
        n_txt = f", n_obs={int(n)}" if n is not None and not pd.isna(n) else ""
        lines.append(f"straddle amp: ±{pvs:.2f} m{n_txt}")
    dev = event_row.get('peak_tide_dev_m')
    if dev is not None and not pd.isna(dev):
        lines.append(f"tide deviation: {dev:+.2f} m")
    if 'confidence' in event_row.index:
        lines.append(f"confidence: {event_row['confidence']:.2f}")

    ax.text(0.01, 0.99, '\n'.join(lines), transform=ax.transAxes,
            ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.7', alpha=0.9))
    return fig, ax


# ---------------------------------------------------------------------------
# GNSS-IR method illustration: one SNR arc + its Lomb-Scargle periodogram
# ---------------------------------------------------------------------------

def _best_arc_spectrum(year: int, doy: int) -> dict | None:
    """Scan a day's arcs/signals and return the cleanest retrieval (highest
    peak-to-noise, not edge-pinned) for the illustration. Dict keys:
    spec (snr.arc_spectrum output), sig, arc (per-sample df), sat, arc_id."""
    import snr
    snr_df = snr.load_snr(year, doy)
    arcs = snr.segment_arcs(snr_df)
    if arcs.empty:
        return None
    best = None
    for arc_id, g in arcs.groupby('arc_id', sort=True):
        sat = int(g.sat.iloc[0])
        for sig in c.signals_for_sat(sat):
            spec = snr.arc_spectrum(g.elev.values, g[sig.snr_col].values, sig)
            if spec is None or not np.isfinite(spec['rh']) or spec['edge_hit']:
                continue
            if best is None or spec['p2n'] > best['spec']['p2n']:
                best = {'spec': spec, 'sig': sig, 'arc': g,
                        'sat': sat, 'arc_id': int(arc_id)}
    return best


def _select_arc(year: int, doy: int,
                arc_id: int | None = None,
                signal_name: str | None = None) -> dict:
    """Pick the arc/signal to illustrate: a specific (arc_id, signal_name)
    if both given, otherwise the day's cleanest arc. Raises if none found."""
    import snr
    if arc_id is not None and signal_name is not None:
        arcs = snr.segment_arcs(snr.load_snr(year, doy))
        g = arcs[arcs.arc_id == arc_id]
        if g.empty:
            raise ValueError(f'arc_id {arc_id} not found on {year}-{doy:03d}')
        sat = int(g.sat.iloc[0])
        sig = next(s for s in c.signals_for_sat(sat) if s.name == signal_name)
        spec = snr.arc_spectrum(g.elev.values, g[sig.snr_col].values, sig)
        sel = {'spec': spec, 'sig': sig, 'arc': g, 'sat': sat, 'arc_id': arc_id}
    else:
        sel = _best_arc_spectrum(year, doy)
    if sel is None or sel['spec'] is None:
        raise ValueError(f'No clean arc found on {year}-{doy:03d}')
    return sel


def _ls_reconstruction(spec: dict, sig) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares amplitude/phase of the SNR oscillation at the retrieved
    RH, evaluated on a fine sin(elev) grid. Returns (se_fine, fit)."""
    se, dsnr = spec['sin_elev'], spec['dsnr']
    omega = 4 * np.pi * spec['rh'] / sig.wavelength_m
    M = np.column_stack([np.sin(omega * se), np.cos(omega * se)])
    ab, *_ = np.linalg.lstsq(M, dsnr, rcond=None)
    se_fine = np.linspace(se.min(), se.max(), 600)
    fit = ab[0] * np.sin(omega * se_fine) + ab[1] * np.cos(omega * se_fine)
    return se_fine, fit


def plot_snr_oscillation(year: int, doy: int, *,
                         arc_id: int | None = None,
                         signal_name: str | None = None):
    """Two-panel GNSS-IR illustration for a single arc/signal:
      (left)  detrended SNR vs sin(elevation) — the interference oscillation,
              with the Lomb-Scargle best-fit sinusoid overlaid;
      (right) the LS periodogram (power vs reflector height) with the
              retrieved RH peak marked.

    If `arc_id`/`signal_name` are omitted, auto-picks the cleanest arc of the
    day (highest peak-to-noise). Returns (fig, (ax_left, ax_right))."""
    import matplotlib.pyplot as plt

    sel = _select_arc(year, doy, arc_id, signal_name)
    spec, sig, g, sat = sel['spec'], sel['sig'], sel['arc'], sel['sat']
    se, dsnr = spec['sin_elev'], spec['dsnr']
    order = np.argsort(se)
    se_fine, fit = _ls_reconstruction(spec, sig)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: SNR oscillation ---
    axL.plot(se[order], dsnr[order], '.', ms=4, color='C0', alpha=0.65,
             label='detrended SNR')
    axL.plot(se_fine, fit, '-', color='C3', lw=1.9,
             label=f'LS fit @ RH = {spec["rh"]:.2f} m')
    axL.set_xlabel('sin(elevation angle)')
    axL.set_ylabel('detrended SNR (volts)')
    axL.set_title('SNR interference oscillation')
    axL.legend(loc='upper right', fontsize=9)
    axL.grid(True, alpha=0.3)

    # --- Right: Lomb-Scargle periodogram ---
    axR.plot(spec['heights'], spec['pgram'], color='C0', lw=1.4)
    axR.axvline(spec['rh'], color='C3', ls='--', lw=1.5)
    axR.plot(spec['rh'], spec['pgram'][spec['i_peak']], 'v',
             color='C3', ms=9)
    axR.set_xlabel('reflector height (m)')
    axR.set_ylabel('normalized LS power')
    axR.set_title('Lomb-Scargle periodogram')
    axR.grid(True, alpha=0.3)
    axR.text(0.97, 0.95,
             f'RH = {spec["rh"]:.2f} m\n'
             f'peak/noise = {spec["p2n"]:.1f}\n'
             f'σ = {spec["sigma_rh"]*100:.1f} cm'
             if np.isfinite(spec["sigma_rh"]) else
             f'RH = {spec["rh"]:.2f} m\npeak/noise = {spec["p2n"]:.1f}',
             transform=axR.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='0.7', alpha=0.9))

    constel = c.constellation_for_sat(sat)
    t_mid = g.time_utc.min() + (g.time_utc.max() - g.time_utc.min()) / 2
    fig.suptitle(
        f'{c.STATION.upper()} GNSS-IR — {constel} PRN {sat}, {sig.name}, '
        f'{sel["arc"]["dir"].iloc[0]} arc\n'
        f'{t_mid.strftime("%Y-%m-%d %H:%M UTC")}  ·  '
        f'elev {spec["elev"].min():.1f}–{spec["elev"].max():.1f}°  ·  '
        f'az {g.azim.mean():.0f}°  ·  {len(se)} samples',
        y=1.02, fontsize=12)
    fig.tight_layout()
    return fig, (axL, axR)


def plot_snr_walkthrough(year: int, doy: int, *,
                         out_dir: Path | None = None,
                         arc_id: int | None = None,
                         signal_name: str | None = None,
                         dpi: int | None = None) -> list[Path]:
    """Break the two-panel GNSS-IR illustration into cumulative frames for a
    presentation walkthrough, saved as numbered PNGs in `out_dir` (default
    `plots/snr_walkthrough/`):

      1_axes.png            empty axes — explain what each panel shows
      2_snr_observations.png  + detrended SNR dots (left)
      3_periodogram.png       + Lomb-Scargle periodogram (right)
      4_peak_and_fit.png      + RH peak (right) and reconstructed wave (left)

    Every frame shares identical axis limits so they overlay cleanly when
    advancing slides. Returns the list of paths written, in order."""
    import matplotlib.pyplot as plt

    sel = _select_arc(year, doy, arc_id, signal_name)
    spec, sig, g, sat = sel['spec'], sel['sig'], sel['arc'], sel['sat']
    se, dsnr = spec['sin_elev'], spec['dsnr']
    order = np.argsort(se)
    se_fine, fit = _ls_reconstruction(spec, sig)

    out_dir = Path(out_dir) if out_dir else (c.PLOTS_DIR / 'snr_walkthrough')
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = dpi or c.DPI

    # Fixed limits so the panels don't shift between frames
    se_pad = 0.02 * (se.max() - se.min())
    d_amp  = float(np.max(np.abs(dsnr))) * 1.1
    xL = (se.min() - se_pad, se.max() + se_pad)
    yL = (-d_amp, d_amp)
    xR = (float(spec['heights'].min()), float(spec['heights'].max()))
    yR = (0.0, float(spec['pgram'].max()) * 1.12)

    constel = c.constellation_for_sat(sat)
    t_mid = g.time_utc.min() + (g.time_utc.max() - g.time_utc.min()) / 2
    suptitle = (f'{c.STATION.upper()} GNSS-IR — {constel} PRN {sat}, {sig.name}, '
                f'{g["dir"].iloc[0]} arc\n'
                f'{t_mid.strftime("%Y-%m-%d %H:%M UTC")}  ·  '
                f'elev {spec["elev"].min():.1f}–{spec["elev"].max():.1f}°  ·  '
                f'{len(se)} samples')

    def blank():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
        axL.set_xlabel('sin(elevation angle)')
        axL.set_ylabel('detrended SNR (volts)')
        axL.set_title('SNR interference oscillation')
        axL.set_xlim(*xL); axL.set_ylim(*yL); axL.grid(True, alpha=0.3)
        axR.set_xlabel('reflector height (m)')
        axR.set_ylabel('normalized LS power')
        axR.set_title('Lomb-Scargle periodogram')
        axR.set_xlim(*xR); axR.set_ylim(*yR); axR.grid(True, alpha=0.3)
        fig.suptitle(suptitle, y=1.02, fontsize=12)
        fig.tight_layout()
        return fig, axL, axR

    def add_dots(axL):
        axL.plot(se[order], dsnr[order], '.', ms=4, color='C0', alpha=0.65,
                 label='detrended SNR')
        axL.legend(loc='upper right', fontsize=9)

    def add_pgram(axR):
        axR.plot(spec['heights'], spec['pgram'], color='C0', lw=1.4)

    written: list[Path] = []

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        written.append(p)
        print(f'  wrote {p.relative_to(c.PROJECT_DIR)}')

    # Frame 1 — empty axes
    fig, axL, axR = blank()
    save(fig, '1_axes.png')

    # Frame 2 — SNR observations (left)
    fig, axL, axR = blank()
    add_dots(axL)
    save(fig, '2_snr_observations.png')

    # Frame 3 — periodogram (right)
    fig, axL, axR = blank()
    add_dots(axL)
    add_pgram(axR)
    save(fig, '3_periodogram.png')

    # Frame 4 — peak (right) + reconstructed component (left)
    fig, axL, axR = blank()
    add_dots(axL)
    axL.plot(se_fine, fit, '-', color='C3', lw=1.9,
             label=f'LS fit @ RH = {spec["rh"]:.2f} m')
    axL.legend(loc='upper right', fontsize=9)
    add_pgram(axR)
    axR.axvline(spec['rh'], color='C3', ls='--', lw=1.5)
    axR.plot(spec['rh'], spec['pgram'][spec['i_peak']], 'v', color='C3', ms=9)
    axR.text(0.97, 0.95,
             (f'RH = {spec["rh"]:.2f} m\npeak/noise = {spec["p2n"]:.1f}\n'
              f'σ = {spec["sigma_rh"]*100:.1f} cm'
              if np.isfinite(spec["sigma_rh"]) else
              f'RH = {spec["rh"]:.2f} m\npeak/noise = {spec["p2n"]:.1f}'),
             transform=axR.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='0.7', alpha=0.9))
    save(fig, '4_peak_and_fit.png')

    return written


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

    # 4. GNSS-IR method illustration: one SNR arc + its periodogram.
    # Use a day near the middle of the record (likely clean summer data).
    mid_t = t_lo + (t_hi - t_lo) / 2
    snr_year, snr_doy = int(mid_t.year), int(mid_t.dayofyear)
    try:
        fig, _ = plot_snr_oscillation(snr_year, snr_doy)
        out = out_dir / f'snr_oscillation_{sub_tag}.png'
        fig.savefig(out, dpi=c.DPI, bbox_inches='tight')
        written['snr_oscillation'] = out
        print(f'  wrote {out.relative_to(c.PROJECT_DIR)}')
    except Exception as e:
        print(f'  snr oscillation plot skipped: {type(e).__name__}: {e}')

    # 5. Step-by-step walkthrough frames of the same illustration
    try:
        frames = plot_snr_walkthrough(snr_year, snr_doy)
        written['snr_walkthrough'] = frames[0].parent if frames else None
    except Exception as e:
        print(f'  snr walkthrough skipped: {type(e).__name__}: {e}')

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
