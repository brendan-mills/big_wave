"""Plots for the 1 Hz roughness pipeline.

  - overview_{tag}.png        invsnr water level + tide over the range, surge
                              events shaded, roughness events marked.
  - roughness_timeline_{tag}.png  hourly roughness activity (max rough_ratio +
                              coherent-sat count) with event markers — makes the
                              episodic clustering obvious.
  - events/event_NN_<trigger>.png   top-N events: rough_ratio per satellite
                              around the window (the detection evidence) over
                              water level + tide.

Reads state.parquet + events.parquet + the per-day roughness caches; loads only
the days it needs (no 23 M-row concat). Driven from main.stage_plots; the
__main__ block runs it standalone on the most recent range.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.dates as mdates        # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

import config as c                       # noqa: E402

TOP_N         = 5         # number of per-event zoom plots (by confidence)
ROUGH_PAD_SEC = 1800      # +/- zoom padding for a roughness event (s) — enough
                          # tide/state context around the brief burst
SURGE_PAD_SEC = 12 * 3600 # +/- zoom padding for a surge event (s) — tide-scale context
_TRIG_COLOR   = {'surge': 'C0', 'roughness': 'C3'}


# ---------------------------------------------------------------------------
# Paths / loading
# ---------------------------------------------------------------------------

def _latest_tag() -> str | None:
    rdir = c.RESULTS_DIR / 'range'
    if not rdir.exists():
        return None
    tags = [p for p in rdir.iterdir() if p.is_dir() and (p / 'events.parquet').exists()]
    if not tags:
        return None
    return max(tags, key=lambda p: p.stat().st_mtime).name


def _roughness_path(year: int, doy: int):
    return c.RESULTS_DIR / f'{year}' / 'roughness' / f'{doy:03d}_obs.parquet'


def _load_roughness_window(t_lo: pd.Timestamp, t_hi: pd.Timestamp) -> pd.DataFrame:
    """Load only the per-day roughness caches spanning [t_lo, t_hi]."""
    frames = []
    d = t_lo.floor('D')
    while d <= t_hi:
        year, doy = d.year, int(d.strftime('%j'))
        p = _roughness_path(year, doy)
        if p.exists():
            df = pd.read_parquet(p, columns=['t_center_utc', 'sat', 'constellation',
                                             'rough_ratio'])
            frames.append(df[(df.t_center_utc >= t_lo) & (df.t_center_utc <= t_hi)])
        d += pd.Timedelta(days=1)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def _style_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.margins(x=0)                      # trim to the data range — no side whitespace
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)


def _gap_break(df: pd.DataFrame, value_cols, t_col='t_utc',
               max_gap_sec: float = 900) -> pd.DataFrame:
    """Insert a NaN row after every sample that precedes a time gap larger than
    `max_gap_sec`, so line plots BREAK at the gap instead of drawing a straight
    line across it (e.g. across winter days where invsnr retrieved nothing)."""
    if df.empty:
        return df
    df = df.sort_values(t_col).reset_index(drop=True)
    gap_after = df[t_col].diff().dt.total_seconds().shift(-1) > max_gap_sec
    if not gap_after.any():
        return df
    breaks = [{t_col: df.loc[i, t_col] + pd.Timedelta(milliseconds=1),
               **{col: np.nan for col in value_cols}}
              for i in df.index[gap_after]]
    return (pd.concat([df, pd.DataFrame(breaks)], ignore_index=True)
            .sort_values(t_col).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_overview(state: pd.DataFrame, events: pd.DataFrame, out) -> None:
    """Water level + tide over the whole range; surge windows shaded, roughness
    events marked as vertical lines."""
    fig, ax = plt.subplots(figsize=(14, 5))
    if len(state):
        sb = _gap_break(state, ['water_level_m', 'tide_m'])
        ax.plot(sb.t_utc, sb.water_level_m, lw=0.8, color='k',
                label='invsnr water level')
        ax.plot(sb.t_utc, sb.tide_m, lw=0.8, color='C0', alpha=0.7,
                label='tide (Gr1kmTM)')
    seen = set()
    for _, ev in events.iterrows():
        trig = ev['trigger']
        col = _TRIG_COLOR.get(trig, 'C2')
        lbl = None if trig in seen else f'{trig} events'
        seen.add(trig)
        if trig == 'surge':
            ax.axvspan(ev['t_start_utc'], ev['t_end_utc'], color=col,
                       alpha=0.15, label=lbl)
        else:
            ax.axvline(ev['t_peak_utc'], color=col, lw=0.8, alpha=0.6, label=lbl)
    ax.set_ylabel('water level (m, MSL)'); ax.set_title('1 Hz pipeline overview')
    ax.legend(loc='upper right', fontsize=8); _style_time_axis(ax)
    fig.tight_layout(); fig.savefig(out, dpi=c.DPI); plt.close(fig)


def plot_roughness_timeline(state: pd.DataFrame, events: pd.DataFrame, out,
                            ratio_min: float) -> None:
    """Hourly roughness activity across the range: max rough_ratio and the
    count of distinct rough satellites per hour, with event peaks marked."""
    if not len(state):
        return
    t_lo, t_hi = state.t_utc.min(), state.t_utc.max()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    # Aggregate per-day to keep memory low, but bin per-hour for the plot.
    hr_max, hr_nsat = [], []
    d = t_lo.floor('D')
    while d <= t_hi:
        year, doy = d.year, int(d.strftime('%j'))
        p = _roughness_path(year, doy)
        if p.exists():
            df = pd.read_parquet(p, columns=['t_center_utc', 'sat', 'rough_ratio'])
            df['hr'] = df.t_center_utc.dt.floor('h')
            g = df.groupby('hr')
            hr_max.append(g.rough_ratio.max())
            flagged = df[df.rough_ratio >= ratio_min]
            hr_nsat.append(flagged.groupby('hr').sat.nunique())
        d += pd.Timedelta(days=1)
    if hr_max:
        m = pd.concat(hr_max); ax1.plot(m.index, m.values, lw=0.6, color='C3')
    ax1.axhline(ratio_min, color='grey', ls=':', lw=0.8)
    ax1.set_ylabel('max rough_ratio / hr')
    if hr_nsat:
        ns = pd.concat(hr_nsat); ax2.plot(ns.index, ns.values, lw=0.6, color='C1')
    ax2.set_ylabel(f'# sats >= {ratio_min:g} / hr')
    rough_ev = (events[events['trigger'] == 'roughness']
                if len(events) and 'trigger' in events.columns else events.iloc[:0])
    for _, ev in rough_ev.iterrows():
        for ax in (ax1, ax2):
            ax.axvline(ev['t_peak_utc'], color='C3', lw=0.8, alpha=0.5)
    ax1.set_title('Roughness activity (event peaks marked)')
    _style_time_axis(ax2)
    fig.tight_layout(); fig.savefig(out, dpi=c.DPI); plt.close(fig)


def plot_event_zoom(event: pd.Series, state: pd.DataFrame, outdir, rank: int,
                    ratio_min: float = 4.0, tide_dev_m: float = 0.75) -> None:
    """Per-event zoom. Bottom panel is always water level + tide. Top panel is
    the trigger's own evidence:
      - roughness: rough_ratio per satellite (the coherent burst)
      - surge:     eta = water level - tide, with the +/- threshold lines
    Window padding is trigger-scaled: minutes for a brief roughness burst,
    +/-12 h for a tide-scale surge so its context is visible."""
    is_surge = event['trigger'] == 'surge'
    pad = pd.Timedelta(seconds=SURGE_PAD_SEC if is_surge else ROUGH_PAD_SEC)
    t_lo = pd.Timestamp(event['t_start_utc']) - pad
    t_hi = pd.Timestamp(event['t_end_utc']) + pad
    t_peak = pd.Timestamp(event['t_peak_utc'])
    st = state[(state.t_utc >= t_lo) & (state.t_utc <= t_hi)]
    stb = _gap_break(st, ['eta_m', 'water_level_m', 'tide_m'])   # break lines at gaps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # --- top panel: the trigger's own signal ---
    if is_surge:
        if len(stb):
            ax1.plot(stb.t_utc, stb.eta_m, color='C0', lw=1.0, label='eta = WL - tide')
        ax1.axhline(tide_dev_m, color='grey', ls=':', lw=0.8)
        ax1.axhline(-tide_dev_m, color='grey', ls=':', lw=0.8)
        ax1.axhline(0, color='k', lw=0.4, alpha=0.3)
        ax1.set_ylabel('eta (m)'); ax1.legend(loc='upper right', fontsize=8)
    else:
        rough = _load_roughness_window(t_lo, t_hi)
        for sat, g in rough.groupby('sat'):
            ax1.plot(g.t_center_utc, g.rough_ratio, lw=0.7, alpha=0.8, label=f'sat {sat}')
        ax1.axhline(ratio_min, color='grey', ls=':', lw=0.8)
        if 0 < rough.sat.nunique() <= 12:
            ax1.legend(loc='upper right', fontsize=6, ncol=3)
        ax1.set_ylabel('rough_ratio')

    # --- bottom panel: water level + tide ---
    if len(stb):
        ax2.plot(stb.t_utc, stb.water_level_m, color='k', lw=1.0, label='water level')
        ax2.plot(stb.t_utc, stb.tide_m, color='C0', lw=1.0, alpha=0.7, label='tide')
        ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylabel('water level (m)')

    # Trim x to the STATE (WL/tide) extent so the bottom line reaches both edges
    # (its 5-min cadence is sparser than the top panel's, which stays dense and
    # fills edge-to-edge within this range). sharex applies it to both panels.
    if len(st):
        ax2.set_xlim(st.t_utc.min(), st.t_utc.max())
    # Roughness bottom panel: WL/tide barely move over the short window, so widen
    # the y-range to keep the two lines reading as close together (not exaggerated).
    if not is_surge and len(st):
        lo = min(st.water_level_m.min(), st.tide_m.min())
        hi = max(st.water_level_m.max(), st.tide_m.max())
        mid, half = (lo + hi) / 2, max(2 * (hi - lo), 0.5)
        ax2.set_ylim(mid - half, mid + half)

    for ax in (ax1, ax2):
        ax.axvspan(event['t_start_utc'], event['t_end_utc'],
                   color=_TRIG_COLOR.get(event['trigger'], 'C2'), alpha=0.18)
    peak = (f"{event['peak_tide_dev_m']:+.2f} m" if is_surge
            else f"{event['peak_rough_ratio']:.1f}x, {int(event['max_sats'])} sats")
    dur = float(event['duration_sec'])
    dur_str = f'{dur/3600:.1f} h' if dur >= 3600 else f'{dur:.0f} s'
    ax1.set_title(f"#{rank} {event['trigger']}  {t_peak:%Y-%m-%d %H:%M} UTC  "
                  f"({dur_str}, {peak})")
    ax2.set_xlabel('UTC')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.tight_layout()
    fn = outdir / f"event_{rank:02d}_{event['trigger']}.png"
    fig.savefig(fn, dpi=c.DPI); plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate(tag: str | None = None, *, rough_config=None) -> None:
    import detect
    tag = tag or _latest_tag()
    if tag is None:
        print('plots_roughness: no range with events found.')
        return
    rdir = c.RESULTS_DIR / 'range' / tag
    state = (pd.read_parquet(rdir / 'state.parquet')
             if (rdir / 'state.parquet').exists() else pd.DataFrame())
    events = (pd.read_parquet(rdir / 'events.parquet')
              if (rdir / 'events.parquet').exists() else pd.DataFrame())
    ratio_min = (rough_config or detect.RoughnessConfig()).rough_ratio_min
    tide_dev_m = detect.DetectorConfig().min_tide_dev_m

    pdir = c.PLOTS_DIR
    pdir.mkdir(parents=True, exist_ok=True)
    edir = pdir / 'events'
    edir.mkdir(parents=True, exist_ok=True)

    print(f'plots_roughness: tag {tag}  ({len(state):,} state rows, '
          f'{len(events)} events)')
    plot_overview(state, events, pdir / f'overview_{tag}.png')
    plot_roughness_timeline(state, events, pdir / f'roughness_timeline_{tag}.png',
                            ratio_min)
    if len(events):
        top = events.nlargest(TOP_N, 'confidence')
        for rank, (_, ev) in enumerate(top.iterrows(), 1):
            plot_event_zoom(ev, state, edir, rank, ratio_min, tide_dev_m)
    print(f'  wrote overview + timeline + {min(TOP_N, len(events))} event plots '
          f'to {pdir}')


if __name__ == '__main__':
    generate()
