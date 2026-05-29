"""Event detection for the 1 Hz pipeline. Two triggers, both interpretable:

  - surge:     the invsnr water level stays >= `min_tide_dev_m` away from the
               predicted tide — |rolling-median(eta)| over `tide_dev_window_sec`,
               where eta = water_level - tide. A sustained non-tidal offset
               (storm surge; or, in winter, sea-ice corruption of the retrieval).
  - roughness: a coherent multi-satellite burst of elevated SNR roughness — a
               wave train (see snr.process_arcs_roughness).

`detect_events(rough_df, state_df)` runs both against the invsnr state (the
water-level reference) + roughness obs and writes one `events.parquet` with a
`trigger` column in {'surge', 'roughness'}.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as c


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Surge trigger (on the invsnr state) + event clustering."""
    tide_dev_window_sec:    float = 3600.0   # 1 hr rolling median of eta
    min_tide_dev_m:         float = 0.75     # |water level - tide| threshold.
                                              # eta noise floor ~0.35 m -> ~2x.
    max_event_gap_sec:      float = 300.0     # merge fired samples within this gap
    min_event_duration_sec: float = 60.0      # drop ultra-short triggers
    max_state_gap_sec:      float = 1800.0    # a state gap larger than this breaks
                                              # a contiguous segment; the rolling
                                              # median never bridges it (no gap-edge
                                              # artifacts — e.g. winter ice gaps)
    surge_edge_guard_sec:   float = 10800.0   # don't flag within this of a segment
                                              # edge (data->NaN / NaN->data). 3 h =
                                              # invsnr knot spacing, the scale over
                                              # which the spline endpoints (and the
                                              # marginal ice-transition retrievals)
                                              # are least reliable.


@dataclass
class RoughnessConfig:
    """Roughness wave-train trigger (on the roughness obs). Defaults set from
    characterize_roughness.py on the 131-day 1 Hz set: calm rough_ratio tail is
    p99~2.7 / p99.9~4.8, and at ratio>=3 only ~1% of flagged bins have >=3 sats.
    4.0/4 requires BOTH clearly-anomalous roughness AND strong multi-sat
    coherence -> ~25 episodic events over doy 110-240. Operating point, not a
    ground-truth-validated threshold (loosen to 3.0/4 for recall, 5.0/4 for
    precision; see the sensitivity grid)."""
    rough_ratio_min:        float = 4.0   # per-sat roughness vs its per-arc median
    min_sats:               int   = 4     # distinct sats coherently rough in a bin
    bin_sec:                float = 30.0  # time bin for the coherence count
    max_event_gap_sec:      float = 60.0
    min_event_duration_sec: float = 20.0


# ---------------------------------------------------------------------------
# Trigger 1 — sustained water-level vs tide deviation (surge)
# ---------------------------------------------------------------------------

def detect_surges(state_df: pd.DataFrame,
                  config: DetectorConfig) -> pd.DataFrame:
    """Flag intervals where the invsnr water level departs from the tide:
    |rolling-median(eta)| >= `min_tide_dev_m` over `tide_dev_window_sec`,
    eta = water_level - tide. The rolling median enforces a *sustained* offset
    (a brief spike won't survive a 1-hr median).

    Gap-aware: the state is split into contiguous segments at gaps larger than
    `max_state_gap_sec` (e.g. winter days where invsnr retrieves nothing through
    ice). The rolling median is computed per segment so it never bridges a gap,
    and a sample is only flagged if its centered window sits fully inside its
    segment — otherwise a one-sided window at a gap edge produces phantom
    surges."""
    if state_df is None or state_df.empty or 'eta_m' not in state_df.columns:
        return pd.DataFrame()

    state = state_df.sort_values('t_utc').reset_index(drop=True).copy()
    window_str = f'{int(config.tide_dev_window_sec)}s'
    max_state_gap = pd.Timedelta(seconds=config.max_state_gap_sec)
    # guard back from each segment edge: at least half the rolling window (so it's
    # complete) and at least surge_edge_guard_sec (so the unreliable spline-endpoint
    # / ice-transition zone near a data<->NaN boundary can't flag).
    guard = pd.Timedelta(seconds=max(config.tide_dev_window_sec / 2,
                                     config.surge_edge_guard_sec))

    seg = (state['t_utc'].diff() > max_state_gap).cumsum()
    eta_smooth = np.full(len(state), np.nan)
    for _, g in state.groupby(seg):
        if len(g) < 3:
            continue
        m = (g.set_index('t_utc')['eta_m']
             .rolling(window_str, center=True, min_periods=3).median()
             .to_numpy(dtype=float, copy=True))
        inside = ((g['t_utc'] >= g['t_utc'].min() + guard) &
                  (g['t_utc'] <= g['t_utc'].max() - guard)).to_numpy()
        m[~inside] = np.nan                       # drop edge / transition-zone windows
        eta_smooth[g.index.to_numpy()] = m
    state['eta_smooth'] = eta_smooth

    flagged_mask = state['eta_smooth'].abs() >= config.min_tide_dev_m   # NaN -> False
    if not flagged_mask.any():
        return pd.DataFrame()

    flagged = state[flagged_mask].copy()
    max_gap = pd.Timedelta(seconds=config.max_event_gap_sec)
    flagged['event_id'] = (flagged['t_utc'].diff() > max_gap).cumsum()

    rows = []
    for _, grp in flagged.groupby('event_id', sort=True):
        t_start, t_end = grp['t_utc'].min(), grp['t_utc'].max()
        duration = (t_end - t_start).total_seconds()
        if duration < config.min_event_duration_sec:
            continue
        peak = grp.loc[grp['eta_smooth'].abs().idxmax()]
        rows.append({
            't_start_utc':           t_start,
            't_end_utc':             t_end,
            't_peak_utc':            peak['t_utc'],
            'duration_sec':          float(duration),
            'trigger':               'surge',
            'peak_tide_dev_m':       float(peak['eta_smooth']),
            'water_level_at_peak_m': float(peak['water_level_m']),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trigger 2 — coherent multi-sat roughness burst (wave train)
# ---------------------------------------------------------------------------

def detect_roughness_bursts(rough_df: pd.DataFrame,
                            config: RoughnessConfig | None = None) -> pd.DataFrame:
    """Flag coherent multi-sat roughness bursts. A wave train roughens the
    surface for many satellites at once; per-sat artifacts don't. Returns one
    row per event."""
    cfg = config or RoughnessConfig()
    if rough_df is None or rough_df.empty or 'rough_ratio' not in rough_df.columns:
        return pd.DataFrame()

    flagged = rough_df[rough_df['rough_ratio'] >= cfg.rough_ratio_min].copy()
    if flagged.empty:
        return pd.DataFrame()

    flagged['tbin'] = flagged['t_center_utc'].dt.floor(f'{int(cfg.bin_sec)}s')
    per_bin = flagged.groupby('tbin').agg(
        n_sats=('sat', 'nunique'),
        n_obs=('sat', 'size'),
        n_con=('constellation', 'nunique'),
        peak_ratio=('rough_ratio', 'max'),
    ).reset_index()

    coherent = per_bin[per_bin['n_sats'] >= cfg.min_sats].sort_values('tbin')
    if coherent.empty:
        return pd.DataFrame()

    gap = pd.Timedelta(seconds=cfg.max_event_gap_sec)
    coherent = coherent.copy()
    coherent['event_id'] = (coherent['tbin'].diff() > gap).cumsum()

    rows = []
    for _, g in coherent.groupby('event_id', sort=True):
        t_start = g['tbin'].min()
        t_end = g['tbin'].max() + pd.Timedelta(seconds=cfg.bin_sec)
        duration = (t_end - t_start).total_seconds()
        if duration < cfg.min_event_duration_sec:
            continue
        rows.append({
            't_start_utc':        t_start,
            't_end_utc':          t_end,
            't_peak_utc':         g.loc[g['peak_ratio'].idxmax(), 'tbin'],
            'duration_sec':       float(duration),
            'trigger':            'roughness',
            'peak_rough_ratio':   float(g['peak_ratio'].max()),
            'max_sats':           int(g['n_sats'].max()),
            'max_constellations': int(g['n_con'].max()),
            'confidence':         float(g['peak_ratio'].max() * np.sqrt(duration / 60.0)),
        })
    return (pd.DataFrame(rows).sort_values('confidence', ascending=False)
            .reset_index(drop=True) if rows else pd.DataFrame())


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_events(rough_df: pd.DataFrame, state_df: pd.DataFrame, *,
                  config: DetectorConfig | None = None,
                  rough_config: RoughnessConfig | None = None,
                  save_to: Path | str | None = None) -> pd.DataFrame:
    """Run both triggers against the invsnr state (water-level reference) and
    the roughness obs; return one event log (column `trigger`). Surge and
    roughness events are kept as distinct rows, ranked by `confidence`."""
    cfg = config or DetectorConfig()
    rcfg = rough_config or RoughnessConfig()

    surges = detect_surges(state_df, cfg)
    if len(surges):
        surges = surges.copy()
        surges['confidence'] = ((surges['peak_tide_dev_m'].abs() / cfg.min_tide_dev_m)
                                * np.sqrt(surges['duration_sec'] / 60.0))
    rough = detect_roughness_bursts(rough_df, rcfg)

    frames = [f for f in (surges, rough) if len(f)]
    if frames:
        events = (pd.concat(frames, ignore_index=True, sort=False)
                  .sort_values('confidence', ascending=False).reset_index(drop=True))
        events['event_id'] = range(len(events))
    else:
        events = pd.DataFrame()

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(save_to, index=False)
    return events
