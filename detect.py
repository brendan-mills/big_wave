"""Event detection on the smoothed water-level state + raw observations.

Three independent triggers, all expressed in absolute meters so the thresholds
are physically interpretable:

  - jump:     `state.water_level_m` changes by ≥ `min_jump_m` within a
              `state_jump_window_sec` rolling window. Catches fast level
              steps over minutes.
  - variance: obs scatter reaches ≥ `min_burst_amp_m` BOTH above and below
              the state line within `var_window_sec` — a "straddle"
              amplitude. Catches symmetric oscillation (wave trains). A
              one-sided spectral-artifact cluster (all obs below the line)
              does NOT fire, because the straddle requires a high excursion
              too. Guards against data-gap artifacts by requiring
              ≥ `min_var_samples` in the window, and drops single-obs
              outliers with `|mahal| > mahal_clip` before the rolling stat.
  - surge:    the smoothed water level departs from the predicted tide —
              |`state.eta_m`| (= water level − tide) stays ≥ `min_tide_dev_m`
              over a `tide_dev_window_sec` rolling median. Catches sustained,
              hours-long non-tidal offsets (storm surge, or — in winter —
              sea-ice corruption of the retrieval).

Any time interval that fires a trigger becomes a candidate event. Overlapping
(or near-overlapping, within `max_event_gap_sec`) candidates merge into one
event, with `trigger` recorded as a single label ('jump', 'variance',
'surge') or a '+'-joined combination ('jump+surge', etc.).

Output is a single `events.parquet`. The per-observation innovation log
is saved as a sidecar `innov.parquet` for diagnostics.
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
    """Three trigger pathways, two clustering knobs, one geometry constant."""
    # --- Trigger A: state-level jump ---
    state_jump_window_sec: float = 600.0   # 10 min look-back
    min_jump_m:            float = 1.0     # |Δwater_level| threshold

    # --- Trigger B: bidirectional obs scatter ("straddle") around state ---
    var_window_sec:        float = 300.0   # 5 min rolling
    min_burst_amp_m:       float = 2.5     # obs must reach ±this around the
                                            # state line (both above AND below).
                                            # Summer scatter floor ≈ 0.95 m, so
                                            # this is ~2.5× background.
    min_var_samples:       int   = 15      # require this many obs in window
                                            # (guards against data-gap inflation)
    mahal_clip:            float = 5.0     # drop obs with |innov/σ| > this
                                            # before the rolling stat — removes
                                            # single-obs spectral artifacts

    # --- Trigger C: sustained deviation of the spline from the tide ---
    tide_dev_window_sec:   float = 3600.0  # 1 hr rolling median of eta
    min_tide_dev_m:        float = 0.75    # |water level − tide| threshold.
                                            # eta noise floor ≈ 0.35 m (p95), so
                                            # this is ~2× background.

    # --- Event clustering ---
    max_event_gap_sec:     float = 300.0   # merge fired intervals within this gap
    min_event_duration_sec: float = 60.0   # drop ultra-short triggers

    antenna_msl_m:         float = c.ANTENNA_MSL_M


# ---------------------------------------------------------------------------
# Per-observation innovations (used by both the variance detector and
# downstream plotting)
# ---------------------------------------------------------------------------

def compute_innovations(obs_df: pd.DataFrame,
                        state_df: pd.DataFrame,
                        tide_model,
                        antenna_msl: float = c.ANTENNA_MSL_M
                        ) -> pd.DataFrame:
    """For each raw windowed obs, compute its innovation against the state
    at that time (linear interp of state.eta_m / eta_sigma_m at obs times).

    Adds columns: `eta_obs, eta_pred, eta_pred_sigma, innov, sigma_combined,
    mahal, in_state_range`. Used by `detect_bursts` and by external plotting.
    """
    if obs_df.empty or state_df.empty:
        return pd.DataFrame()

    obs = obs_df.sort_values('t_center_utc').reset_index(drop=True)
    state = state_df.sort_values('t_utc').reset_index(drop=True)

    tide_at_obs = np.asarray(tide_model.predict(obs['t_center_utc'].tolist()))

    state_t = (state['t_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
               .astype('datetime64[ns]').astype('int64').to_numpy())
    obs_t   = (obs['t_center_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
               .astype('datetime64[ns]').astype('int64').to_numpy())
    eta_pred       = np.interp(obs_t, state_t, state['eta_m'].to_numpy())
    eta_pred_sigma = np.interp(obs_t, state_t, state['eta_sigma_m'].to_numpy())

    in_range = (obs_t >= state_t[0]) & (obs_t <= state_t[-1])

    eta_obs = antenna_msl - tide_at_obs - obs['rh'].to_numpy()
    sigma_obs = obs['sigma'].to_numpy()
    sigma_combined = np.sqrt(sigma_obs**2 + eta_pred_sigma**2)
    innov = eta_obs - eta_pred
    mahal = np.where(sigma_combined > 0, innov / sigma_combined, np.nan)

    out = obs.copy()
    out['eta_obs']        = eta_obs
    out['eta_pred']       = eta_pred
    out['eta_pred_sigma'] = eta_pred_sigma
    out['innov']          = innov
    out['sigma_combined'] = sigma_combined
    out['mahal']          = mahal
    out['in_state_range'] = in_range
    return out


# ---------------------------------------------------------------------------
# Trigger A — state-level jumps
# ---------------------------------------------------------------------------

def detect_jumps(state_df: pd.DataFrame,
                 config: DetectorConfig) -> pd.DataFrame:
    """Flag intervals where the state's water level changes by ≥ `min_jump_m`
    within `state_jump_window_sec`. Uses rolling max−min as the spread metric.
    Returns one row per detected jump event."""
    if state_df is None or state_df.empty:
        return pd.DataFrame()

    state = state_df.sort_values('t_utc').reset_index(drop=True).copy()
    indexed = state.set_index('t_utc')
    window_str = f'{int(config.state_jump_window_sec)}s'

    roll_max = indexed['water_level_m'].rolling(window_str, min_periods=2).max()
    roll_min = indexed['water_level_m'].rolling(window_str, min_periods=2).min()
    state['rolling_spread'] = (roll_max - roll_min).to_numpy()

    flagged_mask = state['rolling_spread'] >= config.min_jump_m
    if not flagged_mask.any():
        return pd.DataFrame()

    flagged = state[flagged_mask].copy()
    max_gap = pd.Timedelta(seconds=config.max_event_gap_sec)
    flagged['event_id'] = (flagged['t_utc'].diff() > max_gap).cumsum()

    rows = []
    for _, grp in flagged.groupby('event_id', sort=True):
        # Event window: from first flag minus look-back, through last flag
        t_first = grp['t_utc'].min()
        t_last  = grp['t_utc'].max()
        t_start = t_first - pd.Timedelta(seconds=config.state_jump_window_sec)
        t_end   = t_last
        duration = (t_end - t_start).total_seconds()
        if duration < config.min_event_duration_sec:
            continue
        peak = grp.loc[grp['rolling_spread'].idxmax()]
        rows.append({
            't_start_utc':           t_start,
            't_end_utc':             t_end,
            't_peak_utc':            peak['t_utc'],
            'duration_sec':          float(duration),
            'trigger':               'jump',
            'delta_m':               float(peak['rolling_spread']),
            'water_level_at_peak_m': float(peak['water_level_m']),
            'peak_burst_amp_m':      np.nan,
            'n_obs_in_window':       np.nan,
            'peak_tide_dev_m':       np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trigger B — bidirectional obs scatter ("straddle") around the state
# ---------------------------------------------------------------------------

# Upper/lower quantiles used to gauge how far obs reach above and below the
# state line. Quantiles (not max/min) keep the metric robust to the few wild
# obs that survive the mahal clip.
_STRADDLE_Q = 0.90


def detect_bursts(innov_df: pd.DataFrame,
                  config: DetectorConfig) -> pd.DataFrame:
    """Flag intervals where obs scatter reaches ≥ `min_burst_amp_m` BOTH
    above and below the state line — a "straddle" amplitude.

    For each rolling window we take the upper quantile (q90) and lower
    quantile (q10) of the innovations. `straddle = min(q90, −q10)`:

      - symmetric wave train (obs swing ±A around the line): q90 ≈ +0.8A,
        q10 ≈ −0.8A → straddle ≈ 0.8A  → fires.
      - one-sided artifact cluster (all obs below the line, e.g. at −5 m):
        q90 ≤ 0 → straddle ≤ 0  → never fires.

    This is what distinguishes a real wave from a spectral-artifact pile-up,
    which plain std cannot do (a one-sided cluster has high std too).
    """
    if innov_df is None or innov_df.empty:
        return pd.DataFrame()

    df = innov_df[innov_df['in_state_range']].copy()
    if df.empty:
        return pd.DataFrame()

    # Drop single-obs outliers (typically spectral artifacts at RH boundary
    # — innov 4-6 m, |mahal| > 10). A real wave event's individual obs sit
    # within a few σ of the state; it's the *aggregate* scatter we want.
    n_pre = len(df)
    df = df[df['mahal'].abs() <= config.mahal_clip]
    n_clipped = n_pre - len(df)
    if n_clipped:
        print(f'  detect_bursts: clipped {n_clipped}/{n_pre} obs '
              f'(|mahal| > {config.mahal_clip})')
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('t_center_utc').reset_index(drop=True)

    indexed = df.set_index('t_center_utc')
    window_str = f'{int(config.var_window_sec)}s'
    roll = indexed['innov'].rolling(
        window_str, center=True, min_periods=config.min_var_samples)
    q_hi = roll.quantile(_STRADDLE_Q).to_numpy()
    q_lo = roll.quantile(1 - _STRADDLE_Q).to_numpy()
    rolling_n = indexed['innov'].rolling(
        window_str, center=True, min_periods=1).count().to_numpy()
    df['straddle'] = np.minimum(q_hi, -q_lo)
    df['var_n']    = rolling_n

    flagged_mask = ((df['straddle'] >= config.min_burst_amp_m)
                    & (df['var_n'] >= config.min_var_samples)
                    & np.isfinite(df['straddle']))
    if not flagged_mask.any():
        return pd.DataFrame()

    flagged = df[flagged_mask].copy()
    max_gap = pd.Timedelta(seconds=config.max_event_gap_sec)
    flagged['event_id'] = (flagged['t_center_utc'].diff() > max_gap).cumsum()

    rows = []
    for _, grp in flagged.groupby('event_id', sort=True):
        t_start  = grp['t_center_utc'].min()
        t_end    = grp['t_center_utc'].max()
        duration = (t_end - t_start).total_seconds() + config.var_window_sec
        if duration < config.min_event_duration_sec:
            continue
        peak     = grp.loc[grp['straddle'].idxmax()]
        rows.append({
            't_start_utc':           t_start,
            't_end_utc':             t_end,
            't_peak_utc':            peak['t_center_utc'],
            'duration_sec':          float(duration),
            'trigger':               'variance',
            'delta_m':               np.nan,
            'water_level_at_peak_m': np.nan,
            'peak_burst_amp_m':      float(peak['straddle']),
            'n_obs_in_window':       int(peak['var_n']),
            'peak_tide_dev_m':       np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trigger C — sustained deviation of the spline from the predicted tide
# ---------------------------------------------------------------------------

def detect_surges(state_df: pd.DataFrame,
                  config: DetectorConfig) -> pd.DataFrame:
    """Flag intervals where the smoothed water level departs from the tide:
    |rolling-median(eta)| ≥ `min_tide_dev_m` over `tide_dev_window_sec`,
    where `eta = water_level − tide`.

    The rolling median enforces a *sustained* offset (a brief spike won't
    survive a 1-hr median), separating real surges from momentary noise.
    """
    if state_df is None or state_df.empty or 'eta_m' not in state_df.columns:
        return pd.DataFrame()

    state = state_df.sort_values('t_utc').reset_index(drop=True).copy()
    indexed = state.set_index('t_utc')
    window_str = f'{int(config.tide_dev_window_sec)}s'
    eta_smooth = indexed['eta_m'].rolling(
        window_str, center=True, min_periods=3).median()
    state['eta_smooth'] = eta_smooth.to_numpy()

    flagged_mask = state['eta_smooth'].abs() >= config.min_tide_dev_m
    if not flagged_mask.any():
        return pd.DataFrame()

    flagged = state[flagged_mask].copy()
    max_gap = pd.Timedelta(seconds=config.max_event_gap_sec)
    flagged['event_id'] = (flagged['t_utc'].diff() > max_gap).cumsum()

    rows = []
    for _, grp in flagged.groupby('event_id', sort=True):
        t_start  = grp['t_utc'].min()
        t_end    = grp['t_utc'].max()
        duration = (t_end - t_start).total_seconds()
        if duration < config.min_event_duration_sec:
            continue
        # Peak = sample with the largest |smoothed eta| in the cluster
        peak = grp.loc[grp['eta_smooth'].abs().idxmax()]
        rows.append({
            't_start_utc':           t_start,
            't_end_utc':             t_end,
            't_peak_utc':            peak['t_utc'],
            'duration_sec':          float(duration),
            'trigger':               'surge',
            'delta_m':               np.nan,
            'water_level_at_peak_m': float(peak['water_level_m']),
            'peak_burst_amp_m':      np.nan,
            'n_obs_in_window':       np.nan,
            'peak_tide_dev_m':       float(peak['eta_smooth']),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge overlapping jumps + bursts into one event list
# ---------------------------------------------------------------------------

_METRIC_COLS = ('delta_m', 'water_level_at_peak_m', 'peak_burst_amp_m',
                'n_obs_in_window', 'peak_tide_dev_m')


def _pathway_strength(delta_m, burst_amp_m, tide_dev_m,
                      config: DetectorConfig) -> float:
    """Relative strength of an event: how many ×-thresholds it clears on the
    strongest of its three pathways. NaN-safe (each pure trigger leaves the
    other metrics NaN)."""
    jump  = abs(delta_m) / config.min_jump_m if pd.notna(delta_m) else 0.0
    var   = burst_amp_m / config.min_burst_amp_m if pd.notna(burst_amp_m) else 0.0
    surge = abs(tide_dev_m) / config.min_tide_dev_m if pd.notna(tide_dev_m) else 0.0
    return max(jump, var, surge)


def _combine_triggers(a: str, b: str) -> str:
    """Union two (possibly already '+'-joined) trigger labels, sorted."""
    return '+'.join(sorted(set(a.split('+')) | set(b.split('+'))))


def _merge_overlapping(frames: list[pd.DataFrame],
                       config: DetectorConfig) -> pd.DataFrame:
    """Concatenate all trigger frames; merge any whose time intervals overlap
    (within `max_event_gap_sec`). A merged event's `trigger` is the '+'-joined
    union of its constituents ('jump', 'variance', 'surge', 'jump+surge', …)."""
    parts = [df for df in frames if df is not None and not df.empty]
    if not parts:
        return pd.DataFrame()
    all_ev = pd.concat(parts, ignore_index=True) \
               .sort_values('t_start_utc').reset_index(drop=True)

    max_gap = pd.Timedelta(seconds=config.max_event_gap_sec)
    merged = []
    cur = all_ev.iloc[0].to_dict()
    for i in range(1, len(all_ev)):
        nxt = all_ev.iloc[i]
        if pd.Timestamp(nxt['t_start_utc']) <= pd.Timestamp(cur['t_end_utc']) + max_gap:
            # Merge nxt into cur
            cur['t_end_utc'] = max(pd.Timestamp(cur['t_end_utc']),
                                    pd.Timestamp(nxt['t_end_utc']))
            cur['duration_sec'] = (pd.Timestamp(cur['t_end_utc'])
                                   - pd.Timestamp(cur['t_start_utc'])).total_seconds()

            if cur['trigger'] != nxt['trigger']:
                cur['trigger'] = _combine_triggers(cur['trigger'], nxt['trigger'])

            # Move the peak time to whichever pathway is strongest
            cur_strength = _pathway_strength(cur.get('delta_m'),
                                             cur.get('peak_burst_amp_m'),
                                             cur.get('peak_tide_dev_m'), config)
            nxt_strength = _pathway_strength(nxt.get('delta_m'),
                                             nxt.get('peak_burst_amp_m'),
                                             nxt.get('peak_tide_dev_m'), config)
            if nxt_strength > cur_strength:
                cur['t_peak_utc'] = nxt['t_peak_utc']

            # Carry over per-trigger metrics (NaN-aware combine)
            for col in _METRIC_COLS:
                nxt_val = nxt[col]
                if pd.isna(nxt_val):
                    continue
                cur_val = cur.get(col)
                if pd.isna(cur_val):
                    cur[col] = nxt_val
                elif col in ('delta_m', 'peak_burst_amp_m', 'peak_tide_dev_m'):
                    # take whichever has larger magnitude
                    if abs(nxt_val) > abs(cur_val):
                        cur[col] = nxt_val
        else:
            merged.append(cur)
            cur = nxt.to_dict()
    merged.append(cur)

    out = pd.DataFrame(merged)
    for col in _METRIC_COLS:               # ensure all metric cols exist
        if col not in out.columns:
            out[col] = np.nan
    out['event_id'] = range(len(out))

    # Confidence: strongest of the three pathways × √(duration in min)
    jump_str  = (out['delta_m'].abs() / config.min_jump_m).fillna(0)
    var_str   = (out['peak_burst_amp_m'] / config.min_burst_amp_m).fillna(0)
    surge_str = (out['peak_tide_dev_m'].abs() / config.min_tide_dev_m).fillna(0)
    pathway   = pd.concat([jump_str, var_str, surge_str], axis=1).max(axis=1)
    out['confidence'] = pathway * np.sqrt(out['duration_sec'] / 60.0)

    return out.sort_values('confidence', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_events(obs_df: pd.DataFrame, state_df: pd.DataFrame,
                  tide_model, *,
                  config: DetectorConfig | None = None,
                  save_to: Path | str | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both detectors, merge into one event log.

    Returns (events_df, innov_df). `save_to` writes the events parquet
    and a sibling `innov.parquet` for downstream plotting / inspection.
    """
    cfg = config or DetectorConfig()
    innov = compute_innovations(obs_df, state_df, tide_model, cfg.antenna_msl_m)
    if innov.empty:
        return pd.DataFrame(), pd.DataFrame()

    jumps  = detect_jumps(state_df, cfg)
    bursts = detect_bursts(innov, cfg)
    surges = detect_surges(state_df, cfg)
    events = _merge_overlapping([jumps, bursts, surges], cfg)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(save_to, compression='snappy', index=False)
        innov_path = save_to.with_name('innov.parquet')
        innov.to_parquet(innov_path, compression='snappy', index=False)
    return events, innov
