"""Rogue-wave / anomaly detection on top of the windowed obs + KF state.

Layer 4 of the pipeline. Consumes:
  - raw per-window observations (`snr.process_arcs_windowed`)
  - smoothed KF state (`estimate.run_batch`)

Produces:
  - `innov_df`: per-obs innovation against the KF state
  - `events_df`: time windows where multiple satellites agreed on a
    coherent water-level anomaly — candidate rogue-wave / surge events.

The discrimination relies on spatial + temporal coherence: real events
are seen by many satellites at near-simultaneous times with the same sign
of departure from the smoothed state; noise hits one satellite at a time.
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
class DetectionConfig:
    """Tuning knobs for spatial+temporal anomaly detection.

    Eight knobs total. An event candidate is a time bin where:

      1. ≥ `min_sats_per_bin` sats show |mahal| > `mahal_threshold` innov,
      2. the same-sign cluster outnumbers the opposite by ≥ 2× (no flag),
      3. within that cluster, |median_innov|/std > `coherence_ratio_min`
         (sats agree on the *value*, not just the sign),
      4. |median_innov| > `min_amplitude_m` (hard floor — "I only care
         about events at least this big"), AND
      5. |median_innov| > `snr_min` × `local_noise_m` (rolling 30-min std
         of innovations) — adaptive to current conditions.

    Adjacent candidate bins within `max_event_gap_sec` merge into one event.

    **Defaults are tuned for iceberg-calving wave trains** — sustained,
    high-amplitude, mixed-sign oscillatory events lasting 3–20 minutes:
    `min_amplitude_m=1.0` catches mid-distance calving (sub-meter events
    from far away are below our noise floor anyway), `max_event_gap_sec=240`
    keeps multi-minute wave trains as one event instead of fragmenting them.
    The events table includes `n_sign_flips` so monotonic events (likely
    artifacts) can be distinguished from oscillatory wave trains.
    """
    bin_sec:           float = 60.0   # innovation clustering window
    mahal_threshold:   float = 2.5    # |mahal| flag — raised from 2.0 to
                                      # match wider per-obs σ from 180s windows
    min_sats_per_bin:  int   = 3      # agreeing sats needed to flag a bin
    coherence_ratio_min: float = 2.5  # within-cluster |median|/std must exceed
    min_amplitude_m:   float = 1.0    # hard floor on |median_innov| (m).
                                      # 1.0 catches mid-distance calving;
                                      # raise to ignore smaller events.
    snr_min:           float = 2.0    # |innov| > snr × local rolling-std
    max_event_gap_sec: float = 240.0  # gap that merges adjacent candidate bins
                                      # — 240s lets multi-minute wave trains
                                      # stay one event instead of fragmenting.
    antenna_msl_m:     float = c.ANTENNA_MSL_M


# ---------------------------------------------------------------------------
# Per-observation innovations
# ---------------------------------------------------------------------------

def compute_innovations(obs_df: pd.DataFrame,
                        state_df: pd.DataFrame,
                        tide_model,
                        antenna_msl: float = c.ANTENNA_MSL_M
                        ) -> pd.DataFrame:
    """For each raw windowed obs, compute its innovation against the KF state
    at that time. Linearly interpolates state at obs timestamps.

    Returns a copy of obs_df with these added columns:
        eta_obs        - observed η = antenna_msl - tide - rh
        eta_pred       - KF state η interpolated at the obs time
        eta_pred_sigma - KF state σ_η interpolated at the obs time
        innov          - eta_obs - eta_pred
        sigma_combined - sqrt(sigma_obs² + eta_pred_sigma²)
        mahal          - signed innov / sigma_combined
        is_anomalous   - |mahal| > threshold (filled later by caller)
    """
    if obs_df.empty or state_df.empty:
        return pd.DataFrame()

    obs = obs_df.sort_values('t_center_utc').reset_index(drop=True)
    state = state_df.sort_values('t_utc').reset_index(drop=True)

    # Vectorized tide prediction at obs times
    tide_at_obs = np.asarray(tide_model.predict(obs['t_center_utc'].tolist()))

    # Linear interp KF state at obs times. Strip tz first because
    # tz-aware datetime64 can't go straight to int64.
    state_t = (state['t_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
               .astype('datetime64[ns]').astype('int64').to_numpy())
    obs_t   = (obs['t_center_utc'].dt.tz_convert('UTC').dt.tz_localize(None)
               .astype('datetime64[ns]').astype('int64').to_numpy())
    eta_pred       = np.interp(obs_t, state_t, state['eta_m'].to_numpy())
    eta_pred_sigma = np.interp(obs_t, state_t, state['eta_sigma_m'].to_numpy())

    # Clip obs that fall outside the state time range — interp would extrapolate
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

    # Local background noise: rolling std of innovations over ±15 min using
    # all obs (not just anomalous). This responds to degraded reflector
    # conditions (sea ice, snow, rough surface) — the cross-sat scatter
    # naturally grows even though no single sat is "wrong". Used downstream
    # to require event amplitude > N × local noise.
    indexed = out.set_index('t_center_utc').sort_index()
    rolling_std = indexed['innov'].rolling('1800s', center=True).std()
    # Re-align to the original row order (out was already sorted by time)
    out['local_noise_m'] = rolling_std.reindex(out['t_center_utc']).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Temporal clustering
# ---------------------------------------------------------------------------

def _bin_anomalies(innov_df: pd.DataFrame, config: DetectionConfig
                   ) -> pd.DataFrame:
    """Aggregate anomalous obs into time bins, applying the four event gates.

    Returns one row per surviving candidate bin.
    """
    bin_freq = f'{int(config.bin_sec)}s'
    bin_td = pd.Timedelta(seconds=config.bin_sec)

    anom = innov_df[
        innov_df['in_state_range']
        & (innov_df['mahal'].abs() >= config.mahal_threshold)
    ].copy()
    if anom.empty:
        return pd.DataFrame()

    anom['_bin_start'] = anom['t_center_utc'].dt.floor(bin_freq)
    anom['_sign']      = np.sign(anom['innov']).astype(int)

    rows = []
    for bin_start, group in anom.groupby('_bin_start', sort=True):
        pos = group[group['_sign'] > 0]
        neg = group[group['_sign'] < 0]
        n_pos, n_neg = pos['sat'].nunique(), neg['sat'].nunique()

        # Gate 1: a clearly dominant sign cluster (≥ 2× the opposite, and
        # opposite must have less than half the dominant)
        if n_pos >= 2 * max(n_neg, 1) and n_pos > n_neg:
            dominant, sign, n_sats = pos, +1, n_pos
        elif n_neg >= 2 * max(n_pos, 1) and n_neg > n_pos:
            dominant, sign, n_sats = neg, -1, n_neg
        else:
            continue

        # Gate 2: need enough sats in the dominant cluster
        if n_sats < config.min_sats_per_bin:
            continue

        median_innov = float(dominant['innov'].median())

        # Gate 3a: hard amplitude floor — "only care about events ≥ this size"
        if abs(median_innov) < config.min_amplitude_m:
            continue

        # Gate 3: sats must agree on the *value*, not just the sign
        if len(dominant) > 1:
            dom_std = float(dominant['innov'].std(ddof=0))
            if dom_std > 0:
                coherence_ratio = abs(median_innov) / dom_std
                if coherence_ratio < config.coherence_ratio_min:
                    continue
            else:
                coherence_ratio = float('inf')
        else:
            coherence_ratio = float('inf')

        # Gate 4: signal-to-noise vs the rolling background — the only
        # adaptive piece. local_noise is the ±15-min innov std at this bin.
        local_noise = float(group['local_noise_m'].median())
        if not np.isfinite(local_noise):
            local_noise = 0.0
        if local_noise > 0 and abs(median_innov) < config.snr_min * local_noise:
            continue

        rows.append({
            't_bin_start':    bin_start,
            't_bin_center':   bin_start + bin_td / 2,
            'n_sats_pos':     int(n_pos),
            'n_sats_neg':     int(n_neg),
            'n_sats':         int(n_sats),
            'n_obs':          int(len(dominant)),
            'sign':           int(sign),
            'median_innov':   median_innov,
            'max_abs_mahal':  float(dominant['mahal'].abs().max()),
            'coherence_ratio': coherence_ratio,
            'local_noise_m':  local_noise,
            'snr':            (abs(median_innov) / local_noise
                                if local_noise > 0 else float('inf')),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event assembly
# ---------------------------------------------------------------------------

def _identify_events(bin_stats: pd.DataFrame, config: DetectionConfig
                      ) -> pd.DataFrame:
    """Stitch contiguous candidate bins (within `max_event_gap_sec`) into
    single events. Compute event-level summary stats."""
    if bin_stats.empty:
        return pd.DataFrame()

    bins = bin_stats.sort_values('t_bin_start').reset_index(drop=True)
    max_gap = pd.Timedelta(seconds=config.max_event_gap_sec)

    # Group bins into events by gap-based clustering
    event_id = (bins['t_bin_start'].diff() > max_gap).cumsum()
    bins = bins.assign(event_id=event_id.values)

    events = []
    for eid, grp in bins.groupby('event_id', sort=True):
        grp_sorted = grp.sort_values('t_bin_start')
        peak_idx = grp_sorted['median_innov'].abs().idxmax()
        peak = grp_sorted.loc[peak_idx]

        # Count sign reversals across this event's bins. A wave train
        # (calving, tsunami, swell group) shows alternating crest/trough
        # bins → ≥1 flip. A monotonic event (single large rise or fall) has
        # 0 flips and is more likely to be a noise/multipath artifact.
        signs = grp_sorted['sign'].to_numpy()
        n_sign_flips = int((np.diff(signs) != 0).sum()) if len(signs) > 1 else 0

        events.append({
            'event_id':    int(eid),
            't_start_utc': grp_sorted['t_bin_start'].min(),
            't_end_utc':   grp_sorted['t_bin_start'].max() + pd.Timedelta(seconds=config.bin_sec),
            't_peak_utc':  peak['t_bin_center'],
            'duration_sec': float((grp_sorted['t_bin_start'].max()
                                    - grp_sorted['t_bin_start'].min()).total_seconds()
                                   + config.bin_sec),
            'n_bins':      int(len(grp_sorted)),
            'amplitude_m': float(peak['median_innov']),
            'direction':   'rise' if peak['sign'] > 0 else 'fall',
            'n_sign_flips': n_sign_flips,
            'oscillates':  bool(n_sign_flips >= 1),
            'n_sats_peak': int(grp_sorted['n_sats'].max()),
            'n_obs_total': int(grp_sorted['n_obs'].sum()),
            'max_mahal':   float(grp_sorted['max_abs_mahal'].max()),
        })

    ev = pd.DataFrame(events)
    if ev.empty:
        return ev

    # Heuristic confidence score: scales with #sats, #bins, amplitude,
    # and oscillation (wave trains get a 1.5× bonus over monotonic events).
    osc_bonus = np.where(ev['oscillates'], 1.5, 1.0)
    ev['confidence'] = (ev['n_sats_peak']
                        * np.sqrt(ev['n_bins'])
                        * np.log10(1.0 + ev['amplitude_m'].abs() * 10.0)
                        * osc_bonus)
    return ev.sort_values('confidence', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def detect_events(obs_df: pd.DataFrame, state_df: pd.DataFrame,
                  tide_model, *,
                  config: DetectionConfig | None = None,
                  save_to: Path | str | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: innovations + binning + event assembly.

    Returns (events_df, innov_df).
    `save_to` writes the events parquet (sibling `_innov.parquet` for the
    full innovation log).
    """
    cfg = config or DetectionConfig()
    innov = compute_innovations(obs_df, state_df, tide_model, cfg.antenna_msl_m)
    if innov.empty:
        return pd.DataFrame(), pd.DataFrame()

    innov['is_anomalous'] = innov['mahal'].abs() >= cfg.mahal_threshold

    bin_stats = _bin_anomalies(innov, cfg)
    events    = _identify_events(bin_stats, cfg)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(save_to, compression='snappy', index=False)
        # Sibling innov.parquet: swap 'events' for 'innov' in the stem.
        # Falls back to bare 'innov.parquet' next to events.parquet when the
        # stem is just 'events' (whole-stem replacement leaves nothing to
        # disambiguate, so we use the bare name).
        new_stem = save_to.stem.replace('events', 'innov')
        if new_stem == save_to.stem:
            new_stem = 'innov'
        innov_path = save_to.with_name(new_stem + save_to.suffix)
        innov.to_parquet(innov_path, compression='snappy', index=False)
    return events, innov


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from tide import GreenlandTideModel

    YEAR, DOYS = 2026, [1, 2, 3]

    obs_path   = (c.RESULTS_DIR / f'{YEAR}' / 'windowed' /
                  f'{DOYS[0]:03d}-{DOYS[-1]:03d}_obs.parquet')
    state_path = (c.RESULTS_DIR / f'{YEAR}' / 'state' /
                  f'{DOYS[0]:03d}-{DOYS[-1]:03d}_state.parquet')
    if not obs_path.exists() or not state_path.exists():
        raise SystemExit('Need raw obs + state parquets — run estimate.py first.')

    obs   = pd.read_parquet(obs_path)
    state = pd.read_parquet(state_path)
    print(f'Loaded {len(obs):,} raw obs and {len(state):,} state rows.')

    tm = GreenlandTideModel(c.LAT, c.LON)
    cfg = DetectionConfig()

    save = c.RESULTS_DIR / f'{YEAR}' / 'events' / f'{DOYS[0]:03d}-{DOYS[-1]:03d}_events.parquet'
    events, innov = detect_events(obs, state, tm, config=cfg, save_to=save)

    print(f'\nInnovation summary across {len(innov):,} obs:')
    print(f'  median |innov| : {innov.innov.abs().median()*100:5.1f} cm')
    print(f'  95th pct |innov|: {innov.innov.abs().quantile(0.95)*100:5.1f} cm')
    print(f'  max |innov|    : {innov.innov.abs().max()*100:5.1f} cm')
    print(f'  obs flagged (|mahal| ≥ {cfg.mahal_threshold}): '
          f'{int(innov.is_anomalous.sum()):,}  '
          f'({100*innov.is_anomalous.mean():.1f}%)')

    print(f'\nDetected events: {len(events)}')
    if not events.empty:
        print(f'  duration range : {events.duration_sec.min():.0f} – '
              f'{events.duration_sec.max():.0f} s')
        print(f'  amplitude range: {events.amplitude_m.min():+.2f} – '
              f'{events.amplitude_m.max():+.2f} m')
        print(f'\nTop 10 events by confidence:')
        show = ['t_peak_utc', 'duration_sec', 'amplitude_m', 'direction',
                'n_sats_peak', 'n_bins', 'max_mahal', 'confidence']
        print(events[show].head(10).to_string(index=False))
    print(f'\nSaved: {save.relative_to(c.PROJECT_DIR)}')
