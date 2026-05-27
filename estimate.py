"""Tide-residual Kalman filter for sea-level estimation.

Layer 3 of the pipeline: consumes the per-arc observation parquets produced
by `pipeline.py` and the deterministic tide prior from `tide.py`, produces
a continuous state estimate `η(t) ± σ(t)` plus a log of any observations
rejected by the chi-square gate.

The same `TideKalman` class drives both batch replay (`run_batch`) and the
future streaming runtime (`run_streaming`) — the only difference is who
feeds it observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

import config as c


def _sibling(out_path: Path, key: str, replacement: str) -> Path:
    """Return a sibling parquet whose name swaps `key` for `replacement` in
    the stem. Works for both `state.parquet -> gated.parquet` (whole-stem)
    and `001-031_state.parquet -> 001-031_gated.parquet` (suffix-pattern)."""
    new_stem = out_path.stem.replace(key, replacement)
    if new_stem == out_path.stem:
        # `key` not present — fall back to a sibling next to it
        new_stem = replacement
    return out_path.with_name(new_stem + out_path.suffix)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One η measurement at a single time from one (sat, signal) source.

    y is preprocessed so the KF sees `y = η + noise` directly:
        η = ANTENNA_MSL_M − tide_model.predict(t) − RH_obs
    """
    t_utc:  pd.Timestamp     # tz-aware UTC
    y:      float            # η in meters
    sigma:  float            # 1σ on y, meters
    sat:    int
    signal: str


@dataclass
class TideKalmanConfig:
    """Tuning knobs for the linear-KF tide-residual estimator.

    Two process models are supported:

    - `model='cv'` — constant-velocity (2-state `[η, dη/dt]`). Good for
      per-arc observations spaced ~30 min apart, where velocity is
      physically meaningful (storm surge rate, etc.).
    - `model='rw'` — position-only random walk (1-state `[η]`). The
      right choice for windowed observations at sub-arc cadence (seconds
      to minutes), where any inferred "velocity" is dominated by obs
      noise and will runaway. `run_batch_windowed` defaults to 'rw'.
    """
    model:          str   = 'cv'   # 'cv' (constant velocity) or 'rw' (random walk)
    sigma_a:        float = 1e-6   # accel noise (m/s²) for 'cv' model
    sigma_p:        float = 5e-3   # position random-walk noise (m/√s) for 'rw'
                                   # — sized to track ~30 cm/hr tide rate while
                                   # smoothing arc-to-arc obs scatter
    gate_threshold: float = 9.0    # Mahalanobis² cutoff (~3σ for 1-D innovation)
    sigma_inflation_m: float = 0.15  # added in quadrature to per-obs sigma.
                                     # Per-arc Lomb-Scargle σ captures within-
                                     # arc precision (~1-3 cm) but misses
                                     # arc-to-arc multipath / footprint
                                     # scatter (~10-30 cm). Windowed obs already
                                     # have honest σ from the shorter spectral
                                     # window so this matters less for 'rw'.
    init_eta:       float | None = None    # if None, init to first obs y
    init_eta_sigma: float = 1.0    # initial 1σ on η (m)
    init_vel:       float = 0.0    # initial dη/dt (m/s) — used only for 'cv'
    init_vel_sigma: float = 0.05   # initial 1σ on dη/dt — used only for 'cv'
    antenna_msl_m:  float = c.ANTENNA_MSL_M


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

class TideKalman:
    """Linear Kalman filter on η = water level − predicted tide.

    Two state-space models are supported (see `TideKalmanConfig`):
    - 'cv': 2-state [η, dη/dt]ᵀ with continuous white-noise acceleration
    - 'rw': 1-state [η] with random-walk dynamics

    Sequential 1-D updates → multi-satellite fusion at the same timestamp
    is just multiple `update()` calls.
    """

    def __init__(self, tide_model, config: TideKalmanConfig | None = None):
        self.tide = tide_model
        self.cfg = config or TideKalmanConfig()
        if self.cfg.model not in ('cv', 'rw'):
            raise ValueError(f"model must be 'cv' or 'rw', got {self.cfg.model!r}")
        self.dim = 2 if self.cfg.model == 'cv' else 1
        self.H = (np.array([[1.0, 0.0]]) if self.dim == 2
                  else np.array([[1.0]]))
        self._initialized = False
        self.t: pd.Timestamp | None = None
        self.x = np.zeros(self.dim)
        self.P = np.eye(self.dim)

    # ---- core ops ----

    def _initialize(self, obs: Observation) -> None:
        # Init mean from first obs (or config override); always use the
        # configured init_eta_sigma as initial uncertainty (using obs.sigma
        # would make the filter overconfident and lock onto an early outlier).
        eta0 = self.cfg.init_eta if self.cfg.init_eta is not None else obs.y
        if self.dim == 2:
            self.x = np.array([eta0, self.cfg.init_vel])
            self.P = np.diag([self.cfg.init_eta_sigma**2,
                              self.cfg.init_vel_sigma**2])
        else:
            self.x = np.array([eta0])
            self.P = np.array([[self.cfg.init_eta_sigma**2]])
        self.t = obs.t_utc
        self._initialized = True

    def predict(self, t_new: pd.Timestamp) -> None:
        if not self._initialized:
            raise RuntimeError('Filter not initialized; call update() first')
        dt = (t_new - self.t).total_seconds()
        if dt < 0:
            raise ValueError(f'Cannot predict backward (dt={dt:.1f}s)')
        if dt == 0:
            return
        if self.dim == 2:
            # Constant-velocity model with CWN acceleration
            F = np.array([[1.0, dt], [0.0, 1.0]])
            q = self.cfg.sigma_a ** 2
            Q = q * np.array([
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt          ],
            ])
        else:
            # Position-only random walk
            F = np.array([[1.0]])
            Q = np.array([[self.cfg.sigma_p**2 * dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t_new

    def update(self, obs: Observation) -> dict:
        """Incorporate one observation. Returns innov/mahal²/accepted dict."""
        if not self._initialized:
            self._initialize(obs)
            return {'innov': 0.0, 'mahal2': 0.0, 'accepted': True}

        if obs.t_utc != self.t:
            self.predict(obs.t_utc)

        # Inflate per-obs sigma in quadrature with the configured floor.
        # Per-arc Lomb-Scargle sigma is too tight for the KF — see
        # TideKalmanConfig docstring.
        R = obs.sigma ** 2 + self.cfg.sigma_inflation_m ** 2
        innov = obs.y - float((self.H @ self.x)[0])
        S = float((self.H @ self.P @ self.H.T)[0, 0]) + R
        mahal2 = innov * innov / S

        if mahal2 > self.cfg.gate_threshold:
            return {'innov': innov, 'mahal2': mahal2, 'accepted': False}

        K = (self.P @ self.H.T).flatten() / S
        self.x = self.x + K * innov
        # Joseph form for numerical stability
        I_KH = np.eye(self.dim) - np.outer(K, self.H[0])
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R
        return {'innov': innov, 'mahal2': mahal2, 'accepted': True}

    def state(self) -> tuple[float, float, float, float]:
        """Return (η, σ_η, dη/dt, σ_dη/dt). Velocity components are 0 for
        the position-only ('rw') model."""
        eta = float(self.x[0])
        eta_sigma = float(np.sqrt(self.P[0, 0]))
        if self.dim == 2:
            vel = float(self.x[1])
            vel_sigma = float(np.sqrt(self.P[1, 1]))
        else:
            vel = 0.0
            vel_sigma = 0.0
        return eta, eta_sigma, vel, vel_sigma


# ---------------------------------------------------------------------------
# Per-arc preprocessing
# ---------------------------------------------------------------------------

def arc_row_to_obs(row: pd.Series, antenna_msl: float, tide_at_t: float
                   ) -> list[Observation]:
    """One per-arc parquet row + precomputed tide -> list of Observation
    (one per applicable, finite signal). Use for fine-grained per-band
    Kalman work (streaming, IMM, etc.)."""
    out = []
    t = row['t_mid_utc']
    sat = int(row['sat'])
    for sig in c.ENABLED_SIGNALS:
        rh_col, sig_col = f'RH_{sig.name}', f'sigma_{sig.name}'
        if rh_col not in row.index:
            continue
        rh = row[rh_col]
        sigma = row[sig_col]
        if not (np.isfinite(rh) and np.isfinite(sigma) and sigma > 0):
            continue
        out.append(Observation(
            t_utc=t,
            y=antenna_msl - tide_at_t - float(rh),
            sigma=float(sigma),
            sat=sat,
            signal=sig.name,
        ))
    return out


def arc_row_to_consensus(row: pd.Series, antenna_msl: float, tide_at_t: float,
                          max_spread_m: float = 0.3) -> Observation | None:
    """Collapse all bands for one arc into a single consensus Observation.

    Robust to multimodal periodograms: drops bands more than `max_spread_m/2`
    from the cross-band median, then takes the inverse-variance weighted mean
    of the survivors. Returns None if fewer than 2 bands survive.

    This is the default observation builder for `run_batch` because
    GNSS-IR periodograms sometimes pick up secondary reflectors / harmonics
    that fool one band but not others; cross-band consensus filters them
    out before they reach the KF.
    """
    rh_vals, sig_vals = [], []
    for sig in c.ENABLED_SIGNALS:
        rh_col, sig_col = f'RH_{sig.name}', f'sigma_{sig.name}'
        if rh_col not in row.index:
            continue
        rh, sigma = row[rh_col], row[sig_col]
        if not (np.isfinite(rh) and np.isfinite(sigma) and sigma > 0):
            continue
        rh_vals.append(float(rh))
        sig_vals.append(float(sigma))

    if len(rh_vals) < 2:
        return None
    rh_arr = np.asarray(rh_vals)
    sig_arr = np.asarray(sig_vals)

    # Outlier drop: keep bands within max_spread_m/2 of the cross-band median
    med = np.median(rh_arr)
    keep = np.abs(rh_arr - med) <= (max_spread_m / 2)
    if keep.sum() < 2:
        return None
    rh_arr, sig_arr = rh_arr[keep], sig_arr[keep]

    # Inverse-variance weighted mean across surviving bands
    w = 1.0 / sig_arr**2
    rh_mean = float((rh_arr * w).sum() / w.sum())
    sigma_mean = float(1.0 / np.sqrt(w.sum()))

    return Observation(
        t_utc=row['t_mid_utc'],
        y=antenna_msl - tide_at_t - rh_mean,
        sigma=sigma_mean,
        sat=int(row['sat']),
        signal='consensus',
    )


def windowed_row_to_obs(row: pd.Series, antenna_msl: float, tide_at_t: float
                         ) -> Observation:
    """One row of long-format windowed obs (from `snr.process_arcs_windowed`)
    -> one Observation."""
    return Observation(
        t_utc=row['t_center_utc'],
        y=antenna_msl - tide_at_t - float(row['rh']),
        sigma=float(row['sigma']),
        sat=int(row['sat']),
        signal=str(row['signal']),
    )


def bin_obs_by_time(obs_df: pd.DataFrame,
                    bin_sec: float = 30.0,
                    max_spread_m: float = 0.5,
                    min_n: int = 1) -> pd.DataFrame:
    """Aggregate long-form windowed obs into fixed time bins with
    multi-satellite consensus.

    For each bin:
      1. Compute the within-bin median RH across all (sat, signal) obs.
      2. Drop obs more than `max_spread_m/2` from that median.
      3. Take the inverse-variance weighted mean of the survivors.
      4. Combined σ is the standard error of the weighted mean
         (`1/√Σ(1/σᵢ²)`), so binning N obs tightens σ by ~1/√N.

    This pre-filter feeds smoother input to the KF without losing rogue-wave
    sensitivity: real events are spatially coherent (all sats in the bin
    agree, large amplitude survives the median), whereas single-sat noise
    spikes get dropped by the outlier filter.

    Parameters
    ----------
    obs_df       : long-form output of `snr.process_arcs_windowed`.
    bin_sec      : bin width in seconds (default 30 s).
    max_spread_m : within-bin outlier rejection threshold around median.
    min_n        : minimum surviving obs to emit a bin row (1 keeps singletons).

    Returns long-form DataFrame ready for `run_batch_windowed`. Columns:
        t_center_utc, rh, sigma, sat, signal,
        n_obs, n_sats, n_dropped, rh_spread_m
    """
    if obs_df.empty:
        return pd.DataFrame()

    obs = obs_df.copy()
    bin_freq = f'{int(bin_sec)}s'
    bin_td = pd.Timedelta(seconds=bin_sec)
    obs['_bin_start'] = obs['t_center_utc'].dt.floor(bin_freq)

    rows = []
    for bin_start, group in obs.groupby('_bin_start', sort=True):
        rhs = group['rh'].to_numpy()
        sigmas = group['sigma'].to_numpy()
        sats = group['sat'].to_numpy()

        # Outlier reject against within-bin median
        med = float(np.median(rhs))
        keep = np.abs(rhs - med) <= (max_spread_m / 2.0)
        n_kept = int(keep.sum())
        if n_kept < min_n:
            continue
        rhs_k = rhs[keep]
        sigmas_k = sigmas[keep]
        sats_k = sats[keep]

        # Inverse-variance weighted mean (standard error of the mean as sigma)
        w = 1.0 / sigmas_k**2
        rh_mean = float((rhs_k * w).sum() / w.sum())
        sigma_mean = float(1.0 / np.sqrt(w.sum()))

        rows.append({
            't_center_utc': bin_start + bin_td / 2,
            'rh':           rh_mean,
            'sigma':        sigma_mean,
            'sat':          -1,            # sentinel: aggregated, not a single sat
            'signal':       'binned',
            'n_obs':        n_kept,
            'n_sats':       int(pd.Series(sats_k).nunique()),
            'n_dropped':    int((~keep).sum()),
            'rh_spread_m':  float(rhs_k.max() - rhs_k.min()) if n_kept > 1 else 0.0,
        })

    return pd.DataFrame(rows).sort_values('t_center_utc').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch driver — dispatches on DataFrame format
# ---------------------------------------------------------------------------

def run_batch(obs_df: pd.DataFrame, tide_model, **kwargs
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay observations chronologically through TideKalman.

    Auto-dispatches by DataFrame format:
    - `t_center_utc` column present  -> windowed (long format)
    - `t_mid_utc` column present     -> per-arc (wide format)

    Returns (state_df, gated_df). See `run_batch_windowed` and
    `run_batch_per_arc` for kwarg details.
    """
    if 't_center_utc' in obs_df.columns:
        return run_batch_windowed(obs_df, tide_model, **kwargs)
    if 't_mid_utc' in obs_df.columns:
        return run_batch_per_arc(obs_df, tide_model, **kwargs)
    raise ValueError(
        'obs_df must have either `t_center_utc` (windowed) or `t_mid_utc` '
        '(per-arc) column'
    )


def run_batch_windowed(obs_df: pd.DataFrame, tide_model, *,
                        config: TideKalmanConfig | None = None,
                        save_to: Path | str | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drive TideKalman with long-format windowed observations
    (output of `snr.process_arcs_windowed`).

    Per-observation state row — sat and signal preserved in the output so
    you can trace which observation drove each update.

    Defaults to position-only random-walk state (`model='rw'`). At sub-arc
    cadence (seconds–minutes) the constant-velocity model overfits: alternating
    high/low obs from different sats pump the velocity estimate into runaway.
    Pass an explicit `config` with `model='cv'` to override.
    """
    if config is None:
        config = TideKalmanConfig(model='rw')
    kf = TideKalman(tide_model, config)
    cfg = config

    obs = obs_df.sort_values('t_center_utc').reset_index(drop=True)

    # Vectorized tide prediction for all window centers
    tide_at_obs = tide_model.predict(obs['t_center_utc'].tolist())

    rows, gated = [], []
    for i, r in obs.iterrows():
        tide_at_t = float(tide_at_obs[i])
        ob = windowed_row_to_obs(r, cfg.antenna_msl_m, tide_at_t)

        res = kf.update(ob)
        if not res['accepted']:
            gated.append({
                't_utc':  ob.t_utc,
                'sat':    ob.sat,
                'signal': ob.signal,
                'y':      ob.y,
                'sigma':  ob.sigma,
                'innov':  res['innov'],
                'mahal2': res['mahal2'],
            })
            continue

        eta, eta_s, vel, vel_s = kf.state()
        rows.append({
            't_utc':         ob.t_utc,
            'sat':           ob.sat,
            'signal':        ob.signal,
            'eta_m':         eta,
            'eta_sigma_m':   eta_s,
            'vel_m_s':       vel,
            'vel_sigma_m_s': vel_s,
            'tide_m':        tide_at_t,
            'water_level_m': tide_at_t + eta,
            'innov':         res['innov'],
            'mahal2':        res['mahal2'],
        })

    state = pd.DataFrame(rows)
    gated_df = pd.DataFrame(gated)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        state.to_parquet(save_to, compression='snappy', index=False)
        if not gated_df.empty:
            gated_df.to_parquet(_sibling(save_to, 'state', 'gated'),
                                compression='snappy', index=False)
    return state, gated_df


def run_batch_per_arc(arcs_df: pd.DataFrame, tide_model, *,
                       config: TideKalmanConfig | None = None,
                       consensus: bool = True,
                       consensus_max_spread_m: float = 0.3,
                       save_to: Path | str | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drive TideKalman with per-arc wide-format observations
    (output of `snr.process_arcs` / `pipeline.load_results`).

    Parameters
    ----------
    consensus : bool, default True
        If True, collapse each arc's bands into one consensus observation
        (recommended — robust to multimodal periodogram peaks). If False,
        feed every band of every arc as an independent observation.
    consensus_max_spread_m : float, default 0.3
        Max cross-band RH spread allowed before bands are filtered against
        the cross-band median. Only used when `consensus=True`.

    Returns (state_df, gated_df). One state row per arc that produced at
    least one observation.
    """
    cfg = config or TideKalmanConfig()
    kf = TideKalman(tide_model, cfg)

    arcs = arcs_df.sort_values('t_mid_utc').reset_index(drop=True)

    # Vectorized tide prediction for all arc midpoints (one model call total)
    tide_at_arcs = tide_model.predict(arcs['t_mid_utc'].tolist())

    rows, gated = [], []
    for i, arc in arcs.iterrows():
        tide_at_t = float(tide_at_arcs[i])
        if consensus:
            single = arc_row_to_consensus(arc, cfg.antenna_msl_m, tide_at_t,
                                           max_spread_m=consensus_max_spread_m)
            obs_list = [single] if single is not None else []
        else:
            obs_list = arc_row_to_obs(arc, cfg.antenna_msl_m, tide_at_t)
            # Apply tightest obs first so cov shrinks before noisier obs hit
            obs_list.sort(key=lambda o: o.sigma)
        if not obs_list:
            continue

        n_assim = n_gated = 0
        innov_max = 0.0
        for obs in obs_list:
            res = kf.update(obs)
            if res['accepted']:
                n_assim += 1
                innov_max = max(innov_max, abs(res['innov']))
            else:
                n_gated += 1
                gated.append({
                    't_utc':   obs.t_utc,
                    'sat':     obs.sat,
                    'signal':  obs.signal,
                    'y':       obs.y,
                    'sigma':   obs.sigma,
                    'innov':   res['innov'],
                    'mahal2':  res['mahal2'],
                })

        eta, eta_s, vel, vel_s = kf.state()
        rows.append({
            't_utc':         arc['t_mid_utc'],
            'eta_m':         eta,
            'eta_sigma_m':   eta_s,
            'vel_m_s':       vel,
            'vel_sigma_m_s': vel_s,
            'tide_m':        tide_at_t,
            'water_level_m': tide_at_t + eta,
            'n_obs_assim':   n_assim,
            'n_obs_gated':   n_gated,
            'innov_max':     innov_max,
        })

    state = pd.DataFrame(rows)
    gated_df = pd.DataFrame(gated)

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        state.to_parquet(save_to, compression='snappy', index=False)
        if not gated_df.empty:
            gated_df.to_parquet(_sibling(save_to, 'state', 'gated'),
                                compression='snappy', index=False)
    return state, gated_df


# ---------------------------------------------------------------------------
# Streaming driver
# ---------------------------------------------------------------------------

def run_streaming(obs_iter: Iterable[Observation], tide_model, *,
                  config: TideKalmanConfig | None = None
                  ) -> Iterator[dict]:
    """Yield filter state after each observation. Same KF, async-friendly."""
    kf = TideKalman(tide_model, config or TideKalmanConfig())
    for obs in obs_iter:
        res = kf.update(obs)
        eta, eta_s, vel, vel_s = kf.state()
        yield {
            't_utc':         obs.t_utc,
            'sat':           obs.sat,
            'signal':        obs.signal,
            'eta_m':         eta,
            'eta_sigma_m':   eta_s,
            'vel_m_s':       vel,
            'vel_sigma_m_s': vel_s,
            'innov':         res['innov'],
            'mahal2':        res['mahal2'],
            'accepted':      res['accepted'],
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time as _time
    import snr
    from tide import GreenlandTideModel

    YEAR, DOYS = 2026, [1, 2, 3]

    # Default mode: windowed observations from snr.process_arcs_windowed
    # (per-arc analysis is also supported via run_batch_per_arc, but the
    # windowed path gives the seconds-to-minutes time resolution needed
    # for rogue-wave / transient detection.)
    print(f'Building windowed observations for doys {DOYS} '
          f'(window={c.WINDOW_SEC}s, stride={c.STRIDE_SEC}s)...')
    frames = []
    t0 = _time.perf_counter()
    for doy in DOYS:
        snr_df = snr.load_snr(YEAR, doy)
        obs_day = snr.process_arcs_windowed(snr_df)
        if not obs_day.empty:
            frames.append(obs_day)
        print(f'  doy {doy:03d}: {len(obs_day):>5d} windowed obs')
    obs_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f'Total: {len(obs_df):,} obs in {_time.perf_counter()-t0:.1f}s')
    if obs_df.empty:
        raise SystemExit('No windowed observations produced.')

    # Save the raw windowed obs (pre-binning) so plots can show the scatter
    obs_path = c.RESULTS_DIR / f'{YEAR}' / 'windowed' / f'{DOYS[0]:03d}-{DOYS[-1]:03d}_obs.parquet'
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    obs_df.to_parquet(obs_path, compression='snappy', index=False)
    print(f'Windowed obs saved: {obs_path.relative_to(c.PROJECT_DIR)}')

    # Bin by time + multi-sat consensus before feeding the KF.
    # 30s bins are mostly singletons (no consensus benefit); 120s gives a
    # median ~3 obs/bin and meaningful √N noise reduction while still
    # resolving events ~30s+ long.
    BIN_SEC = 120.0
    print(f'\nBinning obs (bin={int(BIN_SEC)}s, multi-sat consensus)...')
    obs_binned = bin_obs_by_time(obs_df, bin_sec=BIN_SEC)
    print(f'Binned: {len(obs_df):,} obs -> {len(obs_binned):,} bins '
          f'(σ {obs_df.sigma.median()*100:.1f} cm -> '
          f'{obs_binned.sigma.median()*100:.1f} cm)')
    print(f'Median obs per bin: {obs_binned.n_obs.median():.0f}  '
          f'(sats: {obs_binned.n_sats.median():.0f})')
    binned_path = c.RESULTS_DIR / f'{YEAR}' / 'windowed' / f'{DOYS[0]:03d}-{DOYS[-1]:03d}_binned.parquet'
    obs_binned.to_parquet(binned_path, compression='snappy', index=False)

    tm = GreenlandTideModel(c.LAT, c.LON)
    print(f'Tide model loaded for ({c.LAT}, {c.LON}); '
          f'nearest cell {tm.cell_distance_km:.2f} km away')

    # Default model for windowed mode is random-walk position-only;
    # `run_batch_windowed` chooses this automatically when config=None.
    save = c.RESULTS_DIR / f'{YEAR}' / 'state' / f'{DOYS[0]:03d}-{DOYS[-1]:03d}_state.parquet'
    t0 = _time.perf_counter()
    state, gated = run_batch(obs_binned, tm, save_to=save)
    print(f'KF run: {len(state)} state updates in {_time.perf_counter()-t0:.1f}s')

    print(f'\n  η range          : {state.eta_m.min():+.3f} -> {state.eta_m.max():+.3f}  m')
    print(f'  median σ_η       : {state.eta_sigma_m.median()*100:5.2f} cm')
    print(f'  water-level range: {state.water_level_m.min():+.3f} -> {state.water_level_m.max():+.3f}  m')
    print(f'  vel  range       : {state.vel_m_s.min()*3600:+.2f} -> {state.vel_m_s.max()*3600:+.2f}  m/hr')
    print(f'  obs assimilated  : {len(state)}')
    print(f'  obs gated        : {len(gated)}  ({100*len(gated)/max(1,len(state)+len(gated)):.1f}%)')
    print(f'\nState saved: {save.relative_to(c.PROJECT_DIR)}')
    if not gated.empty:
        print(f'\nGated observations by signal:')
        print(gated.groupby('signal').size().to_string())
