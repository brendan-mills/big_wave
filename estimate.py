"""Tide-residual Kalman filter for sea-level estimation.

Layer 3 of the pipeline: consumes the per-bin observation parquets produced
by `pipeline.py` (after `bin_obs_by_time`) and the deterministic tide prior
from `tide.py`, produces a continuous state estimate `η(t) ± σ(t)` plus a
log of any observations rejected by the chi-square gate.

The filter is a **1-state random-walk Kalman filter** on
    η = observed water level − predicted tide
fed by binned windowed observations at second-to-minute cadence. At those
timescales any inferred velocity is dominated by noise and runs away, so we
do not track velocity (a previous constant-velocity variant was removed).
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
    """Tuning knobs for the 1-D random-walk tide-residual estimator."""
    sigma_p:        float = 5e-3   # position random-walk noise (m/√s) — sized to
                                   # track ~30 cm/hr tide rate while smoothing
                                   # bin-to-bin observation scatter
    gate_threshold: float = 9.0    # Mahalanobis² cutoff (~3σ on 1-D innovation)
    sigma_inflation_m: float = 0.50  # added in quadrature to per-obs sigma.
                                     # The formal per-bin σ (~30 cm) undertells
                                     # actual obs noise — consecutive bins
                                     # routinely disagree by 50-80 cm due to
                                     # correlated multipath that's not in the
                                     # Lomb-Scargle σ. Inflating to 50 cm tells
                                     # the KF to trust each obs less, which
                                     # both smooths the state and grows the σ
                                     # band to honestly reflect per-instant
                                     # water-level uncertainty.
    init_eta:       float | None = None    # if None, init to first obs y
    init_eta_sigma: float = 1.0    # initial 1σ on η (m) — wide so filter
                                   # doesn't lock onto an early outlier
    antenna_msl_m:  float = c.ANTENNA_MSL_M


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

class TideKalman:
    """1-D random-walk Kalman filter on η = water level − predicted tide.

    Sequential update model — multi-satellite fusion at the same timestamp
    is just multiple `update()` calls in a row.
    """

    H = np.array([[1.0]])    # observation matrix: z = η

    def __init__(self, tide_model, config: TideKalmanConfig | None = None):
        self.tide = tide_model
        self.cfg = config or TideKalmanConfig()
        self._initialized = False
        self.t: pd.Timestamp | None = None
        self.x = np.zeros(1)
        self.P = np.eye(1)

    # ---- core ops ----

    def _initialize(self, obs: Observation) -> None:
        # Init mean from first obs (or config override); always use the
        # configured init_eta_sigma as initial uncertainty.
        eta0 = self.cfg.init_eta if self.cfg.init_eta is not None else obs.y
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
        # Position-only random walk: state unchanged, variance grows as σ²·dt
        self.P = self.P + np.array([[self.cfg.sigma_p**2 * dt]])
        self.t = t_new

    def update(self, obs: Observation) -> dict:
        """Incorporate one observation. Returns innov/mahal²/accepted dict."""
        if not self._initialized:
            self._initialize(obs)
            return {'innov': 0.0, 'mahal2': 0.0, 'accepted': True}

        if obs.t_utc != self.t:
            self.predict(obs.t_utc)

        # Per-obs sigma + project-wide inflation in quadrature.
        R = obs.sigma ** 2 + self.cfg.sigma_inflation_m ** 2
        innov = obs.y - float((self.H @ self.x)[0])
        S = float((self.H @ self.P @ self.H.T)[0, 0]) + R
        mahal2 = innov * innov / S

        if mahal2 > self.cfg.gate_threshold:
            return {'innov': innov, 'mahal2': mahal2, 'accepted': False}

        K = (self.P @ self.H.T).flatten() / S
        self.x = self.x + K * innov
        I_KH = np.eye(1) - np.outer(K, self.H[0])
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R
        return {'innov': innov, 'mahal2': mahal2, 'accepted': True}

    def state(self) -> tuple[float, float]:
        """Return (η, σ_η)."""
        return float(self.x[0]), float(np.sqrt(self.P[0, 0]))


# ---------------------------------------------------------------------------
# Per-observation preprocessing
# ---------------------------------------------------------------------------

def windowed_row_to_obs(row: pd.Series, antenna_msl: float, tide_at_t: float
                         ) -> Observation:
    """One row of long-format windowed/binned obs -> one Observation."""
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

    Real events are spatially coherent (all sats in the bin agree, large
    amplitude survives the median), whereas single-sat noise spikes get
    dropped by the outlier filter.

    Returns long-form DataFrame ready for `run_batch`. Columns:
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

        med = float(np.median(rhs))
        keep = np.abs(rhs - med) <= (max_spread_m / 2.0)
        n_kept = int(keep.sum())
        if n_kept < min_n:
            continue
        rhs_k = rhs[keep]
        sigmas_k = sigmas[keep]
        sats_k = sats[keep]

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
# Batch driver
# ---------------------------------------------------------------------------

def run_batch(obs_df: pd.DataFrame, tide_model, *,
              config: TideKalmanConfig | None = None,
              save_to: Path | str | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay long-format windowed/binned observations chronologically
    through `TideKalman`.

    `obs_df` must have columns: `t_center_utc, rh, sigma, sat, signal`.
    Typically the output of `bin_obs_by_time(snr.process_arcs_windowed(...))`.

    Returns `(state_df, gated_df)`. One state row per accepted observation
    with sat and signal preserved. `save_to` writes the state parquet plus
    a sibling `gated.parquet` if any obs were rejected.
    """
    cfg = config or TideKalmanConfig()
    kf = TideKalman(tide_model, cfg)

    obs = obs_df.sort_values('t_center_utc').reset_index(drop=True)
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

        eta, eta_s = kf.state()
        rows.append({
            't_utc':         ob.t_utc,
            'sat':           ob.sat,
            'signal':        ob.signal,
            'eta_m':         eta,
            'eta_sigma_m':   eta_s,
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
        eta, eta_s = kf.state()
        yield {
            't_utc':         obs.t_utc,
            'sat':           obs.sat,
            'signal':        obs.signal,
            'eta_m':         eta,
            'eta_sigma_m':   eta_s,
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

    obs_path = c.RESULTS_DIR / f'{YEAR}' / 'windowed' / f'{DOYS[0]:03d}-{DOYS[-1]:03d}_obs.parquet'
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    obs_df.to_parquet(obs_path, compression='snappy', index=False)
    print(f'Windowed obs saved: {obs_path.relative_to(c.PROJECT_DIR)}')

    BIN_SEC = 120.0
    print(f'\nBinning obs (bin={int(BIN_SEC)}s, multi-sat consensus)...')
    obs_binned = bin_obs_by_time(obs_df, bin_sec=BIN_SEC)
    print(f'Binned: {len(obs_df):,} obs -> {len(obs_binned):,} bins '
          f'(σ {obs_df.sigma.median()*100:.1f} cm -> '
          f'{obs_binned.sigma.median()*100:.1f} cm)')

    tm = GreenlandTideModel(c.LAT, c.LON)
    print(f'Tide model loaded; nearest cell {tm.cell_distance_km:.2f} km away')

    save = c.RESULTS_DIR / f'{YEAR}' / 'state' / f'{DOYS[0]:03d}-{DOYS[-1]:03d}_state.parquet'
    t0 = _time.perf_counter()
    state, gated = run_batch(obs_binned, tm, save_to=save)
    print(f'KF run: {len(state)} state updates in {_time.perf_counter()-t0:.1f}s')

    print(f'\n  η range          : {state.eta_m.min():+.3f} -> {state.eta_m.max():+.3f}  m')
    print(f'  median σ_η       : {state.eta_sigma_m.median()*100:5.2f} cm')
    print(f'  water-level range: {state.water_level_m.min():+.3f} -> {state.water_level_m.max():+.3f}  m')
    print(f'  obs assimilated  : {len(state)}')
    print(f'  obs gated        : {len(gated)}  ({100*len(gated)/max(1,len(state)+len(gated)):.1f}%)')
