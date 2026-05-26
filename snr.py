"""SNR file processing for SSiSLS.

Wraps gnssrefl for RINEX -> SNR conversion and SNR file reading. Provides
arc segmentation and multi-constellation Lomb-Scargle RH estimation.

The wide DataFrame returned by `process_arcs` is the canonical per-day output
consumed by the rest of the pipeline (plots, Kalman filter, etc.).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

import config as c

# gnssrefl reads these at import time, so set them before the imports below
os.environ.setdefault('REFL_CODE', str(c.REFL_CODE))
os.environ.setdefault('ORBITS',    str(c.ORBITS_DIR))
os.environ.setdefault('EXE',       str(c.EXE_DIR))

from gnssrefl.gps import snr_name                          # noqa: E402
from gnssrefl.read_snr_files import read_snr               # noqa: E402
from gnssrefl.rinex2snr_cl import rinex2snr                # noqa: E402


# ---------------------------------------------------------------------------
# Station coordinates from a RINEX header (one-shot helper)
# ---------------------------------------------------------------------------

def coords_from_rinex(rinex_path: Path | str) -> dict:
    """Extract approximate lat/lon/height (WGS-84) from a RINEX 2 obs header.

    Accepts `.d.Z`, `.d.gz`, `.d`, `.o.gz`, `.o`. Hatanaka files are expanded
    via `CRX2RNX` under `config.EXE_DIR`.
    """
    src = Path(rinex_path)
    crx2rnx = c.EXE_DIR / 'CRX2RNX'

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        local = tmp / src.name
        shutil.copy(src, local)

        if local.suffix == '.Z':
            subprocess.run(['uncompress', '-f', str(local)], check=True)
            local = local.with_suffix('')
        elif local.suffix == '.gz':
            subprocess.run(['gunzip', '-f', str(local)], check=True)
            local = local.with_suffix('')

        if local.suffix.endswith('d'):
            subprocess.run([str(crx2rnx), str(local)], check=True)
            local = local.with_suffix(local.suffix[:-1] + 'o')

        with open(local) as f:
            X = Y = Z = None
            for line in f:
                if 'APPROX POSITION XYZ' in line:
                    X, Y, Z = (float(v) for v in line.split()[:3])
                    break
                if 'END OF HEADER' in line:
                    raise ValueError(f'APPROX POSITION XYZ not in header of {src}')

    a, fl = 6378137.0, 1 / 298.257223563
    e2 = fl * (2 - fl); b = a * (1 - fl); ep2 = (a*a - b*b) / (b*b)
    p = math.hypot(X, Y)
    th = math.atan2(Z * a, p * b)
    lon = math.atan2(Y, X)
    lat = math.atan2(Z + ep2 * b * math.sin(th)**3,
                     p - e2 * a * math.cos(th)**3)
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    h = p / math.cos(lat) - N

    return {'lat': math.degrees(lat), 'lon': math.degrees(lon), 'h': h,
            'X': X, 'Y': Y, 'Z': Z}


# ---------------------------------------------------------------------------
# SNR file location, creation, loading
# ---------------------------------------------------------------------------

def _doy_to_month_day(year: int, doy: int) -> tuple[int, int]:
    import datetime as dt
    d = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1)
    return d.month, d.day


def snr_path(year: int, doy: int) -> Path:
    """Resolve the snr66 path (gzipped or plain) for this station/year/doy."""
    month, day = _doy_to_month_day(year, doy)
    fname = snr_name(c.STATION, year, month, day, str(c.SNR_TYPE))
    base = c.REFL_CODE / f'{year}' / 'snr' / c.STATION / fname
    if base.exists():
        return base
    gz = base.with_suffix(base.suffix + '.gz')
    if gz.exists():
        return gz
    return base  # may not exist yet; caller decides what to do


def ensure_snr(year: int, doy: int, force: bool = False) -> Path:
    """Make sure the snr66 file exists for this day; produce it from local
    RINEX via `rinex2snr` if not. Returns the resolved path.
    """
    p = snr_path(year, doy)
    if p.exists() and not force:
        return p

    yy = str(year)[-2:]
    candidates = list(c.RINEX_DIR.glob(f'{c.STATION}{doy:03d}0.{yy}d.*'))
    if not candidates:
        raise FileNotFoundError(
            f'No local RINEX for {c.STATION} {year} doy {doy} in {c.RINEX_DIR}')
    src = candidates[0]

    # gnssrefl reads the RINEX from the current working directory
    dst = Path.cwd() / src.name
    if not dst.exists():
        shutil.copy(src, dst)

    rinex2snr(
        station=c.STATION, year=year, doy=doy,
        nolook=c.NOLOOK, orb=c.ORB, overwrite=force,
    )

    p = snr_path(year, doy)
    if not p.exists():
        raise RuntimeError(f'rinex2snr completed but no snr66 file at {p}')
    return p


def load_snr(year: int, doy: int) -> pd.DataFrame:
    """Read the snr66 file for this day into a DataFrame with named columns.

    Adds derived columns `year`, `doy`, and `time_utc` (absolute UTC timestamp
    per row, datetime64[ns]) so downstream code can build rolling windows or
    concatenate multiple days without reconstructing time from sec-of-day.
    """
    p = ensure_snr(year, doy)
    ok, arr, nrow, ncol = read_snr(str(p))
    if not ok:
        raise RuntimeError(f'gnssrefl.read_snr failed on {p}')
    if ncol != len(c.SNR_COLUMNS):
        raise ValueError(
            f'snr66 has {ncol} columns; expected {len(c.SNR_COLUMNS)} '
            f'({c.SNR_COLUMNS}). File: {p}')
    df = pd.DataFrame(arr, columns=c.SNR_COLUMNS)
    df['sat'] = df['sat'].astype(int)
    df['year'] = int(year)
    df['doy']  = int(doy)
    epoch = pd.Timestamp(f'{year}-01-01', tz='UTC') + pd.Timedelta(days=doy - 1)
    df['time_utc'] = epoch + pd.to_timedelta(df['sec'], unit='s')
    return df


# ---------------------------------------------------------------------------
# Arc segmentation
# ---------------------------------------------------------------------------

def segment_arcs(snr_df: pd.DataFrame) -> pd.DataFrame:
    """Filter SNR data to enabled-signal PRNs within the az/el window, mark
    rise/set, split passes on time gaps, drop too-short arcs. Returns the
    filtered DataFrame with added `dir`, `pass_id`, `arc_id` columns.
    Empty DataFrame if no arcs survive.
    """
    enabled_prns = {p for s in c.ENABLED_SIGNALS
                    for p in range(s.prn_lo, s.prn_hi + 1)}

    sub = snr_df[
        snr_df.sat.isin(enabled_prns)
        & snr_df.azim.between(c.AZ_MIN, c.AZ_MAX)
        & snr_df.elev.between(c.EL_MIN, c.EL_MAX)
    ].copy()
    if sub.empty:
        return sub.assign(dir='', pass_id=0, arc_id=-1)

    sub['dir'] = np.where(sub.edot > 0, 'rise', 'set')
    sub = sub.sort_values(['sat', 'dir', 'sec']).reset_index(drop=True)

    # pass_id increments within each (sat, dir) on every time gap > GAP_SEC
    sub['pass_id'] = (sub.groupby(['sat', 'dir'])['sec']
                         .transform(lambda s: s.diff().gt(c.GAP_SEC).cumsum())
                         .fillna(0).astype(int))
    sub['arc_id'] = sub.groupby(['sat', 'dir', 'pass_id']).ngroup()

    arc_sizes = sub.groupby('arc_id').size()
    keep = arc_sizes[arc_sizes >= c.MIN_ARC_PTS].index
    return sub[sub.arc_id.isin(keep)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-arc, per-signal RH estimation
# ---------------------------------------------------------------------------

def _peak_refine(pgram: np.ndarray, heights: np.ndarray, i: int
                 ) -> tuple[float, float]:
    """3-point parabolic peak refinement. Returns (h_refined, sigma_gauss).
    `sigma_gauss` is the local Gaussian-equivalent peak width (m); NaN if the
    triple isn't a proper maximum (edge of grid or concave-up).
    """
    if i <= 0 or i >= len(pgram) - 1:
        return float(heights[i]), float('nan')
    p_lo, p_mid, p_hi = float(pgram[i-1]), float(pgram[i]), float(pgram[i+1])
    denom = p_lo + p_hi - 2 * p_mid          # < 0 for a proper maximum
    if denom >= 0:
        return float(heights[i]), float('nan')
    dh = float(heights[1] - heights[0])
    delta = 0.5 * (p_lo - p_hi) / denom      # fractional-bin offset in [-0.5, 0.5]
    h_refined = float(heights[i]) + delta * dh
    sigma_gauss = dh * float(np.sqrt(p_mid / abs(denom)))
    return h_refined, sigma_gauss


def estimate_rh_signal(elev_deg: np.ndarray, snr_db: np.ndarray,
                       signal: c.Signal, *,
                       min_pts: int | None = None) -> dict:
    """Lomb-Scargle peak -> RH for one (arc-slice, signal).
    Returns {'rh': m, 'sigma_rh': m, 'p2n': ratio, 'edge_hit': bool}.
    NaN values when the SNR column is empty or the peak fit fails.

    `min_pts` lets callers (e.g. windowed analysis) override the default
    per-arc threshold with a smaller value appropriate for short windows.
    """
    if min_pts is None:
        min_pts = c.MIN_ARC_PTS
    nan_out = {'rh': np.nan, 'sigma_rh': np.nan, 'p2n': np.nan, 'edge_hit': False}

    mask = snr_db > 0
    if mask.sum() < min_pts:
        return nan_out

    e = elev_deg[mask]
    snr_lin = 10 ** (snr_db[mask] / 20.0)            # dB-Hz -> volts
    se = np.sin(np.radians(e))
    coeffs = np.polyfit(se, snr_lin, c.DETREND_ORDER)
    dsnr = snr_lin - np.polyval(coeffs, se)

    heights = np.linspace(c.RH_MIN, c.RH_MAX, c.LS_NHEIGHTS)
    omegas = 4 * np.pi * heights / signal.wavelength_m
    pgram = lombscargle(se, dsnr, omegas, normalize=True)

    i = int(np.argmax(pgram))
    p_max = float(pgram[i])
    p_med = float(np.median(pgram))
    p2n = p_max / p_med if p_med > 0 else np.nan

    edge_tol = max(1, c.LS_NHEIGHTS // 100)          # within 1% of grid edges
    edge_hit = (i < edge_tol) or (i >= c.LS_NHEIGHTS - edge_tol)

    h_refined, sigma_gauss = _peak_refine(pgram, heights, i)
    sigma_rh = (sigma_gauss / np.sqrt(p2n)
                if np.isfinite(sigma_gauss) and np.isfinite(p2n) and p2n > 0
                else np.nan)

    return {'rh': h_refined, 'sigma_rh': sigma_rh,
            'p2n': p2n, 'edge_hit': bool(edge_hit)}


def estimate_arc(arc_df: pd.DataFrame) -> dict:
    """Run all applicable signals on one arc; returns a flat row dict."""
    sat = int(arc_df.sat.iloc[0])
    t0_utc = arc_df.time_utc.min()
    t1_utc = arc_df.time_utc.max()
    row = {
        'arc_id':         int(arc_df.arc_id.iloc[0]),
        'sat':            sat,
        'constellation':  c.constellation_for_sat(sat),
        'dir':            str(arc_df['dir'].iloc[0]),
        'pass_id':        int(arc_df.pass_id.iloc[0]),
        'n_pts':          len(arc_df),
        'az_mean':        round(float(arc_df.azim.mean()), 2),
        'year':           int(arc_df.year.iloc[0]),
        'doy':            int(arc_df.doy.iloc[0]),
        't_start_sec':    float(arc_df.sec.min()),
        't_end_sec':      float(arc_df.sec.max()),
        't_start_utc':    t0_utc,
        't_end_utc':      t1_utc,
        't_mid_utc':      t0_utc + (t1_utc - t0_utc) / 2,
    }
    for sig in c.signals_for_sat(sat):
        res = estimate_rh_signal(
            arc_df.elev.values, arc_df[sig.snr_col].values, sig)
        row[f'RH_{sig.name}']    = (round(res['rh'], 4)
                                     if np.isfinite(res['rh']) else np.nan)
        row[f'sigma_{sig.name}'] = (round(res['sigma_rh'], 4)
                                     if np.isfinite(res['sigma_rh']) else np.nan)
        row[f'p2n_{sig.name}']   = (round(res['p2n'], 2)
                                     if np.isfinite(res['p2n']) else np.nan)
        row[f'edge_{sig.name}']  = bool(res['edge_hit'])
    return row


def process_arcs(snr_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end per-day: segment + multi-signal estimation + P2N gate.
    Returns wide DataFrame, one row per arc.
    """
    arcs = segment_arcs(snr_df)
    if len(arcs) == 0:
        return pd.DataFrame()

    rows = [estimate_arc(g) for _, g in arcs.groupby('arc_id', sort=True)]
    df = pd.DataFrame(rows).sort_values(['t_start_sec', 'sat']).reset_index(drop=True)

    # Quality gates: null out RH (and its sigma) where p2n is too low or the
    # retrieval is pinned at the search-window edge
    for sig in c.ENABLED_SIGNALS:
        rh_col   = f'RH_{sig.name}'
        sig_col  = f'sigma_{sig.name}'
        p2n_col  = f'p2n_{sig.name}'
        edge_col = f'edge_{sig.name}'
        if p2n_col not in df.columns:
            continue
        bad = (df[p2n_col] < c.P2N_MIN) | df[edge_col].fillna(False)
        df.loc[bad, rh_col]  = np.nan
        df.loc[bad, sig_col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Sub-arc / windowed observations
#
# For rogue-wave timescales the full-arc averaging above is too slow
# (~30 min cadence). The functions below slide a fixed-length window through
# each arc and run `estimate_rh_signal` per window per signal, producing a
# long-form observation stream at the seconds-to-minutes timescale required
# for transient detection.
# ---------------------------------------------------------------------------

def rolling_rh_per_arc(arc_long_df: pd.DataFrame,
                        window_sec: float = c.WINDOW_SEC,
                        stride_sec: float = c.STRIDE_SEC) -> pd.DataFrame:
    """Slide a window through one arc's per-sample SNR data; run
    `estimate_rh_signal` on each window for every applicable signal.

    Parameters
    ----------
    arc_long_df : per-sample DataFrame for ONE arc (single arc_id), e.g.
                  `segment_arcs(snr_df).query('arc_id == X')`.
    window_sec  : window length in seconds.
    stride_sec  : window stride in seconds.

    Returns long-form DataFrame, one row per (window, signal) with a finite
    retrieval:
        t_center_utc, arc_id, sat, constellation, dir, pass_id,
        signal, snr_col, n_pts_window, elev_center, azim_center,
        rh, sigma, p2n, edge_hit
    """
    if arc_long_df.empty:
        return pd.DataFrame()

    sat = int(arc_long_df.sat.iloc[0])
    direction = str(arc_long_df['dir'].iloc[0])
    pass_id = int(arc_long_df.pass_id.iloc[0])
    arc_id = int(arc_long_df.arc_id.iloc[0])
    constellation = c.constellation_for_sat(sat)
    applicable = c.signals_for_sat(sat)
    if not applicable:
        return pd.DataFrame()

    t_min = arc_long_df.time_utc.min()
    t_max = arc_long_df.time_utc.max()
    half = pd.Timedelta(seconds=window_sec / 2)
    stride = pd.Timedelta(seconds=stride_sec)

    # arc too short for even one full window
    if t_max - t_min < pd.Timedelta(seconds=window_sec):
        return pd.DataFrame()

    t_centers = pd.date_range(t_min + half, t_max - half, freq=stride)

    out = []
    # Sort once and pre-extract numpy arrays; use searchsorted for fast
    # window slicing. (Comparing tz-aware datetime64 via searchsorted needs
    # both sides in the same naive ns representation.)
    sorted_df = arc_long_df.sort_values('time_utc')
    times_ns = sorted_df.time_utc.dt.tz_convert('UTC').dt.tz_localize(None)\
                       .astype('datetime64[ns]').astype('int64').to_numpy()
    elev_arr = sorted_df.elev.to_numpy()
    azim_arr = sorted_df.azim.to_numpy()
    snr_cols = {sig.snr_col: sorted_df[sig.snr_col].to_numpy()
                for sig in applicable}
    half_ns = int(half.total_seconds() * 1e9)

    for t_c in t_centers:
        t_c_ns = pd.Timestamp(t_c).tz_convert('UTC').tz_localize(None)\
                                   .to_datetime64().astype('datetime64[ns]')\
                                   .astype('int64')
        lo_idx = int(np.searchsorted(times_ns, t_c_ns - half_ns, side='left'))
        hi_idx = int(np.searchsorted(times_ns, t_c_ns + half_ns, side='right'))
        n_pts = hi_idx - lo_idx
        if n_pts < c.MIN_WIN_PTS:
            continue
        e = elev_arr[lo_idx:hi_idx]
        a = azim_arr[lo_idx:hi_idx]
        elev_center = float(np.median(e))
        azim_center = float(np.median(a))

        for sig in applicable:
            res = estimate_rh_signal(
                e, snr_cols[sig.snr_col][lo_idx:hi_idx], sig,
                min_pts=c.MIN_WIN_PTS,
            )
            if not np.isfinite(res['rh']):
                continue
            out.append({
                't_center_utc':  t_c,
                'arc_id':        arc_id,
                'sat':           sat,
                'constellation': constellation,
                'dir':           direction,
                'pass_id':       pass_id,
                'signal':        sig.name,
                'snr_col':       sig.snr_col,
                'n_pts_window':  n_pts,
                'elev_center':   round(elev_center, 3),
                'azim_center':   round(azim_center, 2),
                'rh':            round(float(res['rh']), 4),
                'sigma':         (round(float(res['sigma_rh']), 4)
                                  if np.isfinite(res['sigma_rh']) else np.nan),
                'p2n':           round(float(res['p2n']), 2),
                'edge_hit':      bool(res['edge_hit']),
            })
    return pd.DataFrame(out)


def process_arcs_windowed(snr_df: pd.DataFrame,
                           window_sec: float = c.WINDOW_SEC,
                           stride_sec: float = c.STRIDE_SEC) -> pd.DataFrame:
    """End-to-end per-day windowed pipeline: segment + rolling-window
    multi-signal RH estimation + quality gates.

    Returns long-form DataFrame (one row per surviving window×signal). Use
    this in place of `process_arcs` when you need sub-arc time resolution
    (rogue waves, fast tide rate-of-change, etc.). Per-arc analysis is
    still appropriate for tide tracking — both can coexist.
    """
    arcs = segment_arcs(snr_df)
    if arcs.empty:
        return pd.DataFrame()

    frames = [rolling_rh_per_arc(g, window_sec=window_sec, stride_sec=stride_sec)
              for _, g in arcs.groupby('arc_id', sort=True)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    obs = pd.concat(frames, ignore_index=True)

    # Quality gates: drop windows with low P2N or edge-hit retrievals
    keep = (obs['p2n'] >= c.P2N_WIN_MIN) & (~obs['edge_hit'])
    obs = obs[keep].reset_index(drop=True)
    obs = obs.sort_values(['t_center_utc', 'sat', 'signal']).reset_index(drop=True)
    return obs


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    YEAR, DOY = 2026, 1
    print(f'Station {c.STATION}  year {YEAR}  doy {DOY}')
    print(f'snr file: {snr_path(YEAR, DOY)}')

    snr = load_snr(YEAR, DOY)
    print(f'Loaded {len(snr):,} SNR rows, {len(snr.columns)} columns')
    print(f'PRN range: {snr.sat.min()}-{snr.sat.max()}  '
          f'({snr.sat.nunique()} unique sats)')

    arcs = process_arcs(snr)
    print(f'\nArcs produced: {len(arcs)}')
    print(f'\nConstellation breakdown:')
    print(arcs.constellation.value_counts().to_string())

    print(f'\nPer-signal summary (post-P2N + edge-hit gate):')
    print(f'  {"signal":8s}  {"n":>3s}  {"median RH":>10s}  {"std RH":>8s}  '
          f'{"med sigma":>10s}  {"edge-hits":>10s}')
    for sig in c.ENABLED_SIGNALS:
        rh_col   = f'RH_{sig.name}'
        sig_col  = f'sigma_{sig.name}'
        edge_col = f'edge_{sig.name}'
        if rh_col not in arcs.columns:
            continue
        vals = arcs[rh_col].dropna()
        sigs = arcs[sig_col].dropna()
        n_edge = int(arcs[edge_col].fillna(False).sum())
        if len(vals):
            print(f'  {sig.name:8s}  {len(vals):3d}  {vals.median():10.3f}  '
                  f'{vals.std():8.3f}  {sigs.median()*100:>8.1f} cm  '
                  f'{n_edge:10d}')
        else:
            print(f'  {sig.name:8s}  no valid retrievals')

    print(f'\nFirst 8 arcs (RH ± sigma in m):')
    show_cols = ['arc_id', 'sat', 'constellation', 'dir', 'az_mean', 'n_pts',
                 't_mid_utc']
    print(arcs[show_cols].head(8).to_string(index=False))
    for sig in c.ENABLED_SIGNALS:
        if f'RH_{sig.name}' not in arcs.columns:
            continue
        rhs   = arcs[f'RH_{sig.name}'].head(8)
        sigs  = arcs[f'sigma_{sig.name}'].head(8)
        line  = f'  {sig.name:8s}: '
        line += '  '.join(f'{r:6.3f}±{s*100:>4.1f}cm' if pd.notna(r) else '     —     '
                          for r, s in zip(rhs, sigs))
        print(line)

    # --- Windowed observation pipeline ---
    print(f'\n=== Windowed observations '
          f'(window={c.WINDOW_SEC}s, stride={c.STRIDE_SEC}s) ===')
    import time as _time
    t0 = _time.perf_counter()
    obs = process_arcs_windowed(snr)
    dt = _time.perf_counter() - t0
    print(f'Built {len(obs):,} windowed obs in {dt:.1f}s')
    if len(obs):
        print(f'Time span: {obs.t_center_utc.min()} -> {obs.t_center_utc.max()}')
        print()
        print(f'Per-signal counts and median sigma:')
        for sig in c.ENABLED_SIGNALS:
            sub = obs[obs.signal == sig.name]
            if not len(sub):
                continue
            print(f'  {sig.name:8s}  n={len(sub):5d}  '
                  f'median RH={sub.rh.median():.3f}  '
                  f'median σ={sub.sigma.median()*100:5.1f} cm  '
                  f'median p2n={sub.p2n.median():.1f}')

    # Demonstrate access to per-sample time series within an arc (needed for
    # the streaming/Kalman variant: rolling Lomb-Scargle windows).
    long_df = segment_arcs(snr)
    print(f'\nLong-form DataFrame from segment_arcs: '
          f'{len(long_df):,} rows, columns include time_utc')
    sample_arc = long_df[long_df.arc_id == long_df.arc_id.iloc[0]]
    print(f'  Sample arc {int(sample_arc.arc_id.iloc[0])} '
          f'(sat={int(sample_arc.sat.iloc[0])}, dir={sample_arc.dir.iloc[0]}): '
          f'{len(sample_arc)} points spanning '
          f'{sample_arc.time_utc.min()} -> {sample_arc.time_utc.max()} '
          f'({(sample_arc.time_utc.max() - sample_arc.time_utc.min()).total_seconds()/60:.1f} min)')