"""SNR file processing for SSiSLS.

Wraps gnssrefl for RINEX -> snr66 conversion + reading, then provides arc
segmentation and the per-(window, sat, signal) SNR-residual ROUGHNESS used for
wave-train detection (`process_arcs_roughness`). The water-level reference comes
from invsnr (see invsnr_runner), which reads the snr66 directly — so the old
custom Lomb-Scargle RH retrieval was removed.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config as c

# gnssrefl reads these at import time, so set them before the imports below.
# REFL_CODE is rate-specific (snr66 tree) so 1 Hz output never clobbers the 15 s
# tree for overlapping doys; ORBITS/EXE stay shared (gnssrefl reads them from
# their own env vars). Set REFL_CODE explicitly (not setdefault) so an inherited
# shell value can't pin us to the wrong tree.
os.environ['REFL_CODE'] = str(c.SNR_REFL_CODE)
os.environ.setdefault('ORBITS', str(c.ORBITS_DIR))
os.environ.setdefault('EXE',    str(c.EXE_DIR))

from gnssrefl.gps import snr_name                          # noqa: E402
from gnssrefl.read_snr_files import read_snr               # noqa: E402
from gnssrefl.rinex2snr_cl import rinex2snr                # noqa: E402


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
    base = c.SNR_REFL_CODE / f'{year}' / 'snr' / c.STATION / fname
    if base.exists():
        return base
    gz = base.with_suffix(base.suffix + '.gz')
    if gz.exists():
        return gz
    return base  # may not exist yet; caller decides what to do


def _stage_rinex(src: Path) -> Path:
    """Place an uncompressed copy of `src` in the CWD for gnssrefl's nolook
    search, which recognizes plain '.YYd' (Hatanaka) / '.YYo' but NOT
    '.YYd.gz' / '.YYd.Z'. Strips a '.gz'/'.Z' layer (keeping Hatanaka so
    gnssrefl's own CRX2RNX expands it); copies a plain file as-is. Returns the
    staged path.
    """
    if src.suffix in ('.gz', '.Z'):
        dst = Path.cwd() / src.name[:-len(src.suffix)]
        if not dst.exists():
            if src.suffix == '.gz':
                with gzip.open(src, 'rb') as fi, open(dst, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
            else:  # '.Z' is LZW — Python's gzip can't read it; use the CLI
                tmp = Path.cwd() / src.name
                shutil.copy(src, tmp)
                subprocess.run(['uncompress', '-f', str(tmp)], check=True)
        return dst
    dst = Path.cwd() / src.name
    if not dst.exists():
        shutil.copy(src, dst)
    return dst


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

    # gnssrefl's nolook translator reads the RINEX from — and writes its
    # intermediates (.YYo, re-gzipped .YYd) into — the CWD. Run inside a
    # throwaway temp dir so (a) concurrent day-workers never collide on the
    # staged file / gnssrefl temp files, and (b) the bulky 1 Hz intermediates
    # are auto-removed. The snr66 goes to $REFL_CODE (absolute), so it survives.
    cwd0 = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            _stage_rinex(src)
            rinex2snr(
                station=c.STATION, year=year, doy=doy,
                nolook=c.NOLOOK, orb=c.ORB, overwrite=force, dec=c.RINEX_DEC,
            )
        finally:
            os.chdir(cwd0)

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
# SNR roughness (sea-state / wave-train detection)
#
# A wave train roughens the reflecting surface, injecting FAST SNR fluctuations
# (~the wave period) that the slow geometry can't produce. In a short window the
# RH oscillation (period >= ~30 s) is a smooth low-order trend; the residual RMS
# after removing that trend is the fast-fluctuation "roughness". This needs NO
# frequency fit, so the short window that defeats RH retrieval is fine here.
# ---------------------------------------------------------------------------

def _window_roughness(t_rel: np.ndarray, snr_lin: np.ndarray) -> float:
    """RMS of the SNR residual after a low-order polynomial detrend in time,
    normalized by the window-mean SNR (dimensionless, so it's comparable across
    elevations/sats). NaN if too few points."""
    if len(snr_lin) < c.ROUGH_MIN_PTS:
        return float('nan')
    mean = float(np.mean(snr_lin))
    if mean <= 0:
        return float('nan')
    coeffs = np.polyfit(t_rel, snr_lin, c.ROUGH_DETREND_ORDER)
    resid = snr_lin - np.polyval(coeffs, t_rel)
    return float(np.sqrt(np.mean(resid ** 2)) / mean)


def rolling_roughness_per_arc(arc_long_df: pd.DataFrame,
                              window_sec: float = c.ROUGH_WIN_SEC,
                              stride_sec: float = c.ROUGH_STRIDE_SEC) -> pd.DataFrame:
    """Slide a short window through one arc; per signal, emit the SNR-residual
    roughness. Long-form, one row per (window, signal):
        t_center_utc, arc_id, sat, constellation, dir, pass_id, signal,
        snr_col, n_pts_window, elev_center, azim_center, roughness
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
    if t_max - t_min < pd.Timedelta(seconds=window_sec):
        return pd.DataFrame()
    t_centers = pd.date_range(t_min + half, t_max - half, freq=stride)

    sorted_df = arc_long_df.sort_values('time_utc')
    times_ns = sorted_df.time_utc.dt.tz_convert('UTC').dt.tz_localize(None)\
                       .astype('datetime64[ns]').astype('int64').to_numpy()
    elev_arr = sorted_df.elev.to_numpy()
    azim_arr = sorted_df.azim.to_numpy()
    snr_cols = {sig.snr_col: sorted_df[sig.snr_col].to_numpy() for sig in applicable}
    half_ns = int(half.total_seconds() * 1e9)

    out = []
    for t_c in t_centers:
        t_c_ns = pd.Timestamp(t_c).tz_convert('UTC').tz_localize(None)\
                                   .to_datetime64().astype('datetime64[ns]')\
                                   .astype('int64')
        lo = int(np.searchsorted(times_ns, t_c_ns - half_ns, side='left'))
        hi = int(np.searchsorted(times_ns, t_c_ns + half_ns, side='right'))
        if hi - lo < c.ROUGH_MIN_PTS:
            continue
        t_rel = (times_ns[lo:hi] - t_c_ns) / 1e9          # seconds, window-centered
        elev_center = float(np.median(elev_arr[lo:hi]))
        azim_center = float(np.median(azim_arr[lo:hi]))
        for sig in applicable:
            snr_db = snr_cols[sig.snr_col][lo:hi]
            mask = snr_db > 0
            if mask.sum() < c.ROUGH_MIN_PTS:
                continue
            r = _window_roughness(t_rel[mask], 10 ** (snr_db[mask] / 20.0))
            if not np.isfinite(r):
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
                'n_pts_window':  int(mask.sum()),
                'elev_center':   round(elev_center, 3),
                'azim_center':   round(azim_center, 2),
                'roughness':     round(r, 6),
            })
    return pd.DataFrame(out)


def process_arcs_roughness(snr_df: pd.DataFrame,
                           window_sec: float = c.ROUGH_WIN_SEC,
                           stride_sec: float = c.ROUGH_STRIDE_SEC) -> pd.DataFrame:
    """End-to-end per-day roughness: segment arcs + rolling roughness per signal,
    then add a per-arc calm baseline (median) and the relative `rough_ratio`
    used by the wave-train detector. Long-form DataFrame."""
    arcs = segment_arcs(snr_df)
    if arcs.empty:
        return pd.DataFrame()
    frames = [rolling_roughness_per_arc(g, window_sec, stride_sec)
              for _, g in arcs.groupby('arc_id', sort=True)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    obs = pd.concat(frames, ignore_index=True)
    # Per-arc calm floor -> relative roughness (a wave train spikes above it).
    obs['baseline'] = obs.groupby('arc_id')['roughness'].transform('median')
    obs['rough_ratio'] = obs['roughness'] / obs['baseline'].replace(0, np.nan)
    return obs.sort_values(['t_center_utc', 'sat', 'signal']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time as _time
    YEAR, DOY = 2025, 110            # 1 Hz smoke-test day (must exist in RINEX_DIR)
    print(f'Station {c.STATION}  year {YEAR}  doy {DOY}')
    print(f'snr file: {snr_path(YEAR, DOY)}')

    snr_df = load_snr(YEAR, DOY)
    print(f'Loaded {len(snr_df):,} SNR rows, {len(snr_df.columns)} columns; '
          f'PRN {snr_df.sat.min()}-{snr_df.sat.max()} ({snr_df.sat.nunique()} sats)')

    t0 = _time.perf_counter()
    rough = process_arcs_roughness(snr_df)
    print(f'\nRoughness obs: {len(rough):,} in {_time.perf_counter()-t0:.1f}s')
    if len(rough):
        print('  by constellation:', rough.constellation.value_counts().to_dict())
        print('  rough_ratio: '
              f'p50={rough.rough_ratio.quantile(.5):.2f}  '
              f'p99={rough.rough_ratio.quantile(.99):.2f}  '
              f'max={rough.rough_ratio.max():.2f}')