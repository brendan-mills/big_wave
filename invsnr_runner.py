"""Wrapper around `gnssrefl.invsnr_cl.invsnr` that produces a
`state.parquet`-shaped DataFrame.

Drop-in replacement for stages 3+4 of the pipeline (multi-sat binning +
Kalman filter). invsnr does its own joint inversion across satellites and
frequencies using a B-spline RH(t) forward model, so we skip the windowed-
binning + KF complexity entirely and let invsnr smooth the trajectory.

Stage 2 (windowed obs) is still produced because the event/burst detector
needs per-(sat, signal, window) obs as input — invsnr only outputs the
smoothed RH(t), not raw per-observation retrievals.

Usage: see `main.USE_INVSNR`. Set True to swap in this wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import config as c

# gnssrefl reads these at import time, so set them before the imports below
os.environ.setdefault('REFL_CODE', str(c.REFL_CODE))
os.environ.setdefault('ORBITS',    str(c.ORBITS_DIR))
os.environ.setdefault('EXE',       str(c.EXE_DIR))

from gnssrefl.invsnr_cl import invsnr                             # noqa: E402
from gnssrefl.invsnr_input import invsnr_input                    # noqa: E402


# ---------------------------------------------------------------------------
# Configuration shim — writes the JSON file invsnr reads from $REFL_CODE/input/
# ---------------------------------------------------------------------------

def ensure_config(
    station: str = c.STATION,
    lat: float = c.LAT,
    lon: float = c.LON,
    height: float = c.ANT_HEIGHT_ELL,
    rh_min: float = c.RH_MIN,
    rh_max: float = c.RH_MAX,
    el_min: float = c.EL_MIN,
    el_max: float = c.EL_MAX,
    azim_min: float = c.AZ_MIN,
    azim_max: float = c.AZ_MAX,
    peak2noise: float = 3.0,
) -> None:
    """Create or update the invsnr JSON config for this station."""
    invsnr_input(
        station=station,
        h1=rh_min, h2=rh_max,
        e1=el_min, e2=el_max,
        azim1=azim_min, azim2=azim_max,
        lat=lat, lon=lon, height=height,
        peak2noise=peak2noise,
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_OUT_COLS = ['year', 'month', 'day', 'hour', 'minute', 'second',
             'rh', 'doy', 'mjd', 'n_retrievals']


def _default_output_path(station: str, ext: str = 'txt') -> Path:
    """Where invsnr writes its output if `outfile_name` is empty.

    Empirically gnssrefl writes to `$REFL_CODE/Files/{station}/{station}_invsnr.{ext}`,
    NOT directly under $REFL_CODE as the source hints. The intermediate
    `Files/{station}/` is created automatically.
    """
    return (Path(os.environ['REFL_CODE']) / 'Files' / station
            / f'{station}_invsnr.{ext}')


def parse_output(path: Path | str) -> pd.DataFrame:
    """Parse the text/csv file invsnr produces.

    Columns: YYYY MM DD HH MM SS RH(m) doy MJD Nretrievals.
    Returns DataFrame with tz-aware UTC `t_utc`, plus `rh` and `n_retrievals`.
    """
    path = Path(path)
    df = pd.read_csv(
        path, comment='%', sep=r'\s+|,', header=None,
        names=_OUT_COLS, engine='python',
    )
    if df.empty:
        return df
    t = pd.to_datetime(df[['year', 'month', 'day',
                            'hour', 'minute', 'second']]
                       ).dt.tz_localize('UTC')
    return pd.DataFrame({
        't_utc':         t,
        'rh':            df['rh'].astype(float),
        'n_retrievals':  df['n_retrievals'].astype(int),
    })


# ---------------------------------------------------------------------------
# State-DataFrame shaping
# ---------------------------------------------------------------------------

def to_state(invsnr_df: pd.DataFrame, tide_model,
             antenna_msl: float = c.ANTENNA_MSL_M,
             sigma_estimate_m: float = 0.10) -> pd.DataFrame:
    """Shape parsed invsnr output into a state.parquet-compatible DataFrame.

    invsnr doesn't output per-sample uncertainty (it's a B-spline fit, no
    explicit covariance reported), so we use a flat `sigma_estimate_m`.
    Refine this later by computing residuals against raw obs if desired.

    Output columns match what `estimate.run_batch` produces:
        t_utc, sat, signal, eta_m, eta_sigma_m, tide_m, water_level_m,
        innov, mahal2
    """
    if invsnr_df.empty:
        return pd.DataFrame()

    times = invsnr_df['t_utc']
    rh    = invsnr_df['rh'].to_numpy()

    tide = np.asarray(tide_model.predict(times.tolist()))
    water_level = antenna_msl - rh
    eta = water_level - tide

    # NB: use `times` (Series) not `times.values` — .values strips the
    # tz-awareness and downstream `detect.compute_innovations` requires UTC.
    return pd.DataFrame({
        't_utc':         times.reset_index(drop=True),
        'sat':           -1,             # sentinel: invsnr is multi-sat fused
        'signal':        'invsnr',
        'eta_m':         eta,
        'eta_sigma_m':   sigma_estimate_m,
        'tide_m':        tide,
        'water_level_m': water_level,
        'innov':         0.0,            # no per-update innovations for spline
        'mahal2':        0.0,
    })


# ---------------------------------------------------------------------------
# Per-day caching
# ---------------------------------------------------------------------------

def per_day_state_path(year: int, doy: int) -> Path:
    """Where per-day invsnr state caches land:
    `data/results/{year}/invsnr/{doy:03d}_state.parquet`."""
    return c.RESULTS_DIR / f'{year}' / 'invsnr' / f'{doy:03d}_state.parquet'


# ---------------------------------------------------------------------------
# Chunked processing
#
# Each day's B-spline used to be fit independently, leaving the spline
# endpoints at 00:00/24:00 weakly constrained → jagged midnight seams when
# stitched (which leaked into false 'surge' events). gnssrefl's invsnr fits
# ONE continuous spline across a doy→doy_end range (see spline_functions
# .snr2spline: knots span the whole range, first/last knots are meant to be
# ignored). So we fit overlapping multi-day CHUNKS and keep only the central
# days — the unconstrained endpoints fall in the discarded buffer. Output
# per-day caching is unchanged; only the fitting granularity differs.
# ---------------------------------------------------------------------------

CHUNK_DAYS   = 6     # central output days fit together as one continuous spline
                     # (~25 s/day, ~2.5 min/chunk — the snrfit cost/day sweet
                     #  spot; bigger chunks block longer with little speedup)
OVERLAP_DAYS = 1     # buffer days fit on each side then discarded (kill seams)

import time as _time     # avoid shadowing the `time` column name


def _snr_exists(year: int, doy: int, station: str) -> bool:
    """True if a snr file exists for this station/year/doy."""
    yy = year % 100
    snr_dir = Path(os.environ['REFL_CODE']) / str(year) / 'snr' / station
    return bool(list(snr_dir.glob(f'{station}{doy:03d}0.{yy:02d}.snr*')))


def _day_bounds(year: int, doy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) tz-aware UTC bounds of one (year, doy)."""
    start = pd.Timestamp(f'{year}-01-01', tz='UTC') + pd.Timedelta(days=doy - 1)
    return start, start + pd.Timedelta(days=1)


def _contiguous_runs(dates: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Split sorted (year, doy) dates into maximal runs of consecutive days
    within a single year (a day d is consecutive to d+1, same year). Runs are
    the units we fit continuous splines over; gaps and year boundaries break
    them (and become the only places a seam can remain)."""
    runs: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    for d in dates:
        if cur and d[0] == cur[-1][0] and d[1] == cur[-1][1] + 1:
            cur.append(d)
        else:
            if cur:
                runs.append(cur)
            cur = [d]
    if cur:
        runs.append(cur)
    return runs


def _fit_chunk(
    year: int,
    fit_first: int,
    fit_last: int,
    keep_doys: list[int],
    *,
    tide_model,
    station: str = c.STATION,
    signal: str = 'L1+L2+L5',
    knot_space_hr: int = 3,
    delta_out_sec: int = 300,
    constel: str | None = None,
    peak2noise: float = 2.5,
    refraction: bool = True,
    sigma_estimate_m: float = 0.10,
    outfile_type: str = 'txt',     # 'csv' triggers a format-string bug in
                                    # gnssrefl 4.1.5 (spline_functions L355) —
                                    # stay on 'txt' until fixed upstream.
) -> dict[int, pd.DataFrame]:
    """Fit one continuous invsnr spline over [fit_first, fit_last] (day-of-year,
    same year), then split the output and cache only the `keep_doys`. The
    buffer days outside `keep_doys` exist purely to constrain the spline
    endpoints and are discarded. Returns {doy: state} for the kept days."""
    ensure_config(
        station=station,
        rh_min=c.RH_MIN, rh_max=c.RH_MAX,
        el_min=c.EL_MIN, el_max=c.EL_MAX,
        azim_min=c.AZ_MIN, azim_max=c.AZ_MAX,
        peak2noise=peak2noise,
    )

    # invsnr writes to a fixed default path. Remove any stale file so we
    # don't accidentally parse a previous chunk's output.
    out_path = _default_output_path(station, ext=outfile_type)
    if out_path.exists():
        out_path.unlink()

    invsnr(
        station=station, year=year, doy=fit_first,
        doy_end=(fit_last if fit_last != fit_first else None),
        signal=signal,
        knot_space=knot_space_hr,
        delta_out=delta_out_sec,
        constel=constel,
        peak2noise=peak2noise,
        refraction=refraction,
        plt=False, snrfigs=False, lspfigs=False,
        outfile_type=outfile_type,
    )

    if not out_path.exists():
        raise RuntimeError(
            f'invsnr produced no output for {station} {year} '
            f'doy {fit_first}-{fit_last}. '
            'Check stdout for "no arcs found" or similar errors.'
        )

    parsed = parse_output(out_path)
    results: dict[int, pd.DataFrame] = {}
    for doy in keep_doys:
        lo, hi = _day_bounds(year, doy)
        day_rows = parsed[(parsed['t_utc'] >= lo) & (parsed['t_utc'] < hi)]
        state = to_state(day_rows, tide_model,
                         antenna_msl=c.ANTENNA_MSL_M,
                         sigma_estimate_m=sigma_estimate_m)
        if state.empty:
            continue
        cache = per_day_state_path(year, doy)
        cache.parent.mkdir(parents=True, exist_ok=True)
        state.to_parquet(cache, compression='snappy', index=False)
        results[doy] = state
    return results


def run_one_day(
    year: int,
    doy: int,
    *,
    tide_model,
    force: bool = False,
    station: str = c.STATION,
    **kwargs,
) -> pd.DataFrame:
    """Return one day's state, from cache or a single-day fit.

    NOTE: a bare single-day fit has the weakly-constrained midnight endpoints
    this module's chunking is designed to avoid. Prefer `run_range` (which
    fits overlapping chunks) for production; this is for direct single-day use
    and as a fallback.
    """
    cache = per_day_state_path(year, doy)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    # gnssrefl's invsnr calls sys.exit() (→ SystemExit) if the snr file is
    # missing. Pre-check and raise a normal exception instead.
    if not _snr_exists(year, doy, station):
        raise FileNotFoundError(
            f'no snr file for {station} {year} doy {doy} '
            '(rinex2snr produced none — likely missing/empty RINEX)'
        )

    res = _fit_chunk(year, doy, doy, [doy], tide_model=tide_model,
                     station=station, **kwargs)
    state = res.get(doy)
    if state is None or state.empty:
        raise RuntimeError(
            f'invsnr produced no rows for {station} {year} doy {doy}.')
    return state


def run_range(
    date_filter,
    *,
    tide_model,
    force: bool = False,
    fail_fast: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Fit `date_filter` in overlapping multi-day chunks, then stitch the
    per-day caches into one state DataFrame.

    `date_filter` is an iterable of `(year, doy)` tuples (the shape
    `main.build_date_filter` produces). Days are grouped into contiguous
    same-year runs; each run is fit in `CHUNK_DAYS`-day chunks with
    `OVERLAP_DAYS` buffer on each side (discarded) so the kept splines are
    continuous across midnight. Per-day caches make resumes cheap; set
    `force=True` to refit (required to replace older single-day caches).
    """
    dates = sorted(set(date_filter))
    if not dates:
        return pd.DataFrame()
    station = kwargs.get('station', c.STATION)

    print(f'invsnr: processing {len(dates)} day(s) '
          f'(force={force}, chunk={CHUNK_DAYS}d +{OVERLAP_DAYS}d overlap)')
    t_start = _time.perf_counter()

    # --- Phase 1: fit needed days as overlapping chunks (seam-free splines) ---
    days_with_snr = [(y, d) for (y, d) in dates if _snr_exists(y, d, station)]
    n_missing = len(dates) - len(days_with_snr)
    if n_missing:
        print(f'  ({n_missing} day(s) have no snr file — skipped)')

    for run_days in _contiguous_runs(days_with_snr):
        year = run_days[0][0]
        doys = [d for (_, d) in run_days]
        n = len(doys)
        i = 0
        while i < n:
            chunk = doys[i:i + CHUNK_DAYS]
            need = [d for d in chunk
                    if force or not per_day_state_path(year, d).exists()]
            if need:
                lo_i = max(0, i - OVERLAP_DAYS)
                hi_i = min(n - 1, i + len(chunk) - 1 + OVERLAP_DAYS)
                fit_first, fit_last = doys[lo_i], doys[hi_i]
                t0 = _time.perf_counter()
                try:
                    _fit_chunk(year, fit_first, fit_last, chunk,
                               tide_model=tide_model, **kwargs)
                    el = _time.perf_counter() - t0
                    print(f'  fit {year}-{fit_first:03d}..{fit_last:03d} '
                          f'-> keep {chunk[0]:03d}..{chunk[-1]:03d}  '
                          f'({len(chunk)}d, {el:.1f}s)')
                except (Exception, SystemExit) as e:
                    if fail_fast:
                        raise
                    # One bad day in the span fails the whole chunk; fall back
                    # to single-day fits so the good days still land (those
                    # days keep a midnight seam — acceptable for a few days).
                    print(f'  fit {year}-{fit_first:03d}..{fit_last:03d}: '
                          f'FAILED  {type(e).__name__}: {e}  '
                          f'-> retrying day-by-day')
                    for d in chunk:
                        try:
                            run_one_day(year, d, tide_model=tide_model,
                                        force=True, **kwargs)
                        except (Exception, SystemExit) as e2:
                            print(f'    day {year}-{d:03d}: '
                                  f'FAILED  {type(e2).__name__}: {e2}')
            i += CHUNK_DAYS

    # --- Phase 2: stitch from per-day caches ---
    frames = []
    for (year, doy) in dates:
        cache = per_day_state_path(year, doy)
        if not cache.exists():
            continue
        df = pd.read_parquet(cache)
        if len(df):
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    stitched = (pd.concat(frames, ignore_index=True)
                  .sort_values('t_utc')
                  .reset_index(drop=True))
    total_s = _time.perf_counter() - t_start
    print(f'invsnr: stitched {len(stitched):,} rows from {len(frames)}/'
          f'{len(dates)} days in {total_s:.1f}s')
    return stitched


def run(
    station: str,
    year: int,
    doy_start: int,
    doy_end: int | None,
    *,
    tide_model,
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Backwards-compatible single-year wrapper around `run_range`.
    Days are processed individually with per-day caching."""
    if doy_end is None:
        doy_end = doy_start
    date_filter = {(year, d) for d in range(doy_start, doy_end + 1)}
    return run_range(date_filter, tide_model=tide_model,
                     force=force, station=station, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from tide import GreenlandTideModel

    YEAR, DOY_START, DOY_END = 2025, 90, 92    # 3-day test
    print(f'Running invsnr for {c.STATION} {YEAR} doys {DOY_START}-{DOY_END}')

    tm = GreenlandTideModel(c.LAT, c.LON)
    state = run(c.STATION, YEAR, DOY_START, DOY_END, tide_model=tm)

    print(f'\nReturned state: {len(state):,} rows')
    if len(state):
        print(f'  time range : {state.t_utc.min()} -> {state.t_utc.max()}')
        print(f'  η range    : {state.eta_m.min():+.3f} -> {state.eta_m.max():+.3f}  m')
        print(f'  water-level: {state.water_level_m.min():+.3f} -> '
              f'{state.water_level_m.max():+.3f}  m')
        print(f'  σ_η (fixed): {state.eta_sigma_m.iloc[0]*100:.1f} cm')
        print(f'\nFirst 5 rows:')
        print(state.head(5).to_string(index=False))
