"""Top-level orchestrator for the SSiSLS pipeline.

Runs all stages for a date range in order, with per-stage caching:

    1. pipeline.process_folder            RINEX -> per-day per-arc parquet
    2. pipeline.process_folder_windowed   per-day windowed obs parquet
    3. estimate.bin_obs_by_time           multi-sat bin consensus
    4. estimate.run_batch                 Kalman state
    5. detect.detect_events               coherent-event log
    6. plots.*                            headline figures

Edit the constants in the `RUN CONFIG` block below, then click Run.
Each stage skips if its output parquet already exists; set FORCE = True
to reprocess all stages.

Multi-year ranges are supported via (year, doy) tuples. Range artifacts
land under `data/results/range/{tag}/` keyed by a multi-year tag like
`2025085-2026145`. Per-day caches stay under `data/results/{year}/...`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import config as c
import pipeline
import estimate
import detect


# =============================================================================
# RUN CONFIG — edit these constants then click Run
# =============================================================================

# (year, doy) inclusive bounds. Either can be None to mean "all available".
# START_DATE = (2025, 85)            # 2025-03-26
# END_DATE   = (2026, 145)           # 2026-05-25
START_DATE = (2025, 85)            # 2025-03-26
END_DATE   = (2025, 105)           

RINEX_FOLDER = c.RINEX_DIR         # raw RINEX folder (default from config)

BIN_SEC    = 100.0                 # multi-sat bin width (s)
FORCE      = True                 # True = reprocess everything (ignore caches)
MAKE_PLOTS = True                  # False to skip stage 6


# =============================================================================
# Multi-year date helpers
# =============================================================================

def date_in_range(date: tuple[int, int],
                  lo: tuple[int, int] | None,
                  hi: tuple[int, int] | None) -> bool:
    """Inclusive (year, doy) range test. None bound = open-ended."""
    if lo is not None and date < lo:
        return False
    if hi is not None and date > hi:
        return False
    return True


def build_date_filter(folder: Path,
                       lo: tuple[int, int] | None,
                       hi: tuple[int, int] | None
                       ) -> set[tuple[int, int]] | None:
    """Discover RINEX in folder and return the (year, doy) subset that falls
    within [lo, hi]. Returns None if both bounds are None (process all)."""
    if lo is None and hi is None:
        return None
    discovered = pipeline.discover_rinex(folder)
    return {(y, d) for (y, d, _) in discovered if date_in_range((y, d), lo, hi)}


def range_tag(date_filter: set[tuple[int, int]] | None,
              folder: Path) -> str:
    """Filename tag for the range — multi-year aware. Format:
       '{ys:04d}{ds:03d}-{ye:04d}{de:03d}'  e.g.  '2025085-2026145'."""
    if date_filter:
        dates = sorted(date_filter)
    else:
        dates = sorted({(y, d) for (y, d, _) in pipeline.discover_rinex(folder)})
    if not dates:
        return 'empty'
    (ys, ds), (ye, de) = dates[0], dates[-1]
    return f'{ys:04d}{ds:03d}-{ye:04d}{de:03d}'


# =============================================================================
# Output paths (cached across stages 3-6, under data/results/range/{tag}/)
# =============================================================================

def range_dir(tag: str) -> Path:
    return c.RESULTS_DIR / 'range' / tag


def binned_path(tag: str) -> Path:  return range_dir(tag) / 'binned.parquet'
def state_path(tag: str)  -> Path:  return range_dir(tag) / 'state.parquet'
def events_path(tag: str) -> Path:  return range_dir(tag) / 'events.parquet'


# =============================================================================
# Stage drivers
# =============================================================================

def stage_arcs(folder, date_filter, force):
    """[1/6] RINEX -> per-arc parquets."""
    print('\n[1/6] Per-arc preprocessing')
    return pipeline.process_folder(folder, date_filter=date_filter, force=force)


def stage_windowed(folder, date_filter, force):
    """[2/6] Per-day windowed obs (RH at sliding sub-arc windows)."""
    print('\n[2/6] Windowed observations')
    return pipeline.process_folder_windowed(folder, date_filter=date_filter, force=force)


def stage_bin(tag, obs_df, bin_sec, force):
    """[3/6] Multi-sat consensus binning."""
    out = binned_path(tag)
    if out.exists() and not force:
        print(f'\n[3/6] Binned obs    (cached) {out.relative_to(c.PROJECT_DIR)}')
        return pd.read_parquet(out)
    print(f'\n[3/6] Binning obs ({int(bin_sec)}s bins, multi-sat consensus)')
    binned = estimate.bin_obs_by_time(obs_df, bin_sec=bin_sec)
    out.parent.mkdir(parents=True, exist_ok=True)
    binned.to_parquet(out, compression='snappy', index=False)
    print(f'  {len(obs_df):,} raw obs -> {len(binned):,} bins  '
          f'(median σ {obs_df.sigma.median()*100:.0f} -> '
          f'{binned.sigma.median()*100:.0f} cm)')
    return binned


def stage_kalman(tag, binned_obs, tide_model, force):
    """[4/6] KF on binned obs."""
    out = state_path(tag)
    if out.exists() and not force:
        print(f'[4/6] KF state      (cached) {out.relative_to(c.PROJECT_DIR)}')
        state = pd.read_parquet(out)
        gated_p = out.with_name(out.stem.replace('state', 'gated') + '.parquet')
        gated = pd.read_parquet(gated_p) if gated_p.exists() else pd.DataFrame()
        return state, gated
    print(f'[4/6] Running Kalman filter on {len(binned_obs):,} binned obs')
    state, gated = estimate.run_batch(binned_obs, tide_model, save_to=out)
    print(f'  state rows: {len(state):,}  '
          f'water-level range: {state.water_level_m.min():.2f} -> '
          f'{state.water_level_m.max():.2f} m  '
          f'gated: {len(gated)}')
    return state, gated


def stage_detect(tag, obs_df, state_df, tide_model, force):
    """[5/6] Event detection on raw obs vs smoothed state."""
    out = events_path(tag)
    if out.exists() and not force:
        print(f'[5/6] Events        (cached) {out.relative_to(c.PROJECT_DIR)}')
        return pd.read_parquet(out)
    print(f'[5/6] Detecting events ({len(obs_df):,} raw obs vs state)')
    events, _ = detect.detect_events(obs_df, state_df, tide_model, save_to=out)
    print(f'  {len(events)} candidate event(s) detected')
    return events


def stage_plots():
    """[6/6] Headline plots — defers to plots.py."""
    print('\n[6/6] Plots')
    import subprocess
    subprocess.run([sys.executable, 'plots.py'], check=True,
                   cwd=str(c.PROJECT_DIR))


# =============================================================================
# Run
# =============================================================================

def run():
    folder = Path(RINEX_FOLDER)
    if not folder.exists():
        raise SystemExit(f'RINEX folder not found: {folder}')

    date_filter = build_date_filter(folder, START_DATE, END_DATE)
    tag = range_tag(date_filter, folder)
    print(f'Range: {tag}  '
          f'({"all" if date_filter is None else len(date_filter)} day(s))')

    t_start = time.perf_counter()

    # Stage 1: per-arc parquets (per day, cached under year folder)
    stage_arcs(folder, date_filter, FORCE)

    # Stage 2: per-day windowed obs (concatenated DataFrame returned)
    windowed = stage_windowed(folder, date_filter, FORCE)
    if windowed.empty:
        print('No windowed obs produced. Nothing more to do.')
        return

    # Derive day count from the obs timestamps (year/doy aren't columns
    # in the windowed long-form df)
    if 't_center_utc' in windowed.columns and len(windowed):
        days = windowed['t_center_utc'].dt.floor('D').unique()
        n_days = len(days)
    else:
        n_days = 0
    print(f'\n--- Range: {tag} '
          f'({n_days} days, {len(windowed):,} obs) ---')

    # Stage 3: multi-sat binning
    binned = stage_bin(tag, windowed, BIN_SEC, FORCE)
    if binned.empty:
        print('No binned obs. Stopping.')
        return

    # Stage 4: KF (tide model needed)
    from tide import GreenlandTideModel
    tm = GreenlandTideModel(c.LAT, c.LON)
    state, _ = stage_kalman(tag, binned, tm, FORCE)

    # Stage 5: event detection
    events = stage_detect(tag, windowed, state, tm, FORCE)

    # Stage 6: plots
    if MAKE_PLOTS:
        try:
            stage_plots()
        except Exception as e:
            print(f'  plots failed: {type(e).__name__}: {e}')

    elapsed = time.perf_counter() - t_start
    print(f'\n=== Pipeline complete in {elapsed:.1f}s ===')
    print(f'  per-arc parquets : data/results/{{year}}/*.parquet')
    print(f'  windowed obs     : data/results/{{year}}/windowed/{{doy:03d}}_obs.parquet')
    print(f'  range artifacts  : data/results/range/{tag}/')
    print(f'  detected events  : {len(events)}')
    if not events.empty:
        print(f'\n  Top 5 candidate events by confidence:')
        show = ['t_peak_utc', 'duration_sec', 'amplitude_m', 'direction',
                'n_sats_peak', 'n_bins', 'confidence']
        print(events[show].head(5).to_string(index=False))


if __name__ == '__main__':
    run()
