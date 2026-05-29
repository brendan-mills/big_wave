"""Top-level orchestrator for the SSiSLS 1 Hz pipeline.

Two-event design (see detect.py):
  1. invsnr    -> water-level state (the reference: RH(t) B-spline, eta=WL-tide)
  2. roughness -> per-day SNR-roughness obs (the wave-train signal)
  3. detect    -> surge (state vs tide) + roughness bursts -> events.parquet
  4. plots     -> water-level overview + roughness highlights

Edit the RUN CONFIG constants, then click Run. Each stage caches; FORCE re-runs.
Multi-year ranges via (year, doy) tuples; range artifacts land under
data/.../range/{tag}/, per-day caches under data/.../{year}/.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import config as c
import pipeline
import detect


# =============================================================================
# RUN CONFIG — edit then Run
# =============================================================================

# (year, doy) inclusive bounds. Either can be None to mean "all available".
START_DATE = (2025, 121)         # 2025-05-01, open-water season start
END_DATE   = (2025, 304)         # 2025-10-31, open-water season end

RINEX_FOLDER = c.RINEX_DIR         # raw RINEX folder (default from config)
FORCE        = False                # reprocess every stage (ignore caches)
FORCE_INVSNR = True               # refit invsnr even if per-day caches exist
                                   # (chunked, seam-free) while reusing snr66
MAKE_PLOTS   = True

# =============================================================================
# Date helpers + output paths
# =============================================================================

def date_in_range(date, lo, hi) -> bool:
    if lo is not None and date < lo:
        return False
    if hi is not None and date > hi:
        return False
    return True


def build_date_filter(folder, lo, hi):
    """(year, doy) subset of the RINEX in `folder` within [lo, hi]. None if both
    bounds are None (process all)."""
    if lo is None and hi is None:
        return None
    discovered = pipeline.discover_rinex(folder)
    return {(y, d) for (y, d, _) in discovered if date_in_range((y, d), lo, hi)}


def range_tag(date_filter, folder) -> str:
    """'{ys:04d}{ds:03d}-{ye:04d}{de:03d}', e.g. '2025110-2025240'."""
    dates = (sorted(date_filter) if date_filter
             else sorted({(y, d) for (y, d, _) in pipeline.discover_rinex(folder)}))
    if not dates:
        return 'empty'
    (ys, ds), (ye, de) = dates[0], dates[-1]
    return f'{ys:04d}{ds:03d}-{ye:04d}{de:03d}'


def range_dir(tag) -> Path:    return c.RESULTS_DIR / 'range' / tag
def state_path(tag) -> Path:   return range_dir(tag) / 'state.parquet'
def events_path(tag) -> Path:  return range_dir(tag) / 'events.parquet'


# =============================================================================
# Stages
# =============================================================================

def stage_snr(folder, date_filter) -> None:
    """[1/5] Ensure snr66 exists for the range (parallel rinex2snr). Both invsnr
    and roughness read snr66, so this must run first — otherwise a fresh dataset
    has no snr66 and invsnr skips every day."""
    print('\n[1/5] snr66 files')
    pipeline.ensure_snr_folder(folder, date_filter=date_filter)


def stage_invsnr(tag, date_filter, tide_model, force) -> pd.DataFrame:
    """[2/5] invsnr B-spline inversion -> water-level state (the reference).
    Per-day fits cached under {year}/invsnr/; stitched range output to
    state.parquet. A per-chunk timeout (config.INVSNR_TIMEOUT_SEC) skips any
    pathological chunk instead of hanging."""
    import invsnr_runner
    out = state_path(tag)
    print('\n[2/5] invsnr water-level state')
    if out.exists() and not force:
        print(f'  (cached) {out.relative_to(c.PROJECT_DIR)}')
        return pd.read_parquet(out)
    if not date_filter:
        raise SystemExit('invsnr requires a non-empty date filter.')
    state = invsnr_runner.run_range(date_filter, tide_model=tide_model, force=force)
    if state.empty:
        print('  invsnr produced no state.')
        return state
    out.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(out, compression='snappy', index=False)
    print(f'  {len(state):,} rows  water level '
          f'{state.water_level_m.min():.2f} -> {state.water_level_m.max():.2f} m')
    return state


def stage_roughness(folder, date_filter, force) -> pd.DataFrame:
    """[3/5] Per-day SNR roughness (parallel across days)."""
    print('\n[3/5] SNR roughness')
    return pipeline.process_folder_roughness(folder, date_filter=date_filter,
                                             force=force)


def stage_detect(tag, rough_df, state_df, force) -> pd.DataFrame:
    """[3/4] surge (state vs tide) + roughness bursts -> events."""
    out = events_path(tag)
    print('\n[4/5] Detect events (surge + roughness)')
    if out.exists() and not force:
        print(f'  (cached) {out.relative_to(c.PROJECT_DIR)}')
        return pd.read_parquet(out)
    events = detect.detect_events(rough_df, state_df, save_to=out)
    if not events.empty:
        counts = events['trigger'].value_counts().to_dict()
        print('  ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))
    else:
        print('  no events')
    return events


def stage_plots(tag) -> None:
    """[5/5] Water-level overview + roughness highlights."""
    print('\n[5/5] Plots')
    import plots_roughness
    plots_roughness.generate(tag)


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

    from tide import GreenlandTideModel
    tm = GreenlandTideModel(c.LAT, c.LON)

    stage_snr(folder, date_filter)         # create snr66 first (invsnr/roughness read it)

    state = stage_invsnr(tag, date_filter, tm, FORCE or FORCE_INVSNR)
    if state.empty:
        print('No invsnr state — stopping.')
        return

    rough = stage_roughness(folder, date_filter, FORCE)
    if rough.empty:
        print('No roughness obs — stopping.')
        return

    events = stage_detect(tag, rough, state, FORCE)

    if MAKE_PLOTS:
        try:
            stage_plots(tag)
        except Exception as e:
            print(f'  plots failed: {type(e).__name__}: {e}')

    print(f'\n=== Pipeline complete in {time.perf_counter()-t_start:.1f}s ===')
    print(f'  state   : {state_path(tag).relative_to(c.PROJECT_DIR)}')
    print(f'  events  : {events_path(tag).relative_to(c.PROJECT_DIR)}  ({len(events)})')
    if not events.empty:
        counts = events['trigger'].value_counts().to_dict()
        print('  by trigger: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))
        show = [col for col in ('t_peak_utc', 'trigger', 'duration_sec',
                                'peak_rough_ratio', 'max_sats', 'peak_tide_dev_m',
                                'confidence') if col in events.columns]
        print('\n  Top events by confidence:')
        print(events[show].head(8).round(2).to_string(index=False))


if __name__ == '__main__':
    run()
