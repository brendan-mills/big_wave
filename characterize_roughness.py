"""Characterize the SNR-roughness obs to set detect.RoughnessConfig.

The roughness run (main.py with USE_ROUGHNESS) writes per-day roughness caches
(results/{year}/roughness/{doy}_obs.parquet) with a per-(window, sat, signal)
`rough_ratio` (roughness / per-arc calm baseline). The wave-train detector
(detect.detect_roughness_bursts) flags windows with rough_ratio >=
`rough_ratio_min`, requires >= `min_sats` distinct sats coherently rough in a
`bin_sec` bin, and clusters bins into events. Those three knobs were
placeholders; this script reads the actual roughness distribution and reports
what to set them to.

Sections:
  A. obs inventory (constellation/signal balance — should be balanced, unlike
     the windowed-RH path)
  B. rough_ratio distribution (overall + per constellation) -> rough_ratio_min
  C. multi-sat coherence: distinct rough sats per time bin -> min_sats / bin_sec
  D. event-count sensitivity grid over (rough_ratio_min, min_sats) using the
     REAL detector -> pick an operating point
  E. temporal distribution (clustered around storms, or a uniform noise floor?)

Read-only. Edit the RUN CONFIG constants, then click Run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as c
import detect


# =============================================================================
# RUN CONFIG — edit then Run
# =============================================================================

DOY_SUBSET   = None                  # None = all days with a roughness cache
RATIO_GRID   = (2.5, 3.0, 4.0, 5.0, 6.0, 8.0)   # rough_ratio_min candidates
SATS_GRID    = (3, 4, 5)             # min_sats candidates
BIN_SEC      = 30.0                  # bin for the coherence + event grid (s)
REF_RATIO    = 3.0                   # ratio used for the section-C coherence look
MAKE_PLOTS   = True


# =============================================================================
# Loading
# =============================================================================

_COLS = ['t_center_utc', 'sat', 'constellation', 'signal', 'rough_ratio']


def load_roughness() -> pd.DataFrame:
    """Concatenate per-day roughness caches (only the columns we need)."""
    frames = []
    for f in sorted(c.RESULTS_DIR.glob('*/roughness/*_obs.parquet')):
        doy = int(f.stem.split('_')[0])
        if DOY_SUBSET is not None and doy not in DOY_SUBSET:
            continue
        frames.append(pd.read_parquet(f, columns=_COLS))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


def _q(a, qs=(50, 90, 95, 99, 99.9)) -> str:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return 'no data'
    return '  '.join(f'p{p}={np.percentile(a, p):.2f}' for p in qs)


# =============================================================================
# Report
# =============================================================================

def main() -> None:
    rough = load_roughness()
    if rough.empty:
        print(f'No roughness caches under {c.RESULTS_DIR}/*/roughness/. '
              'Run main.py with USE_ROUGHNESS=True first.')
        return
    n = len(rough)
    days = rough['t_center_utc'].dt.floor('D').nunique()
    print(f'Roughness obs: {n:,}  over {days} day(s)\n')

    # --- A. inventory -----------------------------------------------------
    print('=' * 70)
    print('A. OBS INVENTORY  (should be reasonably balanced across GNSS)')
    print('=' * 70)
    for k, v in rough['constellation'].value_counts().items():
        print(f'    {k:8s} {v:>10,d}  ({100*v/n:4.1f}%)')
    print(f'  distinct satellites: {rough.sat.nunique()}')

    # --- B. rough_ratio distribution --------------------------------------
    print('\n' + '=' * 70)
    print('B. ROUGH_RATIO DISTRIBUTION  (-> where to set rough_ratio_min)')
    print('=' * 70)
    print(f'  ALL        : {_q(rough.rough_ratio)}')
    for con in ('GPS', 'Galileo', 'GLONASS'):
        sub = rough[rough.constellation == con]
        if len(sub):
            print(f'  {con:10s} : {_q(sub.rough_ratio)}')
    print('  (baseline is the per-arc median, so p50~1.0 by construction; the')
    print('   tail p99/p99.9 is the "anomalous" level a threshold should clear.)')

    # --- C. multi-sat coherence -------------------------------------------
    print('\n' + '=' * 70)
    print(f'C. MULTI-SAT COHERENCE  at rough_ratio >= {REF_RATIO}, bin={BIN_SEC:.0f}s')
    print('   (-> min_sats: how many sats genuinely coincide vs by chance)')
    print('=' * 70)
    flagged = rough[rough.rough_ratio >= REF_RATIO].copy()
    flagged['tbin'] = flagged['t_center_utc'].dt.floor(f'{int(BIN_SEC)}s')
    per_bin = flagged.groupby('tbin').agg(n_sats=('sat', 'nunique'),
                                          n_con=('constellation', 'nunique'))
    print(f'  flagged windows: {len(flagged):,}  ->  {len(per_bin):,} bins')
    print(f'  distinct rough sats / bin: {_q(per_bin.n_sats, (50,90,95,99,99.9))}')
    for s in SATS_GRID:
        print(f'    bins with >= {s} sats: {int((per_bin.n_sats >= s).sum()):>6,d}')

    # --- D. event-count sensitivity grid ----------------------------------
    print('\n' + '=' * 70)
    print(f'D. EVENT COUNT vs (rough_ratio_min, min_sats)  [bin={BIN_SEC:.0f}s]')
    print('   (uses the REAL detector; pick a row/col with a sane event rate)')
    print('=' * 70)
    header = '  ratio \\ sats |' + ''.join(f'{s:>10d}' for s in SATS_GRID)
    print(header); print('  ' + '-' * (len(header) - 2))
    grid = {}
    for r in RATIO_GRID:
        cells = []
        for s in SATS_GRID:
            cfg = detect.RoughnessConfig(rough_ratio_min=r, min_sats=s, bin_sec=BIN_SEC)
            ne = len(detect.detect_roughness_bursts(rough, cfg))
            grid[(r, s)] = ne
            cells.append(ne)
        print(f'  {r:>11.1f} |' + ''.join(f'{x:>10d}' for x in cells)
              + f'   ({cells[0]/days:.1f}/day at sats={SATS_GRID[0]})')

    # --- E. temporal distribution -----------------------------------------
    print('\n' + '=' * 70)
    print('E. TEMPORAL SPREAD  (clustered = plausibly real; uniform = noise floor)')
    print('=' * 70)
    # use a mid operating point for the look
    r_mid = RATIO_GRID[len(RATIO_GRID) // 2]
    cfg = detect.RoughnessConfig(rough_ratio_min=r_mid, min_sats=SATS_GRID[0],
                                 bin_sec=BIN_SEC)
    ev = detect.detect_roughness_bursts(rough, cfg)
    if len(ev):
        per_day = ev.groupby(ev.t_peak_utc.dt.floor('D')).size()
        print(f'  at ratio>={r_mid}, sats>={SATS_GRID[0]}: {len(ev)} events over '
              f'{days} days')
        print(f'  events/day: {_q(per_day.values, (50,90,99))}  max={per_day.max()}')
        print(f'  days with 0 events: {days - len(per_day)} / {days}')
        top = ev.nlargest(8, 'confidence')[
            ['t_peak_utc', 'duration_sec', 'peak_rough_ratio', 'max_sats',
             'max_constellations', 'confidence']].copy()
        for col in ('duration_sec', 'peak_rough_ratio', 'confidence'):
            top[col] = top[col].round(2)
        print('\n  strongest events at that setting:')
        print(top.to_string(index=False))

    # --- recommendations --------------------------------------------------
    print('\n' + '=' * 70)
    print('SUGGESTED detect.RoughnessConfig')
    print('=' * 70)
    # aim for a "reviewable" rate (~<=1/day): pick the loosest grid cell under it
    target = days
    pick = None
    for r in RATIO_GRID:
        for s in SATS_GRID:
            if grid[(r, s)] <= target:
                pick = (r, s, grid[(r, s)]); break
        if pick:
            break
    if pick:
        r, s, ne = pick
        print(f'  rough_ratio_min = {r}      # {ne} events over {days} days '
              f'(~{ne/days:.2f}/day)')
        print(f'  min_sats        = {s}')
        print(f'  bin_sec         = {BIN_SEC:.0f}')
    else:
        print('  even the strictest grid cell exceeds ~1/day — widen RATIO_GRID up.')
    print('\n  NOTE: with no ground truth these are an OPERATING POINT, not a')
    print('  validated detection threshold. Prefer a rate you can manually review,')
    print('  then check the strongest events against weather/tide-gauge records.')

    if MAKE_PLOTS:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            outdir = c.PLOTS_DIR / 'rough_char'
            outdir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            for con in ('GPS', 'Galileo', 'GLONASS'):
                sub = rough[rough.constellation == con]['rough_ratio']
                if len(sub):
                    ax.hist(sub.clip(0, 10), bins=120, histtype='step',
                            density=True, log=True, label=f'{con} (n={len(sub):,})')
            for r in RATIO_GRID:
                ax.axvline(r, color='grey', lw=0.5, ls=':')
            ax.set_xlabel('rough_ratio'); ax.set_ylabel('density (log)')
            ax.set_title('Roughness ratio distribution by constellation')
            ax.legend(); fig.tight_layout()
            fig.savefig(outdir / 'rough_ratio_dist.png', dpi=c.DPI); plt.close(fig)
            print(f'\nPlot written to {outdir}')
        except Exception as e:
            print(f'\n(plot skipped: {type(e).__name__}: {e})')


if __name__ == '__main__':
    main()
