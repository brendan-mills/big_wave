"""Batch driver for the SSiSLS SNR -> per-arc-RH pipeline.

Layer 2 only: scans an input folder of RINEX files, runs `snr.process_arcs`
on each day, writes one parquet per day to `config.RESULTS_DIR`. The Kalman
filter / tide residual / rogue-wave detection are downstream modules that
consume these parquets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

import config as c
import snr


RINEX_PATTERN = re.compile(r'^([a-z0-9]{4})(\d{3})\d\.(\d{2})d')


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def output_path(year: int, doy: int) -> Path:
    """Per-day parquet: results/{year}/{doy:03d}.parquet."""
    return c.RESULTS_DIR / f'{year}' / f'{doy:03d}.parquet'


def runs_dir(year: int) -> Path:
    return c.RESULTS_DIR / f'{year}' / '_runs'


# ---------------------------------------------------------------------------
# RINEX discovery
# ---------------------------------------------------------------------------

def parse_rinex_filename(path: Path) -> tuple[str, int, int] | None:
    """Parse station/year/doy from a RINEX 2 obs filename like 'umnq0010.26d.Z'."""
    m = RINEX_PATTERN.match(path.name.lower())
    if not m:
        return None
    station = m.group(1)
    doy = int(m.group(2))
    yy = int(m.group(3))
    year = 2000 + yy if yy < 80 else 1900 + yy  # standard RINEX 2 yy rollover
    return station, year, doy


def discover_rinex(folder: Path, station: str = c.STATION
                   ) -> list[tuple[int, int, Path]]:
    """Return sorted list of (year, doy, path) for matching files in folder."""
    found = []
    for p in folder.iterdir():
        parsed = parse_rinex_filename(p)
        if parsed is None:
            continue
        s, year, doy = parsed
        if s != station:
            continue
        found.append((year, doy, p))
    return sorted(found)


# ---------------------------------------------------------------------------
# Per-day and folder-level processing
# ---------------------------------------------------------------------------

def process_day(year: int, doy: int, *, force: bool = False) -> pd.DataFrame:
    """Run snr.process_arcs for one day and write to parquet. Returns the
    DataFrame. If the parquet already exists and `force=False`, reads and
    returns it without recomputing.
    """
    out = output_path(year, doy)
    if out.exists() and not force:
        return pd.read_parquet(out)

    snr_df = snr.load_snr(year, doy)
    arcs = snr.process_arcs(snr_df)

    out.parent.mkdir(parents=True, exist_ok=True)
    arcs.to_parquet(out, compression='snappy', index=False)
    return arcs


def process_folder(folder: Path, *, force: bool = False, fail_fast: bool = False,
                   doys: set[int] | None = None) -> pd.DataFrame:
    """Process every RINEX day in `folder`. One-line status per day.
    Returns concatenated DataFrame of all arcs across days.
    """
    folder = Path(folder)
    discovered = discover_rinex(folder)
    if doys is not None:
        discovered = [(y, d, p) for (y, d, p) in discovered if d in doys]

    if not discovered:
        print(f'No RINEX files matching station {c.STATION} found in {folder}')
        return pd.DataFrame()

    print(f'Processing {len(discovered)} day(s) from {folder}')
    frames, processed, skipped = [], [], []
    t_total = time.perf_counter()

    for year, doy, _ in discovered:
        t0 = time.perf_counter()
        try:
            df = process_day(year, doy, force=force)
        except Exception as e:
            print(f'  doy {doy:03d}: ERROR  {type(e).__name__}: {e}')
            skipped.append((doy, str(e)))
            if fail_fast:
                raise
            continue
        elapsed = time.perf_counter() - t0

        breakdown = (df.constellation.value_counts().to_dict() if len(df) else {})
        summary = ' / '.join(f'{k} {v}' for k, v in breakdown.items()) or 'no arcs'
        cached = '   (cached)' if not force and elapsed < 0.05 else ''
        print(f'  doy {doy:03d}: {len(df):>3d} arcs ({summary}){cached}  '
              f'{elapsed:5.2f}s')

        df = df.assign(year=year, doy=doy)
        frames.append(df)
        processed.append(doy)

    total = time.perf_counter() - t_total
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f'Done: {len(processed)}/{len(discovered)} days, '
          f'{len(out)} arcs total, {total:.1f}s')
    if skipped:
        print(f'Skipped {len(skipped)} day(s): {[d for d, _ in skipped]}')

    if processed:
        prov = write_provenance(min(y for y, _, _ in discovered),
                                 folder, processed, skipped)
        print(f'Provenance: {prov.relative_to(c.PROJECT_DIR)}')
    return out


# ---------------------------------------------------------------------------
# Loading saved results
# ---------------------------------------------------------------------------

def load_results(year: int, doys: list[int] | None = None) -> pd.DataFrame:
    """Read back per-day parquets. If `doys` is None, loads everything for
    that year in sorted order."""
    yr_dir = c.RESULTS_DIR / f'{year}'
    if not yr_dir.exists():
        return pd.DataFrame()
    files = sorted(yr_dir.glob('*.parquet'))
    if doys is not None:
        wanted = {f'{d:03d}.parquet' for d in doys}
        files = [f for f in files if f.name in wanted]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _gnssrefl_version() -> str:
    try:
        return metadata.version('gnssrefl')
    except metadata.PackageNotFoundError:
        return 'unknown'


def write_provenance(year: int, folder: Path, processed: list[int],
                     skipped: list[tuple[int, str]] | None = None) -> Path:
    """Snapshot config + version info for this run. Timestamped JSON sidecar."""
    runs_dir(year).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
    out = runs_dir(year) / f'run_{stamp}.json'

    snapshot = {k: getattr(c, k) for k in dir(c)
                if k.isupper() and not k.startswith('_')
                and isinstance(getattr(c, k), (int, float, str, tuple, list))}
    # Tuples of paths or Signals don't JSON-serialize; coerce
    snapshot = {k: (str(v) if 'PATH' in k.upper() or 'DIR' in k.upper() else v)
                for k, v in snapshot.items()}

    payload = {
        'run_at_utc':      dt.datetime.now(dt.timezone.utc).isoformat(),
        'station':         c.STATION,
        'input_folder':    str(folder),
        'doys_processed':  processed,
        'doys_skipped':    [{'doy': d, 'error': e} for d, e in (skipped or [])],
        'enabled_signals': [s.name for s in c.ENABLED_SIGNALS],
        'config_snapshot': snapshot,
        'gnssrefl_version': _gnssrefl_version(),
        'pipeline_file':   str(Path(__file__).resolve()),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_doys(s: str) -> set[int]:
    """Accept '1,3,5' or '1-31' or '1-10,15,20-25'."""
    out = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('folder', nargs='?', default=str(c.RINEX_DIR),
                   help=f'RINEX input folder (default: {c.RINEX_DIR})')
    p.add_argument('-f', '--force', action='store_true',
                   help='Reprocess days even if parquet already exists')
    p.add_argument('--fail-fast', action='store_true',
                   help='Stop on the first day that errors')
    p.add_argument('--doys', default=None,
                   help='Restrict to these doys; e.g. "1,3,5" or "1-31"')
    args = p.parse_args(argv)

    doys = _parse_doys(args.doys) if args.doys else None
    df = process_folder(Path(args.folder), force=args.force,
                        fail_fast=args.fail_fast, doys=doys)

    if not df.empty:
        print(f'\nQuick summary across run:')
        for sig in c.ENABLED_SIGNALS:
            col = f'RH_{sig.name}'
            if col in df.columns:
                v = df[col].dropna()
                if len(v):
                    print(f'  {sig.name:8s}  n={len(v):4d}  '
                          f'median={v.median():.3f} m  std={v.std():.3f} m')


if __name__ == '__main__':
    main()
