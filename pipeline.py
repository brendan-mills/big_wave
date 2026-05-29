"""Batch driver for the SSiSLS roughness stage.

Discovers RINEX in a folder and runs `snr.process_arcs_roughness` per day
(parallel across days), caching one parquet per day under
`config.RESULTS_DIR/{year}/roughness/`. Driven from `main.py`. invsnr (the
water-level reference) reads snr66 directly and is driven by `invsnr_runner`.
"""

from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import re
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

import config as c
import snr


RINEX_PATTERN = re.compile(r'^([a-z0-9]{4})(\d{3})\d\.(\d{2})d')


# ---------------------------------------------------------------------------
# RINEX discovery
# ---------------------------------------------------------------------------

def runs_dir(year: int) -> Path:
    return c.RESULTS_DIR / f'{year}' / '_runs'


def parse_rinex_filename(path: Path) -> tuple[str, int, int] | None:
    """Parse station/year/doy from a RINEX 2 obs filename like 'umnq1100.25d.gz'."""
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
        if s == station:
            found.append((year, doy, p))
    return sorted(found)


# ---------------------------------------------------------------------------
# Day-level parallelism
#
# Each day is independent (loads its own snr66, writes its own parquet), so the
# roughness stage fans days across `config.N_WORKERS` processes. Workers write
# their per-day cache and return only a lightweight status; the parent
# concatenates by re-reading the caches (keeps DataFrames out of the Pool's
# pickling path). Workers must be module-level for the 'spawn' start method.
# ---------------------------------------------------------------------------

def _map_days(task_fn, discovered, force, *, label, fail_fast, quiet=False,
              workers=None):
    """Run `task_fn` over discovered (year, doy, path) days — serial if the
    worker count is 1, else across a process Pool — printing per-day status as
    each finishes. Returns the list of (year, doy, n, err) results. `quiet`
    suppresses the per-day success line (errors still print) — used where `n` is
    a bare status flag rather than a meaningful count (e.g. the snr66 stage).
    `workers` overrides the default (c.N_WORKERS) — the snr66 stage passes the
    smaller c.SNR_WORKERS since each of its workers is much heavier on RAM."""
    tasks = [(y, d, force) for (y, d, _) in discovered]
    nw = max(1, min(workers if workers is not None else c.N_WORKERS, len(tasks)))
    print(f'{label}: {len(tasks)} day(s) '
          f'({"serial" if nw == 1 else f"{nw} workers"})')

    def _handle(r):
        _, d, n, err = r
        if err:
            print(f'  doy {d:03d}: ERROR  {err}')
        elif not quiet:
            print(f'  doy {d:03d}: {n:>6d}')
        if err and fail_fast:
            raise RuntimeError(f'doy {d}: {err}')

    results = []
    if nw == 1:
        for t in tasks:
            r = task_fn(t); results.append(r); _handle(r)
    else:
        # maxtasksperchild=1: recycle each worker after one day so gnssrefl's
        # per-day memory (a 1 Hz RINEX + SP3 orbit can be several GB resident) is
        # released back to the OS instead of accumulating over the run.
        with mp.Pool(nw, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(task_fn, tasks):
                results.append(r); _handle(r)
    return results


# ---------------------------------------------------------------------------
# snr66 creation — both invsnr and roughness READ snr66; nothing else makes it,
# so this must run before either stage. Idempotent (skips days already present).
# ---------------------------------------------------------------------------

def _ensure_snr_task(task: tuple[int, int, bool]) -> tuple[int, int, int, str | None]:
    """Worker: create one day's snr66 from RINEX if missing (idempotent). The
    `n` field reports whether work was done: 1 = freshly created, 0 = already
    cached (so the parent can summarize made vs. cached)."""
    year, doy, _force = task
    try:
        existed = snr.snr_path(year, doy).exists()
        snr.ensure_snr(year, doy)        # rinex2snr -> snr66; returns fast if cached
        return (year, doy, 0 if existed else 1, None)
    except Exception as e:               # noqa: BLE001 — report, don't crash pool
        return (year, doy, 0, f'{type(e).__name__}: {e}')


def ensure_snr_folder(folder: Path,
                      date_filter: set[tuple[int, int]] | None = None) -> None:
    """Create snr66 for every RINEX day in `folder` (parallel) if not present.
    Run before invsnr/roughness so they have snr66 to read."""
    folder = Path(folder)
    discovered = discover_rinex(folder)
    if date_filter is not None:
        discovered = [(y, d, p) for (y, d, p) in discovered if (y, d) in date_filter]
    if not discovered:
        print(f'No RINEX files matching station {c.STATION} found in {folder}')
        return
    results = _map_days(_ensure_snr_task, discovered, False,
                        label='Ensuring snr66', fail_fast=False, quiet=True,
                        workers=c.SNR_WORKERS)
    bad = [(d, e) for (_, d, _, e) in results if e]
    made = sum(n for (_, _, n, e) in results if not e)
    cached = len(results) - len(bad) - made
    print(f'snr66 ready: {len(results) - len(bad)}/{len(results)} '
          f'({made} created, {cached} cached)')
    if bad:
        print(f'  failed: {[d for d, _ in bad]}')


# ---------------------------------------------------------------------------
# Roughness stage
# ---------------------------------------------------------------------------

def roughness_output_path(year: int, doy: int) -> Path:
    """Per-day roughness parquet: results/{year}/roughness/{doy:03d}_obs.parquet."""
    return c.RESULTS_DIR / f'{year}' / 'roughness' / f'{doy:03d}_obs.parquet'


def process_day_roughness(year: int, doy: int, *, force: bool = False) -> pd.DataFrame:
    """Run snr.process_arcs_roughness for one day and cache. Idempotent."""
    out = roughness_output_path(year, doy)
    if out.exists() and not force:
        return pd.read_parquet(out)
    obs = snr.process_arcs_roughness(snr.load_snr(year, doy))
    out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(out, compression='snappy', index=False)
    return obs


def _roughness_task(task: tuple[int, int, bool]) -> tuple[int, int, int, str | None]:
    """Worker: compute + cache one day's roughness obs."""
    year, doy, force = task
    try:
        return (year, doy, len(process_day_roughness(year, doy, force=force)), None)
    except Exception as e:                       # noqa: BLE001 — report, don't crash pool
        return (year, doy, 0, f'{type(e).__name__}: {e}')


def process_folder_roughness(folder: Path, *, force: bool = False,
                             fail_fast: bool = False,
                             date_filter: set[tuple[int, int]] | None = None
                             ) -> pd.DataFrame:
    """Run process_day_roughness across every RINEX day in `folder`, in parallel.
    `date_filter` is a set of (year, doy) tuples; None = all discovered.
    Returns concatenated long-form roughness DataFrame."""
    folder = Path(folder)
    discovered = discover_rinex(folder)
    if date_filter is not None:
        discovered = [(y, d, p) for (y, d, p) in discovered if (y, d) in date_filter]
    if not discovered:
        print(f'No RINEX files matching station {c.STATION} found in {folder}')
        return pd.DataFrame()

    t_total = time.perf_counter()
    results = _map_days(_roughness_task, discovered, force,
                        label='Roughness processing', fail_fast=fail_fast)
    processed = [d for (_, d, _, e) in results if not e]
    skipped = [(d, e) for (_, d, _, e) in results if e]

    frames = [df for (y, d, _) in discovered
              if (cache := roughness_output_path(y, d)).exists()
              and len(df := pd.read_parquet(cache))]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f'Done: {len(frames)} days, {len(out):,} roughness obs, '
          f'{time.perf_counter()-t_total:.1f}s')
    if skipped:
        print(f'  {len(skipped)} day(s) errored: {[d for d, _ in skipped]}')
    if processed:
        prov = write_provenance(min(y for y, _, _ in discovered),
                                folder, processed, skipped)
        print(f'  provenance: {prov.relative_to(c.PROJECT_DIR)}')
    return out


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
    snapshot = {k: (str(v) if 'PATH' in k.upper() or 'DIR' in k.upper() else v)
                for k, v in snapshot.items()}

    payload = {
        'run_at_utc':       dt.datetime.now(dt.timezone.utc).isoformat(),
        'station':          c.STATION,
        'input_folder':     str(folder),
        'doys_processed':   processed,
        'doys_skipped':     [{'doy': d, 'error': e} for d, e in (skipped or [])],
        'enabled_signals':  [s.name for s in c.ENABLED_SIGNALS],
        'config_snapshot':  snapshot,
        'gnssrefl_version': _gnssrefl_version(),
        'pipeline_file':    str(Path(__file__).resolve()),
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out
