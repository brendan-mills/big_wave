"""Download high-rate (1 Hz) RINEX for UMNQ from EarthScope, compressing as we go.

Why: UMNQ's standard daily files are 15 s — too coarse to resolve the ~30 s
near-field calving waves we want to detect (15 s sampling is the Nyquist limit
for a 30 s period). 1 Hz data, if archived, resolves them.

Disk: a single day of 1 Hz multi-GNSS RINEX 2 is ~534 MB raw. We therefore work
ONE DAY AT A TIME — download the plain `.YYo`, Hatanaka-compress + gzip it to
`.YYd.gz` (~47 MB, 11x smaller; matches the project's existing `.d.Z` daily
files and is read natively by gnssrefl), then delete the raw file before moving
on. A whole-range download would pile up 60 x 534 MB (~32 GB) of raw files at
once. Already-compressed days are skipped, so the job is resumable.

Auth: EarthScope SSO. gnssrefl reads the token from
`$REFL_CODE/sso_tokens.json` via `earthscope_sdk`. If the token is missing or
expired, gnssrefl will prompt a device-login the first time.

Edit the RUN CONFIG constants, then click Run. (No argparse — VS Code Run.)
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

import config as c

# gnssrefl reads these at import time
os.environ.setdefault('REFL_CODE', str(c.REFL_CODE))
os.environ.setdefault('ORBITS',    str(c.ORBITS_DIR))
os.environ.setdefault('EXE',       str(c.EXE_DIR))

# VS Code's Run button launches the env's python WITHOUT activating the env, so
# its PATH lacks the conda bin. gnssrefl's gnet path tests for lftp via
# `subprocess.call(['which','lftp'])`, which then fails even though lftp IS
# installed. Prepend the interpreter's own bin dir so child processes find it.
_envbin = os.path.dirname(sys.executable)
if _envbin not in os.environ.get('PATH', '').split(os.pathsep):
    os.environ['PATH'] = _envbin + os.pathsep + os.environ.get('PATH', '')

import hatanaka                                              # noqa: E402
from gnssrefl.download_rinex import download_rinex          # noqa: E402


# =============================================================================
# RUN CONFIG — edit then Run
# =============================================================================

STATION    = c.STATION         # 'umnq' — 4-char ID, used for RINEX 2
STATION9   = 'UMNQ'          # 9-char long name, REQUIRED for RINEX 3 downloads
                                # (ID + '00' monument/receiver + 'GRL' country)
# (year, doy) inclusive bounds — same scheme as main.py. Spans years and leap
# years (e.g. (2024, 350) -> (2025, 030) downloads doy 350-366 of 2024 + 1-30
# of 2025). Files are named by 2-digit year so different years don't collide.
START_DATE = (2023, 200)       # 2023-07-19, earliest 1 Hz at UMNQ
END_DATE   = (2026, 149)       # 2026-05-29 (present). Resumable: already-have days
                                # skip, so this just fills forward from the last gap.

RATE       = 'high'            # 'high' = 1 Hz archive folder ('low' = daily 15 s)
SAMPLERATE = 1                 # native interval (s) of the high-rate file
ARCHIVE    = 'unavco'          # EarthScope (UMNQ = Virginia Tech/EarthScope).
                                # If empty, try 'gnet' (Greenland Network).
VERSION    = 2                 # high-rate is usually RINEX 3; v2 confirmed for UMNQ

# Probe availability with ONE day (START_DATE) before committing to the range.
# Confirmed: doy 250 of 2025 downloaded as a 534 MB RINEX 2 .25o, so set False.
PROBE_ONLY = False

# Hatanaka-compress + gzip each day to .YYd.gz (~11x) and delete the raw .YYo.
COMPRESS   = True
# Re-download/re-compress days whose .YYd.gz already exists. Leave False to
# resume an interrupted run (existing compressed days are skipped).
FORCE      = False

# gnssrefl writes downloaded RINEX into the current working directory, so we
# chdir into a dedicated high-rate folder to keep it separate from the 15 s set.
DEST_DIR   = c.DATA_DIR / 'rinex_highrate' / STATION


# =============================================================================
# Driver
# =============================================================================

def date_range(start: tuple[int, int], end: tuple[int, int]):
    """Yield (year, doy) for every day in [start, end] inclusive, spanning year
    and leap-year boundaries. start/end are (year, doy) like main.py."""
    (y0, d0), (y1, d1) = start, end
    cur  = dt.date(y0, 1, 1) + dt.timedelta(days=d0 - 1)
    last = dt.date(y1, 1, 1) + dt.timedelta(days=d1 - 1)
    while cur <= last:
        yield cur.year, cur.timetuple().tm_yday
        cur += dt.timedelta(days=1)


def raw_path(year: int, doy: int) -> Path:
    """download_rinex writes RINEX 2 daily obs as ssssDDD0.YYo in the CWD."""
    return DEST_DIR / f'{STATION}{doy:03d}0.{year % 100:02d}o'


def compressed_path(year: int, doy: int) -> Path:
    """Hatanaka + gzip target: ssssDDD0.YYd.gz (matches the project's .d.Z set)."""
    return DEST_DIR / f'{STATION}{doy:03d}0.{year % 100:02d}d.gz'


# RINEX-2 epoch header: ' YY MM DD HH MM SS.sssssss  F NN<sat list>'. The
# high-rate files routinely arrive with an incomplete FINAL epoch (the archive
# concatenates sub-daily chunks and the last record is cut mid-line), which
# rnx2crx rejects as "truncated in the middle". Matching this lets us drop just
# that last partial epoch (≤1 s of data) so the rest compresses.
_EPOCH_HDR = re.compile(
    rb'^ [ \d]\d [ \d]\d [ \d]\d [ \d]\d [ \d]\d [ \d]\d\.\d{7}  ?\d{1,2} ', re.M)


def _drop_truncated_epoch(raw: Path) -> bool:
    """Truncate `raw` to the end of its last complete epoch. True if it changed."""
    size = raw.stat().st_size
    with open(raw, 'rb') as f:
        f.seek(max(0, size - 4_000_000))          # one epoch is ~15 KB; 4 MB is ample
        base = f.tell()
        tail = f.read()
    hdrs = list(_EPOCH_HDR.finditer(tail))
    if not hdrs:
        return False
    cut = base + hdrs[-1].start()                 # start of the last (partial) epoch
    if cut >= size:
        return False
    os.truncate(raw, cut)
    return True


def compress_raw(raw: Path) -> tuple[str, Path | None]:
    """Hatanaka+gzip `raw` to .YYd.gz, deleting it. Repairs a truncated tail once."""
    raw_mb = raw.stat().st_size / 1e6
    # skip_strange_epochs survives malformed interior epochs; delete=True drops
    # the bulky .YYo once the .YYd.gz is written.
    try:
        out = hatanaka.compress_on_disk(raw, delete=True, skip_strange_epochs=True)
    except hatanaka.HatanakaException:
        if not _drop_truncated_epoch(raw):
            return 'compress FAILED (no epoch header found); kept raw', raw
        try:
            out = hatanaka.compress_on_disk(raw, delete=True, skip_strange_epochs=True)
        except Exception as e:
            return f'compress FAILED after repair ({type(e).__name__}: {e}); kept raw', raw
        return f'{raw_mb:.0f} MB -> {out.stat().st_size/1e6:.1f} MB (repaired tail)', out
    except Exception as e:
        return f'compress FAILED ({type(e).__name__}: {e}); kept raw', raw
    return f'{raw_mb:.0f} MB -> {out.stat().st_size/1e6:.1f} MB', out


def fetch_day(year: int, doy: int) -> tuple[str, Path | None]:
    """Ensure day is compressed. Reuse an existing raw; else download. Returns status."""
    out = compressed_path(year, doy)
    if out.exists() and not FORCE:
        return 'skip (already compressed)', out

    raw = raw_path(year, doy)
    # A leftover raw (from an earlier failed compress) — compress it, don't refetch.
    if raw.exists() and not FORCE:
        return compress_raw(raw) if COMPRESS else (
            f'raw {raw.stat().st_size/1e6:.0f} MB (no compress)', raw)

    sta = STATION9 if VERSION >= 3 else STATION   # RINEX 3 needs the 9-char name
    raw.unlink(missing_ok=True)                   # clear any partial leftover

    cwd0 = Path.cwd()
    os.chdir(DEST_DIR)
    try:
        download_rinex(
            sta, year, doy, 0,                    # (station, year, doy, 0)
            rate=RATE, archive=ARCHIVE, samplerate=SAMPLERATE, version=VERSION,
        )
    except SystemExit as e:
        print(f'    download_rinex exited: {e}')
    except Exception as e:
        print(f'    download_rinex error: {type(e).__name__}: {e}')
    finally:
        os.chdir(cwd0)

    if not raw.exists():
        return 'no file (not archived?)', None
    if not COMPRESS:
        return f'raw {raw.stat().st_size/1e6:.0f} MB (no compress)', raw
    return compress_raw(raw)


if __name__ == '__main__':
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    days = [START_DATE] if PROBE_ONLY else list(date_range(START_DATE, END_DATE))
    mode = 'PROBE (1 day)' if PROBE_ONLY else f'RANGE {START_DATE}..{END_DATE}'
    print(f'{mode}: {STATION}  ({len(days)} day(s), rate={RATE}, {SAMPLERATE}s, '
          f'archive={ARCHIVE}, v{VERSION}, compress={COMPRESS})')
    print(f'  dest: {DEST_DIR}\n')

    ok, missing = [], []
    for year, doy in days:
        print(f'{year} doy {doy:03d}...')
        status, path = fetch_day(year, doy)
        print(f'  {status}')
        (ok if path else missing).append((year, doy))

    print(f'\nDone: {len(ok)}/{len(days)} day(s) present, {len(missing)} missing.')
    if missing:
        print(f'  missing: {missing}')
        print('  Things to try for missing days:')
        print('    - archive="gnet"  (Greenland Network instead of EarthScope)')
        print('    - confirm 1 Hz exists (some stations only archive bursts)')
        print('    - check SSO token at $REFL_CODE/sso_tokens.json (may need re-login)')
