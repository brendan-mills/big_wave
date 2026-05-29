"""Single-station configuration for SSiSLS GNSS-IR processing.

All scripts in this project import constants from here — change a value once,
re-run the pipeline, everything downstream picks it up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Station
# ---------------------------------------------------------------------------
STATION = 'umnq'                       # 4-char gnssrefl/RINEX station ID
LAT, LON = 70.677526, -52.115415       # WGS-84 deg
ANT_HEIGHT_ELL = 38.00                 # antenna ellipsoidal height, m
ANTENNA_MSL_M = 8.88                  # nominal antenna height above MSL, m
                                       # (used to convert RH -> water level)

# ---------------------------------------------------------------------------
# Data-rate switch — the single knob selecting which dataset the pipeline runs
# on. '15s' = the archived daily files (data/rinex/...); '1Hz' = the high-rate
# set fetched by download.py (data/rinex_highrate/...). Everything downstream
# reads the rate-specific paths and window knobs derived below, so flipping
# this re-roots ALL caches (snr66 + parquets live in separate trees, no
# cross-rate collisions) and swaps in the sampling-appropriate window settings.
# The two datasets overlap in time (the 15s set spans 2025085-2026145, which
# includes the 1Hz study window), so this separation is mandatory, not cosmetic.
# ---------------------------------------------------------------------------
DATA_RATE = '1Hz'                      # '15s' | '1Hz'

# ---------------------------------------------------------------------------
# Paths — all runtime/data dirs live under data/ (gitignored).
#         The source tree stays small and standalone.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / 'data'

# Shared across rates. gnssrefl reads $ORBITS and $EXE from their own env vars,
# independent of $REFL_CODE, so the SP3/nav cache and the CRX2RNX/gfzrnx binaries
# are reused by both rates. The EarthScope SSO token also lives under REFL_CODE
# (download.py uses it), so REFL_CODE itself stays put.
REFL_CODE   = DATA_DIR / 'refl_code'                        # exe, orbits, sso_tokens
EXE_DIR     = REFL_CODE / 'exe'                             # CRX2RNX, gfzrnx
ORBITS_DIR  = REFL_CODE / 'orbits'                          # SP3 / nav cache

# Rate-specific roots. SNR_REFL_CODE is where gnssrefl writes/reads snr66 files
# (it becomes $REFL_CODE during processing in snr.py); RESULTS_DIR holds the
# per-day + range parquet caches. For '15s' these stay at the original
# locations so existing caches keep working untouched.
if DATA_RATE == '1Hz':
    RINEX_DIR     = DATA_DIR / 'rinex_highrate' / STATION   # .25d.gz from download.py
    SNR_REFL_CODE = DATA_DIR / 'refl_code_1hz'              # rate-specific snr66 tree
    RESULTS_DIR   = DATA_DIR / 'results_1hz'                # rate-specific parquet caches
else:
    RINEX_DIR     = DATA_DIR / 'rinex' / 'umnq-2025085-2026145'   # raw .d.Z files
    SNR_REFL_CODE = REFL_CODE                                # existing snr66 tree
    RESULTS_DIR   = DATA_DIR / 'results'                     # per-day RH parquet
PLOTS_DIR   = RESULTS_DIR / 'plots'                         # cross-day plots

# Tide model lives under data/ (gitignored). Collaborators download the
# Gr1kmTM files once from ESR and drop them in data/Gr1kTM/.
TIDE_MODEL_DIR = DATA_DIR / 'Gr1kTM'

# ---------------------------------------------------------------------------
# rinex2snr
# ---------------------------------------------------------------------------
ORB = 'gnss3'                          # multi-GNSS SP3 product that works for 2026
SNR_TYPE = 66                          # elevation mask 5-30 deg
NOLOOK = True                          # skip remote archive — local files only
RINEX_DEC = 1 if DATA_RATE == '1Hz' else 0   # rinex2snr decimation target (s).
                                       # 1 keeps the FULL 1 Hz stream (no-op on a
                                       # 1 Hz file); 0 = no decimation for the 15 s
                                       # files. Bump to 2-3 later to trade wave
                                       # resolution for ~N x less snr66/LS compute.
INVSNR_TIMEOUT_SEC = 900               # per-chunk invsnr wall-clock cap. A normal
                                       # 6+2-day chunk fit takes ~270-300 s, so 300
                                       # was TOO TIGHT — chunks timed out and fell
                                       # back to per-day fits, whose unconstrained
                                       # day-edge splines overshoot -> midnight seams
                                       # in the water level. 900 lets chunks finish
                                       # (seam-free, continuous spline) while still
                                       # catching a true hang (the doy-164 hang ran
                                       # >1500 s). Run each chunk in a subprocess.
INVSNR_DEC = 15 if DATA_RATE == '1Hz' else 1   # invsnr's OWN SNR decimation (s),
                                       # independent of the snr66 rate. invsnr fits
                                       # a tide-scale B-spline (knot_space ~3 h) and
                                       # reads the snr66 directly; it gains nothing
                                       # from 1 Hz but chokes on 1.1 M rows. 15 gives
                                       # invsnr the same effective sampling as the
                                       # validated 15 s pipeline (~25 s/day). The
                                       # full 1 Hz stream still feeds the windowed
                                       # detector — these are separate consumers.

# ---------------------------------------------------------------------------
# Arc selection (azimuth wedge + elevation mask + RH search window)
# ---------------------------------------------------------------------------
AZ_MIN, AZ_MAX = 30.0, 180.0           # deg, station-specific (fjord-facing wedge)
EL_MIN, EL_MAX = 5.0, 25.0             # deg, elevation band for Lomb-Scargle
RH_MIN, RH_MAX = 4.0, 16.0             # m. Tide range at UMNQ is ±~1 m (2.27 m
                                       # peak-to-peak measured for Jan 2026), antenna
                                       # at 8.88 m MSL → physical RH ≈ 7.5–9.7 m.
                                       # RH_MAX=11 covers up to ~2 m below MSL (deeper
                                       # than any plausible low tide + surge). Tighter
                                       # than 12 to stop LS from clamping spurious
                                       # peaks above 11 to the grid ceiling, which
                                       # produced "flat negative water level" runs.
                                       # RH_MIN=5 allows ~3 m wave crests above max
                                       # tide before bumping the lower edge.
MIN_ARC_PTS    = 20                    # drop arcs with fewer SNR points than this
GAP_SEC        = 1800                  # >30 min gap inside an arc -> split passes

# ---------------------------------------------------------------------------
# Multi-constellation signal registry
# ---------------------------------------------------------------------------
# gnssrefl writes all constellations into one snr66 file. The PRN ranges are
# their convention: GPS 1-99, GLONASS 101-199, Galileo 201-299, BeiDou 301-399,
# QZSS 401-499, IRNSS 501-599.
#
# A `Signal` describes one carrier band on one constellation: which snr66
# column its SNR sits in, the carrier frequency, and the PRN range it applies
# to. The processing loop iterates over ENABLED_SIGNALS — to add Galileo E6
# or BeiDou B2a, just append the right Signal here.

C_LIGHT = 299_792_458.0

@dataclass(frozen=True)
class Signal:
    name: str              # 'GPS_L1', 'GAL_E5a', ...
    constellation: str     # 'GPS' | 'Galileo' | 'GLONASS' | 'BeiDou'
    snr_col: str           # snr66 column name: 'S1' | 'S2' | 'S5' | 'S6' | 'S7' | 'S8'
    freq_hz: float         # nominal carrier frequency, Hz
    prn_lo: int            # gnssrefl PRN range (inclusive)
    prn_hi: int

    @property
    def wavelength_m(self) -> float:
        return C_LIGHT / self.freq_hz

# --- GPS (CDMA, fixed frequencies) ---
GPS_L1  = Signal('GPS_L1',  'GPS', 'S1', 1575.42e6,  1,  32)
GPS_L2  = Signal('GPS_L2',  'GPS', 'S2', 1227.60e6,  1,  32)
GPS_L5  = Signal('GPS_L5',  'GPS', 'S5', 1176.45e6,  1,  32)   # Block IIF/III only

# --- Galileo (CDMA) ---
GAL_E1  = Signal('GAL_E1',  'Galileo', 'S1', 1575.42e6, 201, 299)
GAL_E5a = Signal('GAL_E5a', 'Galileo', 'S5', 1176.45e6, 201, 299)
GAL_E5b = Signal('GAL_E5b', 'Galileo', 'S7', 1207.14e6, 201, 299)

# --- GLONASS (FDMA — see WARNING below) ---
# GLONASS L1/L2 use a different center frequency per satellite (frequency
# division). The values below are the NOMINAL channel-0 frequencies; using
# them introduces an RH bias of up to a few cm per satellite. For sub-cm work
# you must look up the channel number per PRN from a broadcast nav file and
# adjust each satellite's wavelength. Enable at your own risk.
GLO_G1  = Signal('GLO_G1',  'GLONASS', 'S1', 1602.0e6, 101, 199)
GLO_G2  = Signal('GLO_G2',  'GLONASS', 'S2', 1246.0e6, 101, 199)

# The signals processed by the pipeline. To enable another band/constellation,
# define its `Signal` above (BeiDou/QZSS not tracked at UMNQ) and add it here.
ENABLED_SIGNALS = (
    GPS_L1, GPS_L2, GPS_L5,
    GAL_E1, GAL_E5a, GAL_E5b,
    GLO_G1, GLO_G2,    # nominal channel-0 frequencies — see WARNING above.
                        # Per-sat FDMA bias ~3 cm at RH ~9 m; negligible for
                        # roughness (relative SNR fluctuations, not absolute RH).
)

def signals_for_sat(prn: int, signals=ENABLED_SIGNALS) -> tuple[Signal, ...]:
    """Return the signals whose PRN range covers this satellite."""
    return tuple(s for s in signals if s.prn_lo <= prn <= s.prn_hi)

def constellation_for_sat(prn: int) -> str | None:
    """Map a gnssrefl PRN to its constellation name (None if out of range)."""
    for lo, hi, name in [(1,99,'GPS'), (101,199,'GLONASS'), (201,299,'Galileo'),
                         (301,399,'BeiDou'), (401,499,'QZSS'), (501,599,'IRNSS')]:
        if lo <= prn <= hi:
            return name
    return None

# snr66 file layout — 11 columns, gnssrefl convention
SNR_COLUMNS = ['sat', 'elev', 'azim', 'sec', 'edot',
               'S6', 'S1', 'S2', 'S5', 'S7', 'S8']

# ---------------------------------------------------------------------------
# Parallelism — days are independent, so the per-day stages fan across worker
# processes. Both worker counts are RAM-bounded, not just core-bounded: at 1 Hz
# a single day is hundreds of MB to several GB resident, so naive cores-1
# concurrency exhausts RAM (a 16 GB / 10-core Mac froze mid-run at 9 workers
# while every worker held a 1 Hz RINEX + a multi-GNSS SP3 orbit). Pool workers
# are also recycled after each day (maxtasksperchild=1) so gnssrefl's per-day
# memory is returned to the OS instead of creeping up over a long run.
#
#   SNR_WORKERS — the snr66 creation stage (rinex2snr). HEAVIEST: each worker
#                 parses a full-day 1 Hz RINEX (~534 MB on disk, several GB
#                 resident) AND loads an SP3 orbit. Budget ~4 GB/worker against
#                 RAM minus OS headroom.
#   N_WORKERS   — the roughness stage. Each worker reads one day's (smaller,
#                 reduced) snr66; budget ~2 GB/worker.
# 1 = serial (no Pool — easier to debug).
# ---------------------------------------------------------------------------
_CORES  = os.cpu_count() or 4
_RAM_GB = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3
           if hasattr(os, 'sysconf') and 'SC_PHYS_PAGES' in os.sysconf_names
           else 16.0)
_RAM_HEADROOM_GB = 4.0                  # leave this much for the OS + everything else

N_WORKERS   = max(1, min(_CORES - 1, int((_RAM_GB - _RAM_HEADROOM_GB) // 2)))
SNR_WORKERS = max(1, min(_CORES - 1, int((_RAM_GB - _RAM_HEADROOM_GB) // 4)))

# ---------------------------------------------------------------------------
# SNR roughness (1 Hz wave-train detection) — the fast-wave path.
#
# Windowed RH can't resolve ~10 s waves (a window short enough to localize one
# is too short to constrain RH). Roughness sidesteps RH entirely: in a short
# window the slow RH oscillation is a smooth low-order trend, while a wave train
# injects FAST SNR fluctuations the trend can't absorb. We measure the residual
# RMS after a low-order detrend -> a per-window roughness, normalized per arc to
# a relative ratio. NEEDS full 1 Hz (RINEX_DEC=1) — decimation kills the fast
# fluctuations this keys on. (Meaningless for the 15 s set.)
# ---------------------------------------------------------------------------
ROUGH_WIN_SEC       = 20               # short window for residual RMS (s). ~2
                                       # cycles of a 10 s wave; <1 cycle of the
                                       # >=30 s RH osc, so the detrend absorbs the
                                       # osc and leaves the wave fluctuation.
ROUGH_STRIDE_SEC    = 5                # stride (s) -> ~5 s event timing
ROUGH_DETREND_ORDER = 2                # poly order vs time within the window
ROUGH_MIN_PTS       = 15               # min valid SNR samples (of 20 at 1 Hz)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
DPI = 300
