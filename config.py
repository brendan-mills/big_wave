"""Single-station configuration for SSiSLS GNSS-IR processing.

All scripts in this project import constants from here — change a value once,
re-run the pipeline, everything downstream picks it up.
"""

from __future__ import annotations

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
# Paths — all runtime/data dirs live under data/ (gitignored).
#         The source tree stays small and standalone.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / 'data'

RINEX_DIR   = DATA_DIR / 'rinex' / 'umnq-2025085-2026145'   # raw .d.Z files
REFL_CODE   = DATA_DIR / 'refl_code'                        # gnssrefl $REFL_CODE
EXE_DIR     = REFL_CODE / 'exe'                             # CRX2RNX, gfzrnx
ORBITS_DIR  = REFL_CODE / 'orbits'                          # SP3 / nav cache
RESULTS_DIR = DATA_DIR / 'results'                          # per-day RH parquet
PLOTS_DIR   = RESULTS_DIR / 'plots'                         # cross-day plots

# Tide model lives outside the project tree
TIDE_MODEL_DIR = Path('/Users/brmills/Documents/SSiSLS/Gr1kTM')

# ---------------------------------------------------------------------------
# rinex2snr
# ---------------------------------------------------------------------------
ORB = 'gnss3'                          # multi-GNSS SP3 product that works for 2026
SNR_TYPE = 66                          # elevation mask 5-30 deg
NOLOOK = True                          # skip remote archive — local files only

# ---------------------------------------------------------------------------
# Arc selection (azimuth wedge + elevation mask + RH search window)
# ---------------------------------------------------------------------------
AZ_MIN, AZ_MAX = 30.0, 180.0           # deg, station-specific (fjord-facing wedge)
EL_MIN, EL_MAX = 5.0, 25.0             # deg, elevation band for Lomb-Scargle
RH_MIN, RH_MAX = 6.0, 14.0             # m, brackets the ~10.88 m antenna MSL height
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
GAL_E6  = Signal('GAL_E6',  'Galileo', 'S6', 1278.75e6, 201, 299)

# --- BeiDou (CDMA; not tracked at UMNQ but defined for portability) ---
# Note: BDS column conventions in snr66 vary by receiver firmware. B1I may
# appear in S2 (legacy) or S1 (B1C); verify against your file before enabling.
BDS_B1I = Signal('BDS_B1I', 'BeiDou', 'S2', 1561.098e6, 301, 399)
BDS_B2a = Signal('BDS_B2a', 'BeiDou', 'S5', 1176.45e6,  301, 399)
BDS_B2I = Signal('BDS_B2I', 'BeiDou', 'S7', 1207.14e6,  301, 399)
BDS_B3  = Signal('BDS_B3',  'BeiDou', 'S6', 1268.52e6,  301, 399)

# --- GLONASS (FDMA — see WARNING below) ---
# GLONASS L1/L2 use a different center frequency per satellite (frequency
# division). The values below are the NOMINAL channel-0 frequencies; using
# them introduces an RH bias of up to a few cm per satellite. For sub-cm work
# you must look up the channel number per PRN from a broadcast nav file and
# adjust each satellite's wavelength. Enable at your own risk.
GLO_G1  = Signal('GLO_G1',  'GLONASS', 'S1', 1602.0e6, 101, 199)
GLO_G2  = Signal('GLO_G2',  'GLONASS', 'S2', 1246.0e6, 101, 199)

ALL_SIGNALS = (
    GPS_L1, GPS_L2, GPS_L5,
    GAL_E1, GAL_E5a, GAL_E5b, GAL_E6,
    BDS_B1I, BDS_B2a, BDS_B2I, BDS_B3,
    GLO_G1, GLO_G2,
)

# The signals actually processed by the pipeline. Edit this list to enable
# more constellations/bands. GLONASS off by default due to FDMA bias.
ENABLED_SIGNALS = (
    GPS_L1, GPS_L2, GPS_L5,
    GAL_E1, GAL_E5a, GAL_E5b,
    GLO_G1, GLO_G2,    # nominal channel-0 frequencies — see WARNING above.
                        # Per-sat FDMA bias is ~3 cm at RH ~9 m, well below
                        # our per-window σ floor of ~50 cm.
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
# Lomb-Scargle + quality control
# ---------------------------------------------------------------------------
LS_NHEIGHTS    = 2000                  # density of RH search grid for periodogram
DETREND_ORDER  = 2                     # polynomial order vs sin(elev) before LS
P2N_MIN        = 3.0                   # min peak-to-noise ratio to keep retrieval

# ---------------------------------------------------------------------------
# Windowed observation defaults (for sub-arc / rogue-wave timescale)
#
# Per-arc analysis gives ~1 obs/30 min per (sat, signal) — fine for tides,
# blind to seconds-to-minutes transients. The windowed path slides a window
# through each arc, emitting one obs per (window, signal). At 5-min window /
# 60-s stride, expect ~5000–10000 obs/day across all sats.
# ---------------------------------------------------------------------------
WINDOW_SEC     = 180                   # window length (s) for Lomb-Scargle.
                                       # 180s targets ~30s+ wave events: short
                                       # enough that a 30s wave is ~17% of the
                                       # window (not totally averaged out) while
                                       # still resolving the spectral peak.
STRIDE_SEC     = 30                    # window stride (s) — finer time resolution
MIN_WIN_PTS    = 10                    # min valid SNR samples to evaluate a window
P2N_WIN_MIN    = 2.5                   # lower P2N gate for windows (broader peaks
                                       # expected vs full-arc)

# ---------------------------------------------------------------------------
# Plotting — colored by carrier band (so signals on the same wavelength share
# a color), markered by constellation.
# ---------------------------------------------------------------------------
SNR_COL_COLOR = {
    'S1': 'C0',   # L1 / E1 / B1     (1575.42 MHz region)
    'S2': 'C1',   # L2 / G2 / B1I    (1227.6 MHz region)
    'S5': 'C2',   # L5 / E5a / B2a   (1176.45 MHz)
    'S6': 'C4',   # E6 / B3
    'S7': 'C5',   # E5b / B2I
    'S8': 'C6',   # E5 AltBOC
}
CONSTELLATION_MARKER = {
    'GPS':     'o',
    'Galileo': 's',
    'GLONASS': '^',
    'BeiDou':  'D',
    'QZSS':    'P',
    'IRNSS':   'X',
}
DIR_MARKER = {'rise': '^', 'set': 'v'}
DPI = 120
