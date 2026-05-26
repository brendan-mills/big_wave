# SSiSLS

A multi-constellation, multi-frequency GNSS interferometric reflectometry (GNSS-IR)
pipeline targeting real-time sea-level estimation and rogue-wave detection at
Greenland coastal stations.

Built on top of [gnssrefl](https://github.com/kristinemlarson/gnssrefl).

## What it does

For a single GNSS station (currently **UMNQ**, Uummannaq, Greenland), the
pipeline:

1. Reads local Hatanaka-compressed RINEX 2 observation files.
2. Calls gnssrefl's `rinex2snr` (multi-GNSS SP3 orbits) to produce SNR files.
3. Segments each day's SNR into low-elevation rising/setting arcs.
4. Runs Lomb-Scargle on each arc against each enabled signal
   (GPS L1/L2/L5, Galileo E1/E5a/E5b by default) to retrieve a reflector
   height (RH) with an analytical 1σ uncertainty.
5. Saves one parquet per day, ready for downstream state estimation.
6. Predicts the local tide from the ESR **Gr1kmTM** Greenland 1 km model
   (via pyTMD) — used as a prior for the future Kalman filter.

Per-day output: ~85 arcs × 6 signals = ~500 RH observations per day, each
with σ ≈ 1–10 cm. Ready to drive a tide-residual Kalman filter for
continuous sea-level estimation.

## Architecture

The codebase is organized in layers. Each module reads from the layer above
and writes to the layer below. The Kalman/anomaly modules (yet to be
written) consume the layer-2 parquet output without touching gnssrefl.

```
Layer 1  RAW              raw .d.Z RINEX files
                                   │
Layer 2  OBSERVATIONS     pipeline.py  ←  snr.py
                          per-arc RH + sigma + p2n (parquet)
                                   │
Layer 3  STATE            estimate.py  ←  tide.py (prior)        [not yet built]
                          Kalman / IMM → RH(t) ± sigma
                                   │
Layer 4  ANOMALY          detect.py                              [not yet built]
                          innovation gating, rogue-wave events
                                   │
Layer 5  OUTPUT           plots.py
```

The same Kalman code will eventually serve both batch replay (consuming
parquets) and a streaming runtime (consuming rolling-window observations
directly from `snr.segment_arcs`).

## Modules

| File | Purpose |
|---|---|
| `config.py` | Single source of truth: station coords, paths, az/el/RH windows, signal registry. Edit `ENABLED_SIGNALS` to add Galileo/GLONASS/BeiDou bands. |
| `snr.py` | RINEX → SNR (via gnssrefl), arc segmentation, per-arc multi-signal Lomb-Scargle with analytical σ. |
| `pipeline.py` | Batch driver: scans a RINEX folder, runs `snr.process_arcs` per day, writes parquet. Idempotent. |
| `tide.py` | Greenland 1 km tide model wrapper (pyTMD + OTIS). Per-station harmonic constants and predictions. |
| `notebooks/umnq_test.ipynb` | Exploratory notebook; the .py modules are the production code. |

Runtime data and outputs live under `data/` (gitignored):

```
data/
├── rinex/{station}-{doy_range}/  raw RINEX
├── refl_code/                    gnssrefl $REFL_CODE root (snr, orbits, exe, logs)
└── results/                      per-day parquets + plots + provenance JSON
```

First-run setup populates `data/refl_code/exe/` via `gnssrefl.installexe_cl.installexe`.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate gnssir
```

This pulls everything needed: Python 3.12+, the scientific stack
(numpy/pandas/scipy/matplotlib/xarray/pyproj), pyTMD for the tide model,
gnssrefl from PyPI, and pyarrow for parquet I/O.

If you ever need to update versions after edits to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### 2. Install the gnssrefl executables (one-time)

`gnssrefl` needs `CRX2RNX` (and optionally `gfzrnx`) installed under the
project's `$EXE` directory. From an activated env:

```bash
python -c "from gnssrefl.installexe_cl import installexe; installexe('macos')"
```

Substitute `'linux64'` or `'windows'` as appropriate. This populates
`data/refl_code/exe/` (the path `config.EXE_DIR` points at).

### 3. Drop in raw RINEX

Put RINEX 2 Hatanaka observation files in
`data/rinex/{station}-{doy_start}-{doy_end}/`. Filenames must follow the
RINEX 2 convention `{station}{doy}0.{yy}d.Z`. Update
`config.RINEX_DIR` if you use a different subdirectory name.

### 4. Get the Greenland tide model (one-time, only if running tide code)

Download Gr1kmTM from ESR (free with registration) and place at the path
in `config.TIDE_MODEL_DIR` — by default
`/Users/brmills/Documents/SSiSLS/Gr1kTM/`. Required files:
`grid_Gr1kmTM_v1`, `h_Gr1kmTM_v1`, `Model_Gr1kmTM_v1`, `xy_ll_Gr1kmTM.m`.

## Quick start

Once setup is done:

```bash
conda activate gnssir
cd /path/to/big_wave
```

### Process all RINEX files in the default folder

```bash
python pipeline.py
```

Output: `data/results/{year}/{doy:03d}.parquet`, one per day, plus a
timestamped provenance JSON in `data/results/{year}/_runs/`.

### Process a date range

```bash
python pipeline.py --doys 1-31           # full month
python pipeline.py --doys 5,10,15        # specific days
python pipeline.py --force --doys 1-3    # reprocess
```

### Read results back

```python
import pipeline
df = pipeline.load_results(2026)               # all available days
df = pipeline.load_results(2026, doys=[1,5])   # specific days
```

### Predict the tide

```python
from tide import GreenlandTideModel
tm = GreenlandTideModel(lat=70.677526, lon=-52.115415)
tm.constituent_table()                          # amplitudes + phases at the station
series = tm.predict_range('2026-01-01', '2026-01-31', step_sec=60)
fig, ax = tm.plot('2026-01-01', '2026-01-02')   # quick visual
```

## Per-arc output schema

Each row in a per-day parquet describes one (satellite, rise/set, pass)
arc. Times are tz-aware UTC.

| Column | Type | Meaning |
|---|---|---|
| `arc_id, sat, constellation, dir, pass_id` | identity | arc keys |
| `n_pts, az_mean` | descriptors | sample count, mean azimuth |
| `t_start_utc, t_end_utc, t_mid_utc` | datetime64[ns, UTC] | absolute times |
| `year, doy, t_start_sec, t_end_sec` | descriptors | UTC day + sec-of-day |
| `RH_{signal}` | float, m | reflector height (parabolic-refined) |
| `sigma_{signal}` | float, m | 1σ uncertainty |
| `p2n_{signal}` | float | periodogram peak-to-noise |
| `edge_{signal}` | bool | True if peak pinned at search-window edge |

Signals are e.g. `GPS_L1, GPS_L2, GPS_L5, GAL_E1, GAL_E5a, GAL_E5b`.
NaN where the signal isn't tracked on that satellite.

## Configuration

All knobs live in `config.py`:

- **Station:** `STATION, LAT, LON, ANT_HEIGHT_ELL, ANTENNA_MSL_M`
- **Paths:** derived from `__file__` so the project relocates cleanly
  (except `TIDE_MODEL_DIR`)
- **rinex2snr:** `ORB='gnss3'` (works for 2026 multi-GNSS), `SNR_TYPE=66`
- **Arc selection:** `AZ_MIN/MAX, EL_MIN/MAX, RH_MIN/MAX, GAP_SEC, MIN_ARC_PTS`
- **Signal registry:** `ALL_SIGNALS` (12 defined) and `ENABLED_SIGNALS`
  (6 default — GPS + Galileo CDMA bands). To add a constellation, append
  to `ENABLED_SIGNALS`.
- **Quality control:** `P2N_MIN=3.0` (peak-to-noise gate), edge-hit gate

GLONASS is defined but disabled by default — its FDMA frequencies vary
per satellite (channel offsets from broadcast nav) and would need
per-PRN wavelength lookup for sub-cm work.

## Dependencies

See [`environment.yml`](environment.yml) for the authoritative list. Notable
choices:

- **Python 3.12+** — `config.Signal` uses `dataclass(frozen=True)` and PEP 604
  union types throughout the project.
- **pyTMD 3.x** — current API; `tide.py` uses the xarray-based dataset path.
- **pyarrow via pip, not conda** — the conda-forge binary on macOS arm64 has
  a `libprotobuf`/`liborc` symbol mismatch that breaks `import pyarrow`.
  `environment.yml` already puts pyarrow under the `pip:` block for this
  reason.
- **gnssrefl 4.1+** from PyPI (it isn't on conda-forge).

`gnssrefl` expects the `$REFL_CODE`, `$ORBITS`, and `$EXE` env vars to be set.
`snr.py` configures them from `config.py` at import time, so you don't need
to call `set_environment` before running anything.

## Auxiliary data

The Greenland tide model lives outside the project tree:

```
/Users/brmills/Documents/SSiSLS/Gr1kTM/
├── grid_Gr1kmTM_v1
├── h_Gr1kmTM_v1
├── UV_Gr1kmTM_v1
└── Model_Gr1kmTM_v1
```

Polar stereographic projection (`lat_ts=70, lon_0=-45`, units=km).
`tide.py` reads the OTIS binaries directly via pyTMD; no filesystem
relocation needed.

## Station notes — UMNQ (Uummannaq)

| | |
|---|---|
| Latitude | 70.677526° N |
| Longitude | −52.115415° E |
| Ellipsoidal height | 38.00 m |
| Nominal antenna MSL height | 10.88 m |
| Azimuth wedge (fjord-facing) | 30°–180° |
| Tracked constellations (verified) | GPS, Galileo, GLONASS |

January 2026 retrievals consistently come out at RH ≈ 9.0 m — about 2 m
below the 10.88 m nominal water-level antenna height — consistent with
sea ice cover raising the reflecting surface.

## Roadmap

- [x] `tide.py` — Gr1kmTM wrapper, predictions, constituent table, plots
- [x] `config.py` — multi-constellation signal registry
- [x] `snr.py` — multi-signal Lomb-Scargle with σ, edge-hit gate
- [x] `pipeline.py` — batch RINEX → per-day parquet
- [ ] `plots.py` — standard project plot library
- [ ] `estimate.py` — tide-residual Kalman filter (linear KF first, then IMM)
- [ ] `detect.py` — rogue-wave detection via IMM model probability
- [ ] `streaming.py` — async runtime for live RINEX feed
