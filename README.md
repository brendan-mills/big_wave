# big_wave

A multi-constellation GNSS interferometric reflectometry (GNSS-IR) pipeline for
sea-level estimation and **wave-train detection** at Greenland coastal stations
(currently **UMNQ**, Uummannaq).

Built on top of [gnssrefl](https://github.com/kristinemlarson/gnssrefl).

See [CLAUDE.md](CLAUDE.md) for design rationale and conventions — read that first
if you're contributing.

## What it does

For one GNSS station the pipeline turns raw 1 Hz RINEX into a water-level record
and a candidate event log, in four stages:

1. **invsnr** — gnssrefl's B-spline RH(t) inversion of the SNR data gives a
   smoothed **water-level state** (and η = water level − predicted Gr1kmTM tide).
   This is the reference signal.
2. **roughness** — for every satellite arc, a short sliding window measures the
   **SNR-residual roughness** (RMS of the SNR after a low-order detrend,
   normalized per arc). A wave train roughens the reflecting surface and spikes
   this for many satellites at once.
3. **detect** — two triggers: **surge** (water level stays far from the tide for
   ≥1 hr) and **roughness** (a coherent multi-satellite roughness burst).
4. **plots** — a water-level overview (tide + invsnr line + event flags), a
   roughness-activity timeline, and per-event zooms.

### Why roughness instead of a fast RH retrieval?

A GNSS-IR reflector-height retrieval needs a window long enough to sweep
elevation (the SNR oscillation period is ~30–90 s here). A window short enough
to localize a ~10 s wave *can't constrain RH* — it pins at the search-grid
ceiling. So fast waves are detected from the **roughness** the wave injects, not
from per-window RH. invsnr handles the (slow, well-constrained) water level.

## Architecture

```
download.py ─► 1 Hz RINEX (.YYd.gz)          data/rinex_highrate/
                     │
   main.run():       ▼
   [1/4] invsnr     snr66 ─► water-level state.parquet      (invsnr_runner ← tide)
   [2/4] roughness  snr66 ─► per-day roughness obs          (pipeline ← snr, parallel)
   [3/4] detect     state + roughness ─► events.parquet     (detect: surge + roughness)
   [4/4] plots      ─► overview + timeline + event zooms    (plots_roughness)
```

Both invsnr and roughness read the **snr66** files made by gnssrefl `rinex2snr`
(created on demand). Every stage caches to parquet and is reused on re-runs;
`main.py` is the single entry point. `FORCE=True` reprocesses everything.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Single source of truth: the `DATA_RATE` switch, paths, station coords, signal registry, all numeric knobs |
| `snr.py` | gnssrefl `rinex2snr` wrappers (`ensure_snr`, `load_snr`), arc segmentation, SNR-residual roughness |
| `invsnr_runner.py` | Wraps `gnssrefl.invsnr` → water-level `state.parquet`; seam-free chunks + per-chunk subprocess timeout |
| `tide.py` | Gr1kmTM tide model wrapper (pyTMD/OTIS): harmonic constants + predictions |
| `pipeline.py` | Day discovery + parallel day-map over the roughness stage + run provenance |
| `detect.py` | Two triggers — `surge` + `roughness` — → `events.parquet` |
| `plots_roughness.py` | Water-level overview, roughness timeline, per-event zooms |
| `main.py` | Orchestrator: invsnr → roughness → detect → plots |
| `download.py` | Fetch 1 Hz RINEX from EarthScope + Hatanaka/gzip compress |
| `characterize_roughness.py` | Diagnostic to set `RoughnessConfig` from the roughness distribution |
| `constituents.py` | Validation: harmonic analysis of the invsnr water level vs Gr1kmTM |

Runtime data and outputs live under `data/` (gitignored). The `DATA_RATE` switch
re-roots into separate trees so the 1 Hz and 15 s datasets never collide:

```
data/
├── rinex_highrate/umnq/         1 Hz RINEX (.YYd.gz), from download.py     [DATA_RATE='1Hz']
├── rinex/{station}-{range}/     15 s archived RINEX (.YYd.Z)               [DATA_RATE='15s']
├── refl_code/                   SHARED: exe (CRX2RNX, gfzrnx), orbits, sso_tokens.json
├── refl_code_1hz/{year}/snr/    1 Hz snr66 tree ($REFL_CODE during processing)
├── results_1hz/
│   ├── {year}/roughness/{doy:03d}_obs.parquet   per-day roughness obs
│   ├── {year}/invsnr/{doy:03d}_state.parquet    per-day invsnr state
│   ├── range/{tag}/state.parquet                stitched water-level state
│   ├── range/{tag}/events.parquet               surge + roughness events
│   └── plots/                                   overview, timeline, events/
└── Gr1kTM/                      Greenland tide model
```

`{tag}` is auto-derived from the (year, doy) span, e.g. `2025110-2025240`.

## Setup

### 1. Conda environment

```bash
conda env create -f environment.yml
conda activate gnssir
```

Pulls the scientific stack (numpy/pandas/scipy/matplotlib), pyTMD (tide model),
gnssrefl (PyPI), pyarrow (parquet, **pip not conda** — see Dependencies), and
`hatanaka` (RINEX compression for `download.py`).

### 2. Install gnssrefl executables (one-time)

```bash
python -c "from gnssrefl.installexe_cl import installexe; installexe('macos')"
```

(`'linux64'` / `'windows'` as appropriate.) Populates `data/refl_code/exe/`
(`config.EXE_DIR`) with `CRX2RNX` and `gfzrnx`.

### 3. Get 1 Hz RINEX

`download.py` fetches high-rate RINEX from EarthScope and Hatanaka/gzip-compresses
each day to `data/rinex_highrate/{station}/`. Edit its RUN CONFIG (station, year,
doy range), then run. First use triggers an **EarthScope SSO device login**; the
token is cached at `data/refl_code/sso_tokens.json`. Files land as
`{station}{doy}0.{yy}d.gz`.

### 4. Greenland tide model (one-time)

Download [Gr1kmTM](https://arcticdata.io/catalog/view/doi:10.18739/A2B853K18)
from ESR and place under `data/Gr1kTM/`: `grid_Gr1kmTM_v1`, `h_Gr1kmTM_v1`,
`Model_Gr1kmTM_v1`, etc.

## Quick start

Edit the constants at the top of `main.py`, then run:

```python
# main.py — RUN CONFIG
START_DATE   = (2025, 110)   # (year, doy) inclusive, or None for "all"
END_DATE     = (2025, 240)
FORCE        = False         # True = reprocess every stage
FORCE_INVSNR = False         # refit invsnr while reusing other caches
MAKE_PLOTS   = True
```

```bash
conda activate gnssir
python main.py
```

```
[1/4] invsnr water-level state     gnssrefl B-spline inversion → state.parquet
[2/4] SNR roughness                per-day roughness obs (parallel across days)
[3/4] Detect events                surge + roughness → events.parquet
[4/4] Plots                        overview + roughness timeline + event zooms
```

Each stage skips if its cache exists. invsnr is the slowest stage; roughness is
fast (cheap per-window polyfit, parallelized). **Never run two instances at
once** — invsnr writes a single shared output file and concurrent runs corrupt
each other's caches.

### Re-running plots / tuning the detector

```bash
python plots_roughness.py        # re-render figures from cached state + events
python characterize_roughness.py # roughness-ratio distribution + (ratio × sats) grid
```

To re-tune the detector cheaply: run `characterize_roughness.py`, edit
`detect.RoughnessConfig`, set `FORCE=False`, delete `events.parquet`, re-run
`main.py` (reuses the expensive caches, re-detects in seconds).

### Predict the tide directly

```python
from tide import GreenlandTideModel
tm = GreenlandTideModel(lat=70.677526, lon=-52.115415)
tm.constituent_table()
fig, ax = tm.plot('2026-01-01', '2026-01-02')
```

## Output schemas

### `state.parquet` (invsnr water level)
| Column | Meaning |
|---|---|
| `t_utc` | tz-aware UTC timestamp (invsnr `delta_out`, ~5 min) |
| `water_level_m` | invsnr water level (m, MSL) |
| `tide_m` | predicted Gr1kmTM tide |
| `eta_m` | water level − tide (the surge signal) |
| `eta_sigma_m` | flat uncertainty estimate |

### Roughness obs (`{doy}_obs.parquet`)
| Column | Meaning |
|---|---|
| `t_center_utc` | window midpoint |
| `arc_id, sat, constellation, signal, snr_col` | arc / signal identity |
| `roughness` | normalized SNR-residual RMS in the window |
| `baseline` | per-arc median roughness (calm floor) |
| `rough_ratio` | `roughness / baseline` — what the detector thresholds |

### `events.parquet` (surge + roughness)
| Column | Meaning |
|---|---|
| `event_id, trigger` | sequential; `'surge'` or `'roughness'` |
| `t_start_utc, t_end_utc, t_peak_utc, duration_sec` | extent |
| `confidence` | rank metric (threshold-multiples × √duration) |
| `peak_tide_dev_m, water_level_at_peak_m` | surge rows |
| `peak_rough_ratio, max_sats, max_constellations` | roughness rows |

## Configuration knobs

All tuning lives in `config.py` (data + roughness), `detect.DetectorConfig`
(surge), and `detect.RoughnessConfig` (wave trains). See
[CLAUDE.md](CLAUDE.md#current-tuning) for current values and rationale.

- **`config.py`**: `DATA_RATE` (`'1Hz'`/`'15s'`), `RINEX_DEC` (1 = full rate, required
  for roughness), `INVSNR_DEC`, `INVSNR_TIMEOUT_SEC`, `N_WORKERS`,
  `ROUGH_WIN_SEC/STRIDE/MIN_PTS`, `RH_MIN/MAX`, az/el window, `ENABLED_SIGNALS`.
- **`detect.DetectorConfig`** (surge): `tide_dev_window_sec`, `min_tide_dev_m`.
- **`detect.RoughnessConfig`** (waves): `rough_ratio_min`, `min_sats`, `bin_sec`.

GLONASS uses nominal channel-0 frequencies (≈3 cm per-sat RH bias) — negligible
for roughness, which keys on relative SNR fluctuations, not absolute RH.

## Dependencies

See [`environment.yml`](environment.yml). Notable choices:

- **gnssrefl 4.1+** from PyPI (not on conda-forge); provides `rinex2snr` + `invsnr`.
- **pyTMD 3.x** — `tide.py` reads the Gr1kmTM OTIS binaries.
- **`hatanaka` (pip)** — `download.py` uses it to Hatanaka+gzip compress 1 Hz files
  (gnssrefl ships `CRX2RNX` to *de*compress but not `RNX2CRX`).
- **pyarrow via pip, not conda** — the conda-forge arm64 build has a
  `libprotobuf` symbol mismatch that breaks `import pyarrow`.

`gnssrefl` reads `$REFL_CODE`/`$ORBITS`/`$EXE`; `snr.py` and `invsnr_runner.py`
set them from `config.py` at import (the snr66 tree is rate-specific; orbits/exe
are shared).

## Station notes — UMNQ (Uummannaq, west Greenland)

| | |
|---|---|
| Latitude | 70.677526° N |
| Longitude | −52.115415° E |
| Antenna ellipsoidal height | 38.00 m |
| Antenna MSL height | 8.88 m (user-calibrated) |
| Azimuth wedge (fjord-facing) | 30°–180° |
| Constellations | GPS, Galileo, GLONASS (no BeiDou, no QZSS — verified) |
| Archive cadence | 15 s; 1 Hz fetched separately by `download.py` for wave work |
| Conditions | Open water summer (May–Oct); variable→fractured ice (Nov–Apr) |

Winter ice corrupts the signal (fractured ice + water + snow), so water level is
effectively unobservable and surge can false-fire from ice-driven offsets.

## Status

- [x] `download.py` — 1 Hz fetch + Hatanaka/gzip
- [x] `snr.py` — snr66 + SNR roughness
- [x] `invsnr_runner.py` — water-level state, seam-free chunks, hang-proof timeout
- [x] `pipeline.py` — parallel roughness stage
- [x] `detect.py` — surge + roughness
- [x] `plots_roughness.py` — water-level overview + roughness highlights
- [x] `main.py` — invsnr → roughness → detect → plots
- [ ] Ground-truth validation of events
- [ ] Streaming / real-time runtime
- [ ] Multi-station support
