# big_wave

A multi-constellation, multi-frequency GNSS interferometric reflectometry
(GNSS-IR) pipeline for sea-level estimation and rogue-wave detection at
Greenland coastal stations.

Built on top of [gnssrefl](https://github.com/kristinemlarson/gnssrefl).

See [CLAUDE.md](CLAUDE.md) for project context, design rationale, and
conventions — read that first if you're contributing.

## What it does

For a single GNSS station (currently **UMNQ**, Uummannaq, Greenland), the
pipeline runs five stages from raw RINEX to a candidate event log:

1. **RINEX → SNR** via gnssrefl's `rinex2snr` with multi-GNSS SP3 orbits.
2. **Sub-arc windowed Lomb-Scargle** — slides a 3-minute window through every
   satellite arc, retrieving a reflector height (RH) every 30 seconds per
   (sat, band) for **GPS L1/L2/L5, Galileo E1/E5a/E5b, and GLONASS G1/G2**.
3. **Multi-sat consensus binning** — averages obs from all sats/bands inside
   each 2-minute time bin using an inverse-variance weighted mean. Drops the
   per-bin noise floor from ~50 cm to ~30 cm.
4. **Random-walk Kalman filter** on η = water level − predicted tide
   (Gr1kmTM model). Produces a smoothed water-level time series with σ band.
5. **Coherent-event detection** — walks the *raw* (pre-binning) obs stream
   and flags time bins where many sats agree on a same-sign departure from
   the smoothed state. Output is a per-event log with confidence scores.

A 30-second wave of 2 m amplitude is visible across a few consecutive 2-minute
bins in 3+ satellites simultaneously — that's the signature the detector
hunts for.

## Architecture

```
Layer 1  RAW              raw .d.Z RINEX files (data/rinex/)
                                   │
Layer 2  OBSERVATIONS     pipeline.py  ←  snr.py
                          per-day per-arc + windowed obs parquets
                                   │
Layer 3  STATE            estimate.py  ←  tide.py (prior)
                          bin obs → 1-D random-walk Kalman → state series
                                   │
Layer 4  ANOMALY          detect.py
                          coherence test on raw obs → event log
                                   │
Layer 5  OUTPUT           plots.py
```

Every stage is idempotent — outputs cache to parquet and are reused on
subsequent runs. `main.py` is the single entry point that drives all five
stages.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Single source of truth: paths, station coords, az/el/RH windows, signal registry, all numeric knobs |
| `snr.py` | gnssrefl wrappers; per-arc and per-window Lomb-Scargle with analytical σ |
| `pipeline.py` | Per-day caching layer over `snr.py` — `process_folder` / `process_folder_windowed` |
| `tide.py` | Gr1kmTM tide model wrapper (pyTMD + OTIS): harmonic constants and predictions |
| `estimate.py` | 1-D random-walk Kalman filter on η = water level − tide. Multi-sat consensus binning |
| `detect.py` | Spatial+temporal coherence test → candidate event log (8 tunable gates) |
| `plots.py` | Headline water-level plot, residual plot, per-event window plots |
| `main.py` | Top-level orchestrator — edit constants, click Run |

Runtime data and outputs live under `data/` (gitignored):

```
data/
├── rinex/{station}-{doy_range}/  raw RINEX
├── refl_code/                    gnssrefl $REFL_CODE root (snr, orbits, exe, logs)
└── results/
    ├── {year}/{doy:03d}.parquet           per-day per-arc (vestigial; mostly for caching)
    ├── {year}/windowed/{doy:03d}_obs.parquet  per-day windowed obs
    └── range/{tag}/                       per-run aggregates
        ├── binned.parquet                   multi-sat consensus bins
        ├── state.parquet                    KF state series
        ├── gated.parquet                    obs rejected by KF gate
        ├── events.parquet                   candidate events
        └── innov.parquet                    full innovation log
```

`{tag}` is auto-derived from the (year, doy) span, e.g. `2025085-2026145`.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate gnssir
```

This pulls Python 3.12+, the scientific stack
(numpy/pandas/scipy/matplotlib/xarray/pyproj), pyTMD for the tide model,
gnssrefl from PyPI, and pyarrow for parquet I/O.

To update after edits:

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

Download RINEX observations files from [UNAVCO](https://www.unavco.org/data/gps-gnss/data-access-methods/dai1/perm_sta.php?pview=original&offset_check2=0_check&filter_station_code=checked&station_code=UMNQ&action=View+Results). This is the link for UMNQ. 

Put RINEX 2 Hatanaka observation files in
`data/rinex/{station}-{doy_start}-{doy_end}/`. Filenames must follow the
RINEX 2 convention `{station}{doy}0.{yy}d.Z`. Update `config.RINEX_DIR`
to point at your folder.

### 4. Get the Greenland tide model (one-time, only if running tide code)

Download [Gr1kmTM](https://arcticdata.io/catalog/view/doi:10.18739/A2B853K18)
from ESR (free with registration) and place under `data/Gr1kTM/` (the
default `config.TIDE_MODEL_DIR`). Required files:
`grid_Gr1kmTM_v1`, `h_Gr1kmTM_v1`, `Model_Gr1kmTM_v1`, `xy_ll_Gr1kmTM.m`.

## Quick start

Run the whole pipeline by editing constants at the top of `main.py` then
running it:

```python
# main.py — RUN CONFIG block
START_DATE = (2025, 200)         # (year, doy) inclusive — or None for "all"
END_DATE   = (2025, 213)         # (year, doy) inclusive — or None for "all"
RINEX_FOLDER = c.RINEX_DIR
BIN_SEC    = 120.0               # multi-sat consensus bin width
FORCE      = False               # True = reprocess every stage
MAKE_PLOTS = True
```

```bash
conda activate gnssir
python main.py
```

The orchestrator runs:

```
[1/6] Per-arc preprocessing        gnssrefl RINEX → SNR → per-arc parquet
[2/6] Windowed observations        per-day windowed obs parquet
[3/6] Binning obs                  multi-sat consensus → binned parquet
[4/6] Running Kalman filter        state parquet
[5/6] Detecting events             events parquet
[6/6] Plots                        defers to plots.py
```

Each stage skips if its cached parquet exists. Stage 2 is the slow one
(~30–60 min for a full year of fresh days); stages 3–5 are < 5 minutes
total.

### Re-running plots only

`plots.py` reads from the saved parquets — no recomputation needed:

```bash
python plots.py
```

It auto-picks the most recent `data/results/range/{tag}/` directory; edit
`TAG` at the top of its `__main__` block to override.

### Predict the tide directly

```python
from tide import GreenlandTideModel
tm = GreenlandTideModel(lat=70.677526, lon=-52.115415)
tm.constituent_table()
fig, ax = tm.plot('2026-01-01', '2026-01-02')
```

## Output schemas

### Windowed observations (`{doy}_obs.parquet`)

Long-form, one row per (window, signal). Times are tz-aware UTC.

| Column | Type | Meaning |
|---|---|---|
| `t_center_utc` | datetime64 | window midpoint |
| `arc_id, sat, constellation, dir, pass_id` | identity | arc keys |
| `signal, snr_col` | str | e.g. `GPS_L1`, `S1` |
| `n_pts_window` | int | SNR samples in this window |
| `elev_center, azim_center` | float, deg | geometry midpoint |
| `rh` | float, m | retrieved reflector height (parabolic-refined) |
| `sigma` | float, m | analytical 1σ uncertainty |
| `p2n` | float | periodogram peak-to-noise ratio |
| `edge_hit` | bool | True if peak pinned at search-window edge |

### Binned observations (`binned.parquet`)

Long-form, one row per time bin. Inverse-variance weighted mean across
all sats and signals in the bin, after outlier rejection.

| Column | Meaning |
|---|---|
| `t_center_utc, rh, sigma, sat=-1, signal='binned'` | consumed by KF |
| `n_obs, n_sats, n_dropped` | bin diagnostics |
| `rh_spread_m` | max RH spread within bin (pre-aggregation) |

### KF state series (`state.parquet`)

One row per accepted observation.

| Column | Meaning |
|---|---|
| `t_utc, sat, signal` | identity (sat=-1, signal='binned' if binned input) |
| `eta_m, eta_sigma_m` | filter state and σ |
| `tide_m` | predicted tide at this time |
| `water_level_m` | `tide_m + eta_m` — absolute water level |
| `innov, mahal2` | innovation and Mahalanobis² for this update |

### Events (`events.parquet`)

One row per detected candidate event.

| Column | Meaning |
|---|---|
| `event_id` | sequential |
| `t_start_utc, t_end_utc, t_peak_utc, duration_sec` | event extent |
| `amplitude_m` | signed innovation at peak |
| `direction` | 'rise' or 'fall' |
| `n_sats_peak, n_obs_total, n_bins` | size/coverage metrics |
| `max_mahal` | worst-case mahal² of any obs in the event |
| `confidence` | heuristic score (n_sats × √n_bins × log10(1 + 10×amp)) |

## Configuration knobs

All numeric tuning lives in `config.py` (data), `estimate.TideKalmanConfig`
(KF), and `detect.DetectionConfig` (detector). Defaults are tuned for
detecting waves ≥ ~2 m at UMNQ.

**`config.py` (data + windowing):**
- Station: `STATION, LAT, LON, ANT_HEIGHT_ELL, ANTENNA_MSL_M`
- Paths: derived from `__file__` so the project relocates cleanly
- rinex2snr: `ORB='gnss3'`, `SNR_TYPE=66`
- Arc selection: `AZ_MIN/MAX, EL_MIN/MAX, RH_MIN/MAX, GAP_SEC, MIN_ARC_PTS`
- **Windowed: `WINDOW_SEC=180, STRIDE_SEC=30`** (current setting for
  wave-event detection — narrower windows than the original 300/60 to
  catch ~30 s+ events)
- Signal registry: `ALL_SIGNALS` (12 defined) and `ENABLED_SIGNALS`
  (8 default: GPS L1/L2/L5, Galileo E1/E5a/E5b, GLONASS G1/G2). Append
  to `ENABLED_SIGNALS` to add bands/constellations.

**`estimate.TideKalmanConfig` (5 fields):**
- `sigma_p` — random-walk process noise on η
- `gate_threshold` — Mahalanobis² cutoff for KF rejection
- `sigma_inflation_m` — adds to obs σ in quadrature
- `init_eta_sigma` — initial state uncertainty
- `antenna_msl_m` — pulled from config by default

**`detect.DetectionConfig` (8 fields):**
- `bin_sec`, `mahal_threshold`, `min_sats_per_bin`, `coherence_ratio_min`,
  `min_amplitude_m`, `snr_min`, `max_event_gap_sec`, `antenna_msl_m`

See [CLAUDE.md](CLAUDE.md#current-tuning-as-of-latest-run) for the current
values and what each knob does physically.

GLONASS uses nominal channel-0 frequencies (1602.0 MHz / 1246.0 MHz)
rather than per-satellite channel lookup. This introduces a per-sat RH
bias of ≈3 cm at typical RH — well below the per-window σ floor of ~50 cm
and washed out by multi-sat consensus binning. For sub-cm research you
would read channel numbers from a broadcast nav file and compute per-sat
wavelengths; the current usage doesn't warrant that complexity.

## Dependencies

See [`environment.yml`](environment.yml) for the authoritative list.
Notable choices:

- **Python 3.12+** — `config.Signal` uses `dataclass(frozen=True)` and
  PEP 604 union types throughout the project.
- **pyTMD 3.x** — current API; `tide.py` uses the xarray-based dataset path.
- **pyarrow via pip, not conda** — the conda-forge binary on macOS arm64
  has a `libprotobuf`/`liborc` symbol mismatch that breaks `import pyarrow`.
  `environment.yml` already puts pyarrow under the `pip:` block for this
  reason.
- **gnssrefl 4.1+** from PyPI (it isn't on conda-forge).

`gnssrefl` expects `$REFL_CODE`, `$ORBITS`, and `$EXE` env vars to be set.
`snr.py` configures them from `config.py` at import time, so you don't
need to call `set_environment` before running anything.

## Auxiliary data

The Greenland tide model lives under `data/Gr1kTM/` (gitignored along with
the rest of `data/`):

```
data/Gr1kTM/
├── grid_Gr1kmTM_v1
├── h_Gr1kmTM_v1
├── UV_Gr1kmTM_v1
└── Model_Gr1kmTM_v1
```

Polar stereographic projection (`lat_ts=70, lon_0=-45`, units=km).
`tide.py` reads the OTIS binaries directly via pyTMD; no filesystem
relocation needed.

## Station notes — UMNQ (Uummannaq, west Greenland)

| | |
|---|---|
| Latitude | 70.677526° N |
| Longitude | −52.115415° E |
| Ellipsoidal height | 38.00 m |
| Antenna MSL height | 8.88 m (user-calibrated; was 10.88 nominal) |
| Azimuth wedge (fjord-facing) | 30°–180° |
| Constellations tracked | GPS, Galileo, GLONASS (no BeiDou, no QZSS — verified) |
| Typical conditions | Open water summer (May–Oct); variable to fractured ice (Nov–Apr) |

Winter ice corrupts the signal — the reflecting "surface" becomes a
fractured, mobile patchwork of ice + water + snow. RH retrievals vary
by 1–3 m across satellites at the same instant. The detector correctly
produces near-zero events during these periods (the adaptive SNR gate
suppresses), but water level is effectively unobservable through the ice.

## Status

- [x] `tide.py` — Gr1kmTM wrapper, predictions, constituent table, plots
- [x] `config.py` — multi-constellation signal registry (GPS/Galileo/GLONASS)
- [x] `snr.py` — per-arc + windowed multi-signal Lomb-Scargle with σ
- [x] `pipeline.py` — batch RINEX → per-day per-arc + windowed parquets
- [x] `estimate.py` — random-walk Kalman filter + multi-sat consensus binning
- [x] `detect.py` — coherent-event detection with 8 tunable gates
- [x] `plots.py` — water-level overlay, residual, per-event window plots
- [x] `main.py` — single-entry-point orchestrator with caching
- [ ] Validation against known events (no ground truth yet)
- [ ] Streaming / real-time runtime
- [ ] Multi-station support
