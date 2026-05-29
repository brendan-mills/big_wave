# big_wave — project context for Claude

A multi-constellation GNSS interferometric reflectometry (GNSS-IR) pipeline for
sea-level estimation and **wave-train detection** at Greenland coastal stations
(currently UMNQ, Uummannaq). Built on top of
[gnssrefl](https://github.com/kristinemlarson/gnssrefl).

This file documents the working conventions, design decisions, and current
challenges — read it first before suggesting changes.

---

## How to run

**Activate the env first.** Everything depends on the `gnssir` conda env:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate gnssir
```

**Single entry point is `main.py`.** The user runs it from VS Code's Run button
(or `python main.py`) — **never add argparse**. Edit the constants at the top of
`main.py` (`START_DATE`, `END_DATE`, `FORCE`, `FORCE_INVSNR`, `MAKE_PLOTS`), then
run. Each stage caches; `FORCE=True` reprocesses everything.

To fetch new 1 Hz data: edit + run `download.py`. To re-tune the detector after a
run: run `characterize_roughness.py`. To regenerate just the figures from cached
results: run `plots_roughness.py`.

---

## Architecture — a linear 1 Hz flow

```
download.py ──► RINEX (1 Hz, .YYd.gz)         data/rinex_highrate/
                        │
  ┌─────────────────────┴───────────────────────────────────────┐
  │ main.run()                                                    │
  │                                                               │
  │  1. invsnr   (invsnr_runner)  snr66 ─► water-level STATE      │  the reference
  │               B-spline RH(t) vs tide  → state.parquet         │  (eta = WL − tide)
  │                                                               │
  │  2. roughness (pipeline ← snr) snr66 ─► per-day ROUGHNESS obs │  the wave signal
  │               SNR-residual roughness, parallel across days    │
  │                                                               │
  │  3. detect   (detect)  state + roughness ─► events.parquet    │  TWO triggers:
  │               surge (WL vs tide) + roughness burst            │  surge, roughness
  │                                                               │
  │  4. plots    (plots_roughness)  water-level overview + …      │
  └───────────────────────────────────────────────────────────────┘
```

Both `invsnr` and `roughness` read the **snr66** files (made by gnssrefl
`rinex2snr`, created on demand). invsnr does its own RH retrieval internally — we
do **not** run a custom Lomb-Scargle RH retrieval anymore. Range artifacts land
under `data/.../range/{tag}/`; per-day caches under `data/.../{year}/`.

---

## Module map

| File | Purpose |
|---|---|
| `config.py` | Single source of truth: the `DATA_RATE` switch, paths, station coords, signal registry (`ENABLED_SIGNALS`), and all numeric knobs. **No env vars on import beyond what snr/invsnr_runner set; no CLI.** |
| `snr.py` | gnssrefl `rinex2snr` wrapper (`ensure_snr`, `load_snr`), arc segmentation, and the per-(window, sat, signal) **SNR-residual roughness** (`process_arcs_roughness`). No RH retrieval. |
| `invsnr_runner.py` | Wraps `gnssrefl.invsnr` (B-spline RH inversion) → water-level `state.parquet`. Overlapping seam-free chunks; per-chunk **subprocess timeout** to skip hangs; `INVSNR_DEC` decimation. |
| `tide.py` | Gr1kmTM tide model wrapper (per-station harmonic constants + predictions). |
| `pipeline.py` | Day discovery + **parallel day-map** (`N_WORKERS`) over the roughness stage + run provenance. |
| `detect.py` | **Two triggers**: `surge` (sustained WL−tide) + `roughness` (coherent multi-sat burst). `detect_events` → `events.parquet`. |
| `plots_roughness.py` | Water-level overview (tide + invsnr WL + event flags), roughness clustering timeline, top-5 per-event zooms. |
| `main.py` | Orchestrator: invsnr → roughness → detect → plots. Edit constants → Run. |
| `characterize_roughness.py` | Diagnostic: `rough_ratio` distribution + (ratio × sats) event-count grid to set `RoughnessConfig`. |
| `download.py` | Fetch 1 Hz RINEX from EarthScope, Hatanaka+gzip compress per day. |
| `constituents.py` | Validation: harmonic analysis of the invsnr water level vs Gr1kmTM constituents. |

---

## Working conventions (do these unprompted)

- **Module-level constants, never argparse** — the user runs from VS Code Run.
- **Use the `gnssir` conda env** for all Bash; use `python`, never system `python3`.
- **Keep `data/` out of git** (raw RINEX, snr66, parquets, plots, SSO token). RINEX
  file-format patterns are also in `.gitignore` as belt-and-suspenders.
- **Never run two pipeline instances at once** — invsnr writes a *single shared
  output file*, so concurrent runs corrupt each other's per-day caches.
- **Full 1 Hz is required for roughness** (`RINEX_DEC=1`); decimation destroys the
  fast SNR fluctuations the wave detector keys on.
- **Don't reintroduce** the removed windowed-RH retrieval, the per-arc Lomb-Scargle,
  or a Kalman filter — invsnr is the water-level reference now.
- **Don't suggest microservices, async runtimes, or web dashboards** — local tool.
- **`pip install pyarrow`**, never conda (libprotobuf incompatibility on arm64).

---

## Current tuning

### `config.py`
| knob | value | role |
|---|---|---|
| `DATA_RATE` | `'1Hz'` | selects rate-specific paths + `RINEX_DEC`/`INVSNR_DEC`; `'15s'` re-roots to the archived set |
| `RINEX_DEC` | 1 (1 Hz) | rinex2snr decimation; 1 = full rate (required for roughness) |
| `INVSNR_DEC` | 15 (1 Hz) | invsnr's own SNR decimation — tide-scale spline gains nothing from 1 Hz and hangs on it |
| `INVSNR_TIMEOUT_SEC` | 300 | per-chunk wall-clock cap; a hung chunk is skipped, not fatal |
| `N_WORKERS` / `SNR_WORKERS` | RAM-bounded (6 / 3 on a 16 GB box) | parallel days — roughness / snr66 creation. Capped by RAM, not cores: at 1 Hz a day is several GB resident, so cores−1 froze a 16 GB Mac. Workers recycle per day (`maxtasksperchild=1`). |
| `ROUGH_WIN_SEC / STRIDE / MIN_PTS` | 20 / 5 / 15 | short-window residual-RMS roughness |
| `RH_MIN, RH_MAX` | 4, 16 m | invsnr RH search range |
| `AZ / EL window` | 30–180° / 5–25° | fjord-facing arcs |
| `ENABLED_SIGNALS` | GPS L1/L2/L5, Gal E1/E5a/E5b, GLONASS G1/G2 | (GLONASS nominal FDMA freqs; ~3 cm bias, fine for roughness) |

### `detect.DetectorConfig` (surge)
| knob | value | role |
|---|---|---|
| `tide_dev_window_sec` | 3600 | 1 hr rolling median of eta |
| `min_tide_dev_m` | 0.75 | sustained |WL−tide| threshold |

### `detect.RoughnessConfig` (wave trains) — set from `characterize_roughness.py`
| knob | value | role |
|---|---|---|
| `rough_ratio_min` | 4.0 | per-sat roughness vs its per-arc median (~p99.5 of the calm tail) |
| `min_sats` | 4 | distinct sats coherently rough in a bin (spatial coherence) |
| `bin_sec` | 30 | coherence time bin |

These give ~25 episodic events over doy 110–240 / 2025 (clustered May 6–12 +
late-Jul/early-Aug). Loosen to 3.0/4 for recall, tighten to 5.0/4 for precision.

---

## Key design decisions and rationale

### Why roughness, not windowed RH, for fast waves
A GNSS-IR RH retrieval needs a window long enough to sweep elevation and
constrain the SNR oscillation frequency (period τ ≈ 30–90 s here). A window short
enough to localize a ~10 s wave **can't constrain RH** — empirically a 60 s window
pins RH at the grid ceiling (~15 m vs true ~9 m). So fast waves are detected from
**SNR-residual roughness**: in a short window the slow RH oscillation is a smooth
low-order trend, while a wave train injects fast fluctuations the trend can't
absorb → residual RMS rises. No frequency fit needed, so a 20 s window is fine.
Bonus: roughness has no P2N/RH gate, so all constellations contribute (GPS 43 % /
Gal 39 % / GLONASS 17 %), unlike windowed RH which was 95 % GLONASS.

### Why invsnr is the water-level reference
invsnr (gnssrefl) does a joint multi-sat/multi-freq B-spline RH(t) inversion —
robust and validated, and it replaces the old custom LS retrieval + Kalman filter
entirely. The surge trigger and the plots both read its `state.parquet`
(`water_level_m`, `tide_m`, `eta_m`).

### Why two events only
`surge` = sustained non-tidal water-level offset (storm surge / winter sea-ice
corruption). `roughness` = a coherent multi-sat wave-train burst. The old jump and
windowed-RH "straddle" triggers were removed.

### Why the `DATA_RATE` switch re-roots everything
The 1 Hz and 15 s datasets overlap in time, and every cache key is
station/year/doy. `DATA_RATE` re-roots `RINEX_DIR`, the snr66 tree
(`SNR_REFL_CODE`), and `RESULTS_DIR` into separate trees so the two never collide;
orbits/exe/SSO-token are shared.

### Why invsnr runs in a timeout subprocess
gnssrefl's invsnr occasionally spins at 100 % CPU forever on a pathological chunk
(seen ~doy 164). Each chunk runs in a spawned subprocess capped at
`INVSNR_TIMEOUT_SEC`; on timeout it's killed and that chunk falls back to per-day
fits (the bad day is skipped, the run continues).

---

## Known challenges and open questions

### No ground-truth validation (the big one)
The ~25 events are an **operating point, not validated detections**. They cluster
in time (plausible for real sea-state episodes), but none are confirmed.
**Next step:** cross-check the clustered dates (esp. May 6–12) against Greenland
weather/storm records or a nearby tide gauge.

### Sea ice corrupts the signal
In winter the fjord fills with mobile ice + water + snow; RH retrievals scatter by
1–3 m across sats and the water level becomes unobservable. Surge can false-fire
in these periods (sustained eta offset from ice, not surge).

### Detector thresholds are calibrated, not validated
`RoughnessConfig` (4.0/4/30) was chosen from the roughness-ratio distribution, not
against known waves. Re-run `characterize_roughness.py` if the data or window
changes.

### Real-time runtime not built
Everything is batch replay from disk.

---

## Station notes — UMNQ (Uummannaq, west Greenland)

| | |
|---|---|
| Latitude | 70.677526° N |
| Longitude | −52.115415° E |
| Antenna ellipsoidal height | 38.00 m |
| Antenna height above MSL | 8.88 m (user-calibrated) |
| Azimuth wedge (fjord-facing) | 30°–180° |
| Constellations tracked | GPS, Galileo, GLONASS (no BeiDou, no QZSS — verified) |
| Daily archive cadence | 15 s; **1 Hz** fetched separately by `download.py` for wave work |
| Typical conditions | Open water summer (May–Oct), variable→fractured ice (Nov–Apr) |

---

## What NOT to do

| Don't | Why |
|---|---|
| Add CLI argparse to any entrypoint | User runs from VS Code Run button |
| Run two pipeline instances at once | invsnr's shared output file → cache corruption |
| Decimate the 1 Hz RINEX (`RINEX_DEC`>1) | kills the fast SNR signal roughness needs |
| Reintroduce windowed-RH / per-arc LS / Kalman | deliberately removed; invsnr is the reference |
| Suggest microservices, async, or web UIs | out of scope; researcher's local tool |
| Touch `data/` from git | gitignored; includes the EarthScope SSO token |
| Use system `python3` / conda-install pyarrow | needs the `gnssir` env; pyarrow via pip only |

---

## Status

- [x] `download.py` — 1 Hz fetch + Hatanaka/gzip compression
- [x] `snr.py` — snr66 + SNR roughness
- [x] `invsnr_runner.py` — water-level state, seam-free chunks, hang-proof timeout
- [x] `pipeline.py` — parallel roughness stage
- [x] `detect.py` — surge + roughness
- [x] `plots_roughness.py` — water-level overview + roughness highlights
- [x] `main.py` — invsnr → roughness → detect → plots
- [ ] Ground-truth validation of events
- [ ] Streaming / real-time runtime
- [ ] Multi-station support
