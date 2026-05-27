# big_wave — project context for Claude

A multi-constellation, multi-frequency GNSS interferometric reflectometry
(GNSS-IR) pipeline targeting real-time sea-level estimation and rogue-wave
detection at Greenland coastal stations. Built on top of
[gnssrefl](https://github.com/kristinemlarson/gnssrefl).

See [README.md](README.md) for setup. This file documents working conventions,
design decisions, and current challenges — read first before suggesting
changes.

---

## How to run

**Always activate the env first.** All scripts depend on packages in the
`gnssir` conda env:

```bash
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate gnssir
```

**Single entry point** is `main.py`. The user runs everything from VS Code's
Run button — **do not add argparse** anywhere. Edit constants at the top
of `main.py` (`START_DATE`, `END_DATE`, `FORCE`, `MAKE_PLOTS`), then click
Run.

Plots are produced by `plots.py`, also driven by constants at the top of
its `__main__` block.

---

## Architecture (5 layers + plots)

```
Layer 1  RAW              raw .d.Z RINEX files (data/rinex/)
                                   │
Layer 2  OBSERVATIONS     pipeline.py  ←  snr.py
                          per-day per-arc + windowed obs parquets
                                   │
Layer 3  STATE            estimate.py  ←  tide.py (prior)
                          1-D random-walk KF on η = water level − tide
                                   │
Layer 4  ANOMALY          detect.py
                          spatial+temporal coherence test → event log
                                   │
Layer 5  OUTPUT           plots.py
```

Each stage is idempotent (skips when its parquet exists). `main.py`
orchestrates all of them with per-stage caching under
`data/results/range/{tag}/`.

---

## Module map

| File | Purpose |
|---|---|
| `config.py` | Single source of truth: paths, station coords, signal registry, all numeric knobs. **No env vars, no CLI.** |
| `snr.py` | gnssrefl wrappers + per-arc + windowed Lomb-Scargle. Two outputs: per-arc (wide) and per-window (long form). |
| `pipeline.py` | Per-day caching layer over `snr.py`. Both `process_folder` (per-arc) and `process_folder_windowed`. |
| `tide.py` | Gr1kmTM tide model wrapper. Per-station harmonic constants and predictions. |
| `estimate.py` | 1-D random-walk KF on (water level − tide). `bin_obs_by_time` for multi-sat consensus. |
| `detect.py` | Coherent-event detection. 8 knobs, all in `DetectionConfig`. |
| `plots.py` | All visualization. Loads from the cached parquets; doesn't re-run analysis. |
| `main.py` | Top-level driver. Edit constants → click Run. |

---

## Working conventions (do these unprompted)

- **Module-level constants, never argparse.** The user runs everything
  from VS Code Run. CLI args break that workflow.
- **Use the `gnssir` conda env** for all `Bash` invocations.
- **Edit per-day parquet caches with care.** Stage 2 (windowed) takes
  ~30–60 min for the full year. Don't `--force` casually.
- **Keep `data/` out of git.** Everything under `data/` (raw RINEX, SNR
  files, parquets, plots, sso_tokens.json) is gitignored. Don't bring
  any of it into the source tree.
- **Don't suggest microservices, async runtime, or web dashboards.** This
  is a researcher's local pipeline. Keep it that way.

---

## Current tuning (as of latest run)

### `config.py`
| knob | value | what it does |
|---|---|---|
| `WINDOW_SEC` | 180 | Lomb-Scargle window length. Was 300; lowered to catch ~30 s wave events |
| `STRIDE_SEC` | 30 | Window stride |
| `RH_MIN, RH_MAX` | 6, 14 m | Water-level search range |
| `ANTENNA_MSL_M` | 8.88 | User-calibrated. Used to convert RH ↔ water level |
| `ENABLED_SIGNALS` | 8 | GPS L1/L2/L5, Galileo E1/E5a/E5b, **GLONASS G1/G2** (nominal FDMA freqs — see below) |

### `estimate.TideKalmanConfig` (5 knobs)
| knob | value | role |
|---|---|---|
| `sigma_p` | 5e-3 m/√s | random-walk process noise on η |
| `gate_threshold` | 9.0 | mahal² cutoff for KF outlier rejection |
| `sigma_inflation_m` | 0.15 | adds to obs σ — absorbs cross-sat scatter the per-window σ misses |
| `init_eta_sigma` | 1.0 m | wide initial uncertainty so filter doesn't lock on early outlier |
| `antenna_msl_m` | from config | converts RH → η |

### `detect.DetectionConfig` (8 knobs)
| knob | value | role |
|---|---|---|
| `bin_sec` | 60 | time bin for clustering anomalous obs |
| `mahal_threshold` | 2.5 | per-obs anomaly flag (in sigmas) |
| `min_sats_per_bin` | 3 | spatial coherence: need ≥3 sats agreeing |
| `coherence_ratio_min` | 2.5 | within dominant cluster: |median|/std ratio |
| `min_amplitude_m` | 1.5 | hard floor — "only wave-class events" |
| `snr_min` | 2.0 | event must be N× rolling 30-min innov std |
| `max_event_gap_sec` | 120 | merge nearby candidate bins into one event |

The two amplitude knobs combined currently mean: detect events where many
sats coherently see ≥1.5 m AND ≥2× the local noise. For typical local noise
~80 cm, this means ≥2 m events make it through.

---

## Key design decisions and rationale

### Why random-walk KF instead of constant-velocity
At sub-arc cadence (seconds–minutes), any inferred velocity is dominated by
noise. The previous constant-velocity model would build spurious velocity
estimates and run away — KF state diverging by meters in minutes, then
gating all the *correct* obs as outliers. The 1-D position-only random walk
has no velocity to corrupt.

### Why multi-sat consensus binning before the KF
Per-window obs have ~50 cm σ. Binning by time (default 120 s — see
`main.BIN_SEC`) and taking the inverse-variance weighted mean across all
sats/bands in the bin tightens σ to ~30 cm. The KF sees a much cleaner
signal; events still pop because they're spatially coherent (median
survives outlier rejection).

### Why detect events on raw obs, not on KF state
The KF state is smoothed — events get partially absorbed. The detector
walks the **raw** windowed obs, computing each obs's innovation against
the smoothed state, and looks for coherent multi-sat clusters of large
innovations. The raw stream preserves event amplitude that the state
smooths over.

### Why the adaptive SNR gate
Background noise varies massively: ~30 cm in calm summer, ~140 cm in
fractured sea ice. A fixed amplitude threshold either floods the event
log in winter (too low) or misses real events in summer (too high).
"Event > N× local rolling-std" auto-scales.

### Why we re-added `min_amplitude_m` after consolidating gates
The adaptive SNR gate alone has a failure mode: if local_noise is
abnormally low (e.g., short calm window), even small events trigger.
For "wave-class only" use, a hard amplitude floor in meters is the
cleanest way to express user intent.

### Why GLONASS is enabled despite FDMA caveat
+40% more obs/day from ~19 satellites. The per-sat bias from using
nominal channel-0 frequencies is ~3 cm at typical RH — well below
per-window σ of ~50 cm. For sub-cm work you'd read channel numbers
from a broadcast nav file; for wave detection it doesn't matter.

---

## Known challenges and open questions

### Sea ice conditions corrupt the signal
At UMNQ in winter, the fjord fills with sea ice. The reflecting "surface"
becomes a fractured, mobile patchwork of ice + water + snow. RH retrievals
vary by 1–3 m across satellites at the same instant. The current detector
correctly produces near-zero events during these periods (adaptive noise
gate suppresses). But the underlying water level becomes unobservable —
the system can't gauge water through ice.

### No ground-truth validation
We've never tested the detector against a known event. The 254 events
flagged on the year-long dataset are unverified. Some look plausibly
event-like (sustained multi-bin, 2 m+ amplitude, spatial coherence) but
none have been confirmed against tide gauges, camera observations, or
weather records.

**Next obvious step:** find a known summer storm or surge event at a
nearby Greenland tide gauge and check whether the detector flagged it.

### Detection sensitivity is hard to tune blindly
Without a known event to calibrate against, the threshold knobs are
guesses. Current settings target 2 m+ events; if real waves are smaller
or larger we won't know until we have ground truth.

### Per-arc analysis is partially vestigial
`pipeline.process_folder` still runs full-arc analysis on every day (stage
1 of `main.py`). Its outputs aren't consumed by the rest of the pipeline
anymore — only the `snr66` files it creates as a side effect of calling
`rinex2snr` matter downstream. Could be simplified by replacing stage 1
with a thin "ensure snr66 exists" loop.

### plots.py auto-detects "most recent" range
If you have multiple range/* dirs from different runs, plots.py picks
the most recently modified one. Could surprise the user. If we add a
results browser later this gets cleaner.

### The 120 s bin width is a fixed compromise
30 s bins are mostly singletons (no consensus benefit). 600 s bins
average events away. 120 s gives ~3 obs/bin in current data. If we
change `STRIDE_SEC` (now 30 s vs old 60 s), more obs per bin →
binning gets more powerful, possibly time to revisit.

### Real-time runtime not built
Everything is batch replay from disk. The clean separation between
`snr.process_arcs_windowed` (per-arc, called as needed) and `estimate.run_batch`
(stateful, processes obs in order) means streaming should be a thin wrapper
around the same kernels. Not yet implemented.

---

## What NOT to do

| Don't | Why |
|---|---|
| Add CLI argparse to any entrypoint | User runs from VS Code Run button |
| Force-rerun stages without permission | Stage 2 takes 30–60 min for full year |
| Suggest re-architecting around microservices, async, or web UIs | Out of scope; researcher's local tool |
| Re-enable removed dead code (CV Kalman, per-arc consensus) | Was deliberately deleted for clarity |
| Add knobs without removing equivalent ones | The user explicitly wants fewer tunables |
| Touch `data/` from git | Gitignored. Includes EarthScope SSO token |
| Use the system `python3` | Project depends on `gnssir` conda env |
| Install pyarrow via conda | Binary incompatibility with `libprotobuf` on macOS arm64. Use `pip install pyarrow` |

---

## Station notes — UMNQ (Uummannaq, west Greenland)

| | |
|---|---|
| Latitude | 70.677526° N |
| Longitude | −52.115415° E |
| Antenna ellipsoidal height | 38.00 m |
| Antenna height above MSL | 8.88 m (user-calibrated) |
| Azimuth wedge (fjord-facing) | 30°–180° |
| Constellations tracked | GPS, Galileo, GLONASS (no BeiDou, no QZSS — verified across full year) |
| Typical conditions | Open water summer (May–Oct), variable to fractured ice (Nov–Apr) |

---

## Status checklist

- [x] `tide.py` — Gr1kmTM wrapper
- [x] `config.py` — multi-constellation signal registry
- [x] `snr.py` — per-arc + windowed Lomb-Scargle with σ
- [x] `pipeline.py` — batch RINEX → per-day parquets
- [x] `estimate.py` — random-walk Kalman filter, binning, multi-sat consensus
- [x] `detect.py` — coherent-event detection with 8 knobs
- [x] `plots.py` — water level overlay, residual, per-event window plots
- [x] `main.py` — single-entry orchestrator
- [ ] Validation against known events
- [ ] Streaming / real-time runtime
- [ ] Multi-station support
