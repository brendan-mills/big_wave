"""Greenland 1 km tide model interface for SSiSLS.

Wraps pyTMD's OTIS reader around the local Gr1kmTM model files. The model is
loaded once per `GreenlandTideModel` instance, the harmonic constants are
interpolated to the station location, and subsequent `predict()` calls only
do the constituent sum — fast enough for the real-time Kalman loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD
import pyproj
import xarray as xr


GR1KMTM_DIR = Path('/Users/brmills/Documents/SSiSLS/Gr1kTM')

# Polar stereographic projection used by the Gr1kmTM grid (see xy_ll_Gr1kmTM.m)
_GR1KMTM_CRS = {
    'proj': 'stere', 'datum': 'WGS84', 'type': 'crs',
    'lat_0': 90, 'lat_ts': 70, 'lon_0': -45,
    'x_0': 0, 'y_0': 0, 'units': 'km',
}

# pyTMD's time_series uses days since 1992-01-01 (Modified TIDE epoch)
_TIDE_EPOCH = datetime(1992, 1, 1, tzinfo=timezone.utc)


def _to_days_since_epoch(times) -> np.ndarray:
    """Convert any reasonable time input to float days since 1992-01-01 UTC."""
    idx = pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(times), utc=True))
    dt64 = idx.tz_convert('UTC').tz_localize(None).to_numpy()  # naive ns
    epoch64 = np.datetime64(_TIDE_EPOCH.replace(tzinfo=None), 'ns')
    return (dt64 - epoch64) / np.timedelta64(1, 'D')


@dataclass
class GreenlandTideModel:
    """Tide model bound to a single station location.

    Load once, predict many times. Holds the point-interpolated harmonic
    constants in `self.point` (an xarray.Dataset with no spatial dims).
    """

    lat: float
    lon: float
    model_dir: Path = GR1KMTM_DIR
    method: str = 'nearest'  # 'nearest' is robust near coasts; 'linear' if open ocean
    max_distance_km: float = 5.0  # warn if nearest ocean cell is further than this

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self._proj = pyproj.Proj(**{k: v for k, v in _GR1KMTM_CRS.items()
                                    if k not in ('type',)})
        self.x_km, self.y_km = self._proj(self.lon, self.lat)

        self._grid = pyTMD.io.OTIS.open_otis_dataset(
            model_file=self.model_dir / 'h_Gr1kmTM_v1',
            grid_file=self.model_dir / 'grid_Gr1kmTM_v1',
            group='z',
            crs=_GR1KMTM_CRS,
        )

        self.point = self._grid.interp(x=self.x_km, y=self.y_km, method=self.method)
        self.cell_x_km = float(self.point.x)
        self.cell_y_km = float(self.point.y)
        self.cell_distance_km = float(np.hypot(
            self.cell_x_km - self.x_km, self.cell_y_km - self.y_km))

        if not self._point_is_ocean():
            raise ValueError(
                f'No ocean cell within {self.method} interp at '
                f'({self.lat:.4f}, {self.lon:.4f}) -> ({self.x_km:.2f}, {self.y_km:.2f}) km. '
                f'Station may be inland or grid mask is wrong.')

        if self.cell_distance_km > self.max_distance_km:
            import warnings
            warnings.warn(
                f'Nearest tide-model ocean cell is {self.cell_distance_km:.2f} km '
                f'from station — predictions may be inaccurate.')

    def _point_is_ocean(self) -> bool:
        mask = float(self.point.mask)
        return mask > 0 and np.isfinite(complex(self.point.m2.item()))

    @property
    def constituent_names(self) -> list[str]:
        """Major constituents stored in the model (e.g. m2, s2, k1, o1, n2, p1, k2, q1)."""
        return [v for v in self._grid.data_vars
                if v not in ('bathymetry', 'mask')]

    def constituents(self) -> dict[str, complex]:
        """Complex tidal amplitudes (meters) at the station, keyed by constituent."""
        return {name: complex(self.point[name].item())
                for name in self.constituent_names}

    def constituent_table(self) -> pd.DataFrame:
        """Amplitude (cm) and Greenwich phase (deg) per constituent."""
        rows = []
        for name, z in self.constituents().items():
            rows.append({
                'constituent': name.upper(),
                'amplitude_cm': abs(z) * 100,
                'phase_deg':    (np.degrees(np.angle(z))) % 360,
            })
        return pd.DataFrame(rows).sort_values('amplitude_cm', ascending=False).reset_index(drop=True)

    def predict(self, times, infer_minor: bool = True) -> np.ndarray:
        """Predict tide elevation (meters) at the station for the given times.

        Parameters
        ----------
        times : datetime, np.datetime64 array, pd.DatetimeIndex, or list
            Times in UTC. Naive datetimes are assumed UTC.
        infer_minor : bool
            Add the standard set of inferred minor constituents (default True).

        Returns
        -------
        np.ndarray of float, same length as `times`. Tide elevation in meters,
        relative to the model's vertical datum (mean sea level for Gr1kmTM).
        """
        t_days = _to_days_since_epoch(times)
        major = pyTMD.predict.time_series(t_days, self.point)
        if infer_minor:
            minor = pyTMD.predict.infer_minor(t_days, self.point)
            return np.asarray(major + minor)
        return np.asarray(major)

    def predict_range(self, start, end, step_sec: float = 60.0,
                      infer_minor: bool = True) -> pd.Series:
        """Evenly-spaced tide prediction. Returns a pandas Series indexed by UTC time."""
        idx = pd.date_range(start=start, end=end, freq=f'{int(step_sec)}s', tz='UTC')
        return pd.Series(self.predict(idx, infer_minor=infer_minor), index=idx, name='tide_m')

    def plot(self, start, end, step_sec: float = 300.0, *,
             ax=None, mark_extrema: bool = True, label: str | None = None,
             **plot_kwargs):
        """Plot a tide prediction over [start, end]. Returns (fig, ax).

        Parameters
        ----------
        start, end : anything pandas.Timestamp accepts (str, datetime, …) in UTC.
        step_sec   : sample spacing in seconds (default 5 min).
        ax         : existing matplotlib Axes; new figure created if None.
        mark_extrema : annotate high- and low-water peaks (default True).
        label      : line label (default auto from station coords).
        **plot_kwargs : forwarded to `ax.plot`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

        series = self.predict_range(start, end, step_sec=step_sec)

        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 4))
        else:
            fig = ax.figure

        if label is None:
            label = f'Gr1kmTM @ ({self.lat:.3f}, {self.lon:.3f})'
        ax.plot(series.index, series.values, label=label, **plot_kwargs)

        if mark_extrema:
            _mark_extrema(ax, series)

        ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
        ax.set_ylabel('Tide elevation (m, rel. MSL)')
        ax.set_xlabel('Time (UTC)')
        loc = AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout()
        return fig, ax

    def plot_constituents(self, *, ax=None, top_n: int | None = None):
        """Bar chart of constituent amplitudes (cm) at the station. Returns (fig, ax)."""
        import matplotlib.pyplot as plt

        table = self.constituent_table()
        if top_n is not None:
            table = table.head(top_n)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 3.5))
        else:
            fig = ax.figure

        bars = ax.bar(table.constituent, table.amplitude_cm,
                      color='C0', edgecolor='k', linewidth=0.5)
        for b, amp, phase in zip(bars, table.amplitude_cm, table.phase_deg):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f'{phase:.0f}°', ha='center', va='bottom', fontsize=8, color='0.3')
        ax.set_ylabel('Amplitude (cm)')
        ax.set_title(f'Gr1kmTM constituents at ({self.lat:.3f}, {self.lon:.3f})'
                     f'  — phase labels in degrees')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        return fig, ax


def _mark_extrema(ax, series: pd.Series, min_separation_hr: float = 3.0):
    """Annotate local maxima/minima on a tide series."""
    from scipy.signal import find_peaks
    dt_sec = (series.index[1] - series.index[0]).total_seconds()
    distance = max(1, int(min_separation_hr * 3600 / dt_sec))
    y = series.values
    hi, _ = find_peaks(y,  distance=distance)
    lo, _ = find_peaks(-y, distance=distance)
    ax.plot(series.index[hi], y[hi], 'v', color='C3', ms=6, label='high water')
    ax.plot(series.index[lo], y[lo], '^', color='C2', ms=6, label='low water')


def predict_tide(lat: float, lon: float, times, model_dir: Path = GR1KMTM_DIR) -> np.ndarray:
    """One-shot convenience wrapper. For repeated predictions instantiate
    `GreenlandTideModel` directly to avoid reloading the grid each call."""
    return GreenlandTideModel(lat=lat, lon=lon, model_dir=model_dir).predict(times)


if __name__ == '__main__':
    # Smoke test: UMNQ, 2026-01-01 (24 hr window)
    UMNQ = dict(lat=70.677526, lon=-52.115415)
    START = '2026-01-01T00:00'
    END   = '2026-01-02T00:00'

    tm = GreenlandTideModel(**UMNQ)
    print(f'Station ({UMNQ["lat"]:.4f}, {UMNQ["lon"]:.4f}) projects to '
          f'({tm.x_km:.2f}, {tm.y_km:.2f}) km')
    print(f'Nearest ocean cell at ({tm.cell_x_km:.2f}, {tm.cell_y_km:.2f}) km '
          f'(distance {tm.cell_distance_km:.2f} km)')
    print()
    print('Harmonic constants at station:')
    print(tm.constituent_table().to_string(index=False))
    print()

    series = tm.predict_range(START, END, step_sec=3600)
    print(f'Hourly tide prediction (m), {START} -> {END}:')
    for t, v in series.items():
        print(f'  {t.strftime("%Y-%m-%d %H:%M")}  {v:+.3f}')
    print()
    print(f'Range: {series.min():+.3f}  to  {series.max():+.3f}  m  '
          f'(peak-to-peak {series.max()-series.min():.3f} m)')

    # save plots so we can eyeball them after the smoke test
    import matplotlib
    matplotlib.use('Agg')  # headless backend for the smoke test
    import config as _c
    out_dir = _c.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = tm.plot(START, END, step_sec=300)
    fig.suptitle(f'UMNQ tide — {START[:10]}', y=1.02)
    fig.savefig(out_dir / 'tide_umnq.png', dpi=120, bbox_inches='tight')

    fig, _ = tm.plot_constituents()
    fig.savefig(out_dir / 'tide_umnq_constituents.png', dpi=120, bbox_inches='tight')

    print(f'\nPlots written to {out_dir}/')
