"""
openmesh_data_io.py
===================
Download, load, and create samples from the OpenMesh CML and PWS datasets.

Zenodo records
--------------
CML : https://zenodo.org/records/15287692
PWS : https://zenodo.org/records/17508286
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr

try:
    import netCDF4 as nc4
    _HAS_NC4 = True
except ImportError:
    _HAS_NC4 = False

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    class _tqdm:
        def __init__(self, total=None, **kw): self.n = 0; self.total = total
        def update(self, n=1): self.n += n
        def __enter__(self): return self
        def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Constants  (all paths relative to the notebook / script working directory)
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).parent / "data" / "jacoby_2025_OpenMesh"   # absolute — stable regardless of CWD
SAMPLE_DIR = Path(__file__).parent / "data" / "samples"
CML_PATH   = DATA_DIR / "raw" / "ds_openmesh.nc"
PWS_PATH   = DATA_DIR / "raw" / "pws_wu_os.nc"

INTERVAL_DURATIONS: dict[str, timedelta] = {
    "1h":  timedelta(hours=1),
    "1d":  timedelta(days=1),
    "1w":  timedelta(weeks=1),
    "1mo": timedelta(days=30),
}
# Arbitrary intervals like "10d", "2w", "6h" are also accepted by _resolve_time_window.

DEFAULT_START_TIME = "2024-01-15T00:00:00"

_ZENODO = {
    "cml": {  # Zenodo record 15287692 — CML (NYC Mesh wireless links, RSL time-series)
        "url":      "https://zenodo.org/records/15287692/files/OpenMesh.zip?download=1",
        "filename": "OpenMesh.zip",
    },
    "pws": {  # Zenodo record 17508286 — PWS (Weather Underground NYC, crowd-sourced stations)
        "url":      "https://zenodo.org/records/17508286/files/PWS_NYC_WU.zip?download=1",
        "filename": "PWS_NYC_WU.zip",
    },
}

_PWS_VARS = [
    "rainfall_rate", "rainfall_amount", "temperature",
    "relative_humidity", "wind_velocity", "wind_direction", "air_pressure",
]

# Variables that should be summed (not averaged) when resampling.
# Rates (mm/h) are averaged, only accumulations (mm) are summed.
_ACCUMULATION_VARS = {
    "rainfall_amount", "precip_amount",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rel(path: Path) -> Path:
    """Return path relative to cwd; fall back to filename only to avoid leaking home dir."""
    try:
        return path.relative_to(Path.cwd())
    except ValueError:
        return Path(path.name)


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    dest = Path(dest)
    if dest.exists():
        print(f"  Already exists : {dest.name} ({dest.stat().st_size / 1e6:.1f} MB) — skipping.")
        return dest
    print(f"  Downloading    : {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp  = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh:
        with _tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
    print(f"  Saved          : {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _resample_ds(ds: xr.Dataset, rule: str) -> xr.Dataset:
    """Resample a dataset: sum for accumulation vars, mean for all others.
    Non-time coordinates (lat, lon, elev) are restored after resampling."""
    parts = {}
    for var in ds.data_vars:
        if var in _ACCUMULATION_VARS:
            parts[var] = ds[var].resample(time=rule).sum(skipna=True)
        else:
            parts[var] = ds[var].resample(time=rule).mean(skipna=True)
    ds_r = xr.Dataset(parts)
    # Restore non-time coordinates (lat, lon, elev) dropped by resample
    for coord in ds.coords:
        if "time" not in ds[coord].dims and coord not in ds_r.coords:
            ds_r = ds_r.assign_coords({coord: ds[coord]})
    return ds_r


def _compress_encoding(ds: xr.Dataset, level: int = 4) -> dict:
    return {
        name: {"zlib": True, "complevel": level}
        for name, var in ds.data_vars.items()
        if var.dtype.kind in {"f", "i", "u"}
    }


def save_compressed_netcdf(
    ds: xr.Dataset,
    path: Union[str, Path],
    *,
    compression_level: int = 4,
    mode: str = "w",
) -> Path:
    """Write *ds* to NetCDF with zlib compression using the netCDF4 engine.

    Matches the encoding used by :func:`create_openmesh_sample`,
    :func:`create_pws_sample`, and :func:`create_asos_sample` (only float/int
    data variables are compressed; requires ``netcdf4`` installed).
    """
    path = Path(path)
    ds.to_netcdf(
        path,
        mode=mode,
        encoding=_compress_encoding(ds, compression_level),
        engine="netcdf4",
    )
    return path


def _infer_interval_label(start: datetime, end: datetime) -> str:
    total_seconds = int((end - start).total_seconds())
    if total_seconds % 3600 == 0:
        h = total_seconds // 3600
        if h % 720 == 0: return f"{h // 720}mo"
        if h % 24  == 0: return f"{h // 24}d"
        return f"{h}h"
    return f"{total_seconds // 60}min"


def _parse_interval(time_interval: str) -> timedelta:
    """Parse an interval string into a timedelta.

    Accepts the fixed shortcuts in INTERVAL_DURATIONS *and* arbitrary
    ``Nh`` / ``Nd`` / ``Nw`` / ``Nmo`` patterns (e.g. ``"10d"``, ``"48h"``).
    """
    import re
    if time_interval in INTERVAL_DURATIONS:
        return INTERVAL_DURATIONS[time_interval]
    m = re.fullmatch(r"(\d+)(h|d|w|mo)", time_interval)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit == "h":  return timedelta(hours=n)
        if unit == "d":  return timedelta(days=n)
        if unit == "w":  return timedelta(weeks=n)
        if unit == "mo": return timedelta(days=30 * n)
    raise ValueError(
        f"Unknown time_interval '{time_interval}'. "
        f"Use a fixed shortcut {list(INTERVAL_DURATIONS)} "
        f"or an arbitrary pattern like '10d', '48h', '2w', '3mo'."
    )


def _resolve_time_window(
    start_time: str,
    time_interval: Optional[str],
    end_time: Optional[str],
) -> tuple[datetime, datetime, str]:
    start_dt = datetime.fromisoformat(start_time)
    if end_time is not None:
        end_dt = datetime.fromisoformat(end_time)
        label  = _infer_interval_label(start_dt, end_dt)
    elif time_interval is not None:
        end_dt = start_dt + _parse_interval(time_interval)
        label  = time_interval
    else:
        raise ValueError("Provide either time_interval or end_time.")
    return start_dt, end_dt, label


def _resolve_source(
    src_path: Union[str, Path, None],
    default_path: Path,
    mode: str,
    download_fn,
) -> Path:
    path = Path(src_path) if src_path is not None else default_path
    if mode == "download":
        if not path.exists():
            print("  Source not found — downloading ...")
            download_fn()
        else:
            print(f"  Source   : {_rel(path)}  (exists, skipping download)")
    elif mode == "local":
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Tip: use mode='download' to fetch it automatically."
            )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'local' or 'download'.")
    return path


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_openmesh(output_dir: Union[str, Path] = DATA_DIR) -> Path:
    """Download and extract the OpenMesh CML dataset from Zenodo.
    Skips if already present locally.

    Returns
    -------
    Path  →  <output_dir>/raw/
    """
    output_dir = Path(output_dir)
    raw_dir    = output_dir / "raw"
    cml_nc     = raw_dir / "ds_openmesh.nc"

    if cml_nc.exists():
        print(f"  CML found locally  : {cml_nc}")
        return raw_dir

    zip_path = output_dir / "archived" / _ZENODO["cml"]["filename"]
    _download_file(_ZENODO["cml"]["url"], zip_path)

    ext_map  = {".nc": raw_dir, ".csv": output_dir / "meta",
                ".html": output_dir / "meta" / "maps", ".ipynb": output_dir / "examples"}
    docs_dir = output_dir / "docs"
    for d in [*ext_map.values(), docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            print(f"  Extracting {len(members)} files ...")
            zf.extractall(tmp)
        for f in Path(tmp).rglob("*"):
            if f.is_file():
                dest = ext_map.get(f.suffix.lower(), docs_dir) / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)

    print(f"  Extraction complete: {raw_dir}")
    return raw_dir


def download_pws_wu(output_dir: Union[str, Path] = DATA_DIR) -> Path:
    """Download and extract the PWS Weather Underground dataset from Zenodo.
    Skips if already present locally.

    Returns
    -------
    Path  →  <output_dir>/raw/pws_wu_os.nc
    """
    output_dir = Path(output_dir)
    pws_nc     = output_dir / "raw" / "pws_wu_os.nc"

    if pws_nc.exists():
        print(f"  PWS found locally  : {pws_nc}")
        return pws_nc

    zip_path = output_dir / "archived" / _ZENODO["pws"]["filename"]
    _download_file(_ZENODO["pws"]["url"], zip_path)

    pws_nc.parent.mkdir(parents=True, exist_ok=True)
    print("  Extracting pws_wu_os.nc ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("pws_wu_os.nc"):
                pws_nc.write_bytes(zf.read(name))
                break

    print(f"  Saved              : {_rel(pws_nc)} ({pws_nc.stat().st_size / 1e6:.1f} MB)")
    return pws_nc


# ---------------------------------------------------------------------------
# CML — load
# ---------------------------------------------------------------------------

def load_cml(path: Union[str, Path, None] = None) -> xr.Dataset:
    """Load the full CML NetCDF into memory. Defaults to CML_PATH."""
    path = Path(path) if path is not None else CML_PATH
    ds   = xr.load_dataset(path)   # load_dataset vs open_dataset — reads fully into memory
    print(f"  Loaded CML        : {dict(ds.sizes)}  ←  {_rel(path)}")
    return ds


def load_cml_sample(path: Union[str, Path]) -> xr.Dataset:
    """Load a CML sample NetCDF into an xarray Dataset."""
    path = Path(path)
    ds   = xr.open_dataset(path)
    print(f"  Loaded CML sample : {dict(ds.sizes)}  ←  {path.name}")
    return ds


# ---------------------------------------------------------------------------
# CML — sample creation
# ---------------------------------------------------------------------------

def create_openmesh_sample(
    src_path: Union[str, Path, None] = None,
    output_dir: Union[str, Path] = SAMPLE_DIR,
    *,
    ds: Optional[xr.Dataset] = None,
    mode: str = "local",
    start_time: str = DEFAULT_START_TIME,
    time_interval: Optional[str] = "1d",
    end_time: Optional[str] = None,
    cml_ids: Optional[List[str]] = None,
    compression_level: int = 4,
    resample: Optional[str] = None,
    include_date: bool = False,
    replace: bool = False,
    verbose: bool = True,
    return_type: str = "path",              # "path" | "sample"
    save: bool = True,
) -> Union[Path, xr.Dataset]:
    """Create a time- and/or CML-filtered sample from the OpenMesh dataset.

    Parameters
    ----------
    src_path : path-like or None
        Source NetCDF. Defaults to CML_PATH.
    output_dir : path-like
        Output directory. Defaults to SAMPLE_DIR (./data/samples).
    ds : xr.Dataset or None
        Already-loaded dataset — skips all file I/O when provided.
    mode : "local" | "download"
        "local"    — load from disk (raises FileNotFoundError if missing).
        "download" — auto-download from Zenodo if not found locally.
    start_time : str
        ISO-8601 window start.
    time_interval : str or None
        "1h", "1d", "1w", "1mo". Ignored when end_time is given.
    end_time : str or None
        Explicit ISO-8601 window end. Overrides time_interval.
    cml_ids : list of str or None
        CML IDs to keep. None = all 75 links.
    compression_level : int
        zlib level 1–9.
    resample : str or None
        Temporal resampling rule (e.g. ``"1h"``, ``"5min"``). RSL is averaged.
        When set, the rule is included in the filename: ``openmesh_cml_1h_1d.nc``.
    include_date : bool
        If True, append ``_YYYYMMDD_YYYYMMDD`` start/end dates to the filename
        (e.g. ``openmesh_cml_1d_20240115_20240116.nc``).
        Default False → ``openmesh_cml_1d.nc`` (canonical root copy).
    verbose : bool
    return_type : str
        "path" (default) — returns file path.
        "sample" — returns xr.Dataset.
    save : bool
        Write the sample to disk. Default True.
        When ``return_type='sample'`` and ``save=False``, returns the
        dataset without writing to disk.

    Returns
    -------
    Path or xr.Dataset
        If ``return_type='path'`` (default), returns the written sample file path.
        If ``return_type='sample'``, returns the in-memory xr.Dataset instead.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt, label = _resolve_time_window(start_time, time_interval, end_time)
    date_suffix   = f"_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}" if include_date else ""
    resample_tag  = f"_{resample}" if resample else ""
    cml_tag       = f"_cmls{len(cml_ids)}" if cml_ids is not None else ""
    dst_path      = output_dir / f"openmesh_cml{resample_tag}_{label}{date_suffix}{cml_tag}.nc"

    _owns_ds = ds is None
    if ds is None:
        resolved = _resolve_source(src_path, CML_PATH, mode, download_openmesh)
        if verbose:
            print(f"  Source   : {resolved}")
        ds = xr.open_dataset(resolved)
    else:
        if verbose:
            print("  Source   : provided ds")

    if verbose:
        print(f"  Window   : {start_dt}  →  {end_dt}  ({label})")
        print(f"  CML IDs  : {'all' if cml_ids is None else cml_ids}")
        print(f"  Output   : {dst_path.name}")

    ds_sample = ds.sel(time=slice(start_dt, end_dt))
    if cml_ids is not None:
        ds_sample = ds_sample.sel(cml_id=cml_ids)
    if resample:
        ds_sample = _resample_ds(ds_sample, resample)

    if verbose:
        print(f"  Timesteps: {ds_sample.sizes['time']:,}")
        print(f"  CML count: {ds_sample.sizes['cml_id']}")

    ds_sample.attrs.update({
        "title":      f"OpenMesh CML sample ({label}), start {start_dt.strftime('%Y-%m-%d')}",
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date":   end_dt.strftime("%Y-%m-%d"),
        "time_range": f"{start_dt.isoformat()} / {end_dt.isoformat()}",
        "cml_subset": "all" if cml_ids is None else ",".join(cml_ids),
    })

    if save:
        if dst_path.exists() and not replace:
            if verbose:
                print(f"  Exists   : {_rel(dst_path)}  (skipped — pass replace=True to overwrite)")
            return dst_path if return_type == "path" else xr.open_dataset(dst_path)
        try:
            if dst_path.exists():
                try:
                    dst_path.unlink()
                except PermissionError:
                    print(f"  WARNING: could not delete existing file {dst_path} (permission). "
                          f"Attempting to overwrite in-place.")

            try:
                save_compressed_netcdf(
                    ds_sample, dst_path, compression_level=compression_level
                )
            except PermissionError as e:
                print(f"  WARNING: could not write NetCDF file {dst_path} (permission): {e}")
                print("           Skipping write for this interval, downstream code should handle "
                      "missing files gracefully.")
        except Exception as e:
            print(f"  WARNING: unexpected error while writing {dst_path}: {e}")

        if verbose and dst_path.exists():
            print(f"  Saved    : {_rel(dst_path)}  ({dst_path.stat().st_size / 1e6:.2f} MB)")
    else:
        if verbose:
            print("  Save     : skipped (save=False)")

    if _owns_ds:
        ds.close()

    return ds_sample if return_type == "sample" else dst_path


def create_openmesh_samples_batch(
    intervals: List[str] = ("1h", "1d", "1w"),
    start_time: str = DEFAULT_START_TIME,
    resample: Optional[str] = None,
    **kwargs,
) -> dict[str, Path]:
    """Create multiple CML samples in one call.

    Parameters
    ----------
    intervals : list of str
    start_time : str
    **kwargs : forwarded to create_openmesh_sample

    Returns
    -------
    dict  interval → Path
    """
    results: dict[str, Path] = {}
    sep = "─" * 55

    for interval in intervals:
        if kwargs.get("verbose", True):
            print(f"\n{sep}")
            print(f"  Interval : {interval}")
            print(sep)
        results[interval] = create_openmesh_sample(
            start_time=start_time, time_interval=interval, resample=resample, **kwargs
        )

    if kwargs.get("verbose", True):
        print(f"\n{sep}")
        print("  Batch complete:")
        for label, p in results.items():
            print(f"    {label:<6} → {p.name}")
        print(sep)

    return results


# ---------------------------------------------------------------------------
# PWS — load
# ---------------------------------------------------------------------------

def load_pws_sample(path: Union[str, Path]) -> xr.Dataset:
    """Load a flat (id, time) PWS sample NetCDF into an xarray Dataset."""
    path = Path(path)
    ds   = xr.open_dataset(path)
    print(f"  Loaded PWS sample : {dict(ds.sizes)}  ←  {path.name}")
    return ds


# ---------------------------------------------------------------------------
# PWS — sample creation
# ---------------------------------------------------------------------------

def create_pws_sample(
    src_path: Union[str, Path, None] = None,
    output_dir: Union[str, Path] = SAMPLE_DIR,
    *,
    mode: str = "local",
    start_time: str = DEFAULT_START_TIME,
    time_interval: Optional[str] = "1d",
    end_time: Optional[str] = None,
    compression_level: int = 4,
    resample: Optional[str] = None,
    include_date: bool = False,
    replace: bool = False,
    verbose: bool = True,
    return_type: str = "path",              # "path" | "sample"
    save: bool = True,
) -> Union[Path, xr.Dataset]:
    """Crop PWS NetCDF (group-per-station) to a time window and save as
    a flat (id, time) compressed sample.

    Parameters
    ----------
    src_path : path-like or None
        Defaults to PWS_PATH.
    output_dir : path-like
        Defaults to SAMPLE_DIR.
    mode : "local" | "download"
    start_time, time_interval, end_time : same as create_openmesh_sample
    compression_level : int
    verbose : bool

    Returns
    -------
    Path or xr.Dataset
        If ``return_type='path'`` (default), returns the written sample file path.
        If ``return_type='sample'``, returns the in-memory xr.Dataset instead.
    """
    if not _HAS_NC4:
        raise ImportError("netCDF4 is required for PWS sampling.  pip install netCDF4")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt, label = _resolve_time_window(start_time, time_interval, end_time)
    date_suffix  = f"_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}" if include_date else ""
    resample_tag = f"_{resample}" if resample else ""
    dst_path     = output_dir / f"openmesh_wu_pws{resample_tag}_{label}{date_suffix}.nc"

    resolved = _resolve_source(src_path, PWS_PATH, mode, download_pws_wu)

    with nc4.Dataset(str(resolved), "r") as root:
        group_ids = list(root.groups.keys())

    if verbose:
        print(f"  Source   : {resolved}")
        print(f"  Window   : {start_dt}  →  {end_dt}  ({label})")
        print(f"  Stations : {len(group_ids)} groups")
        print(f"  Output   : {dst_path.name}")

    common_time: Optional[pd.DatetimeIndex] = None
    for gid in group_ids:
        try:
            ds_g = xr.open_dataset(resolved, group=gid, engine="netcdf4")
            t    = ds_g.sel(time=slice(start_dt, end_dt)).time
            if t.sizes["time"] > 0:
                common_time = pd.DatetimeIndex(t.values)
                ds_g.close()
                break
            ds_g.close()
        except Exception:
            pass

    if common_time is None:
        raise RuntimeError(f"No PWS data found between {start_dt} and {end_dt}.")

    n_times = len(common_time)
    ids_out, lats_out, lons_out, elevs_out = [], [], [], []
    var_rows: dict[str, list] = {v: [] for v in _PWS_VARS}
    var_attrs: dict[str, dict] = {}
    n_stations = 0

    for gid in group_ids:
        try:
            ds_g = xr.open_dataset(resolved, group=gid, engine="netcdf4")
            ds_c = ds_g.sel(time=slice(start_dt, end_dt))
            if ds_c.sizes["time"] == 0:
                ds_g.close()
                continue

            ids_out.append(gid)
            lats_out.append(float(ds_c["lat"].values.flat[0])   if "lat"  in ds_c else np.nan)
            lons_out.append(float(ds_c["lon"].values.flat[0])   if "lon"  in ds_c else np.nan)
            elevs_out.append(float(ds_c["elev"].values.flat[0]) if "elev" in ds_c else np.nan)

            for vname in _PWS_VARS:
                if vname not in ds_c:
                    var_rows[vname].append(np.full(n_times, np.nan))
                    continue
                if not var_attrs.get(vname):
                    var_attrs[vname] = dict(ds_c[vname].attrs)
                arr = ds_c[vname].values.squeeze()
                s   = pd.Series(arr, index=pd.DatetimeIndex(ds_c.time.values))
                var_rows[vname].append(s[~s.index.duplicated()].reindex(common_time).values)

            n_stations += 1
            ds_g.close()
        except Exception as e:
            warnings.warn(f"Skipping PWS group '{gid}': {e}")

    if n_stations == 0:
        raise RuntimeError(f"No PWS stations had data between {start_dt} and {end_dt}.")

    if verbose:
        print(f"  Loaded   : {n_stations} stations, {n_times:,} timesteps")

    ds_out = xr.Dataset(
        {v: (["id", "time"], np.vstack(var_rows[v]), var_attrs.get(v, {}))
         for v in _PWS_VARS if var_rows[v]},
        coords={
            "id":   (["id"],   ids_out),
            "time": (["time"], common_time),
            "lat":  (["id"],   lats_out,  {"units": "degrees_north"}),
            "lon":  (["id"],   lons_out,  {"units": "degrees_east"}),
            "elev": (["id"],   elevs_out, {"units": "m"}),
        },
        attrs={
            "title":       f"OpenMesh PWS sample ({label}), start {start_dt.strftime('%Y-%m-%d')}",
            "start_date":  start_dt.strftime("%Y-%m-%d"),
            "end_date":    end_dt.strftime("%Y-%m-%d"),
            "time_range":  f"{start_dt.isoformat()} / {end_dt.isoformat()}",
            "source":      "https://zenodo.org/records/17508286",
            "conventions": "OpenSense v1.0",
        },
    )

    if resample:
        ds_out = _resample_ds(ds_out, resample)
        if verbose:
            print(f"  Resampled: {resample} → {ds_out.sizes['time']:,} timesteps")

    if save:
        if dst_path.exists() and not replace:
            if verbose:
                print(f"  Exists   : {_rel(dst_path)}  (skipped — pass replace=True to overwrite)")
            return dst_path if return_type == "path" else xr.open_dataset(dst_path)
        # Overwrite existing files explicitly; if this fails, warn but do not raise.
        try:
            if dst_path.exists():
                try:
                    dst_path.unlink()
                except PermissionError as e:
                    print(f"  WARNING: could not delete existing file {dst_path} (permission): {e}")

            try:
                save_compressed_netcdf(
                    ds_out, dst_path, compression_level=compression_level
                )
            except PermissionError as e:
                print(f"  WARNING: could not write PWS NetCDF file {dst_path} (permission): {e}")
        except Exception as e:
            print(f"  WARNING: unexpected error while writing PWS sample {dst_path}: {e}")

        if verbose and dst_path.exists():
            print(f"  Saved    : {_rel(dst_path)}  ({dst_path.stat().st_size / 1e6:.2f} MB)")
    else:
        if verbose:
            print("  Save     : skipped (save=False)")

    return ds_out if return_type == "sample" else dst_path



# ---------------------------------------------------------------------------
# ASOS — sample creation
# ---------------------------------------------------------------------------

def create_asos_sample(
    output_dir: Union[str, Path] = SAMPLE_DIR,
    *,
    start_time: str = DEFAULT_START_TIME,
    time_interval: Optional[str] = "1d",
    end_time: Optional[str] = None,
    stations: Optional[List[str]] = None,
    variables: List[str] = ("precip_amount", "precip_rate", "temperature", "wind_speed"),
    meta_path: Union[str, Path, None] = None,
    compression_level: int = 4,
    resample: Optional[str] = None,
    include_date: bool = False,
    replace: bool = False,
    save: bool = True,
    verbose: bool = True,
    return_type: str = "path",              # "path" | "sample"
) -> Union[Path, xr.Dataset]:
    """Fetch ASOS 1-min data from the IEM API and save as a flat (id, time)
    compressed NetCDF sample following the OpenMesh naming convention.

    Requires a live internet connection (IEM API call).

    Parameters
    ----------
    output_dir : path-like
        Output directory. Defaults to SAMPLE_DIR (data/samples).
    start_time : str
        ISO-8601 window start. Default ``2024-01-15T00:00:00``.
    time_interval : str or None
        ``"1h"``, ``"1d"``, ``"1w"``, ``"1mo"``. Ignored when end_time given.
    end_time : str or None
        Explicit ISO-8601 window end. Overrides time_interval.
    stations : list of str or None
        ASOS station IDs (e.g. ``["KJFK", "KEWR"]``).
        None → ``["JFK", "EWR", "LGA", "NYC"]`` (NYC-area defaults).
    variables : list of str
        Variable names from ``ws_opensense_conversion.ASOS_AVAILABLE_VARS``.
    meta_path : path-like or None
        Path to ASOS station metadata CSV.
        None → ``data/meta/ASOS_stations.csv`` (or fetched from IEM if missing).
    compression_level : int
        zlib level 1–9.
    include_date : bool
        If True, append ``_YYYY-MM-DD`` start date to the filename
        (e.g. ``openmesh_asos_1d_2024-01-15.nc``).
        Default False → ``openmesh_asos_1d.nc``.
    verbose : bool
    return_type : str
        ``"path"`` (default) — returns file path.
        ``"sample"`` — returns the in-memory xr.Dataset.

    Returns
    -------
    Path or xr.Dataset
    """
    import ws_opensense_conversion as wsoc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt, end_dt, label = _resolve_time_window(start_time, time_interval, end_time)
    date_suffix  = f"_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}" if include_date else ""
    resample_tag = f"_{resample}" if resample else ""
    dst_path     = output_dir / f"openmesh_asos_ws{resample_tag}_{label}{date_suffix}.nc"

    _stations = stations or ["JFK", "EWR", "LGA", "NYC"]

    if verbose:
        print(f"  Window   : {start_dt}  →  {end_dt}  ({label})")
        print(f"  Stations : {_stations}")
        print(f"  Variables: {list(variables)}")
        print(f"  Output   : {dst_path.name}")

    # Load or fetch station metadata
    _meta_path = Path(meta_path) if meta_path is not None else DATA_DIR / "meta" / "ASOS_stations.csv"
    if _meta_path.exists():
        meta = wsoc.load_asos_metadata(_meta_path)
    else:
        if verbose:
            print("  Metadata not found locally — fetching from IEM ...")
        meta = wsoc.fetch_asos_stations_nyc(save_path=_meta_path, verbose=verbose)

    # Fetch from IEM API
    asos_raw = wsoc.fetch_asos_raw(
        start_dt, end_dt, _stations,
        variables=list(variables),
        verbose=verbose,
    )
    if not asos_raw:
        raise RuntimeError(
            f"No ASOS data returned for stations {_stations} "
            f"between {start_dt} and {end_dt}."
        )

    ds_out = wsoc.to_opensense_dataset(
        asos_raw,
        source="asos",
        meta=meta,
        start_dt=start_dt,
        end_dt=end_dt,
        extra_attrs={
            "title":     f"OpenMesh ASOS sample ({label}), start {start_dt.strftime('%Y-%m-%d')}",
            "reference": "https://zenodo.org/records/15287692",
        },
    )

    if verbose:
        print(f"  Loaded   : {len(asos_raw)} stations, {ds_out.sizes['time']:,} timesteps")

    if resample:
        ds_out = _resample_ds(ds_out, resample)
        if verbose:
            print(f"  Resampled: {resample} → {ds_out.sizes['time']:,} timesteps")

    if save:
        if dst_path.exists() and not replace:
            if verbose:
                print(f"  Exists   : {_rel(dst_path)}  (skipped — pass replace=True to overwrite)")
            return dst_path if return_type == "path" else xr.open_dataset(dst_path)

        try:
            if dst_path.exists():
                dst_path.unlink()
            save_compressed_netcdf(
                ds_out, dst_path, compression_level=compression_level
            )
            if verbose:
                print(f"  Saved    : {_rel(dst_path)}  ({dst_path.stat().st_size / 1e6:.2f} MB)")
        except Exception as e:
            print(f"  WARNING: could not write ASOS sample {_rel(dst_path)}: {e}")
    else:
        if verbose:
            print("  Save     : skipped (save=False)")

    return ds_out if return_type == "sample" else dst_path


# ── create_all_datasets ───────────────────────────────────────
def create_all_datasets(
    start_time: str,
    end_time:   str,
    cml_ids:    Optional[List[str]] = None,   # None = all 75
    pws_ids:    Optional[List[str]] = None,   # None = all stations
    asos_ids:   Optional[List[str]] = None,   # None = all in meta
    output_dir: Union[str, Path] = SAMPLE_DIR,
    resample: Optional[str] = None,
    include_date: bool = False,
    replace: bool = False,
    save: bool = True,
    verbose:    bool = True,
) -> dict[str, Optional[Union[Path, "xr.Dataset"]]]:
    """Create CML, PWS and ASOS samples for a given time window.

    Parameters
    ----------
    save : bool
        If True (default), write samples to disk and return Paths.
        If False, skip disk read/write entirely — always create from
        the raw downloaded files and return xr.Dataset objects.

    Returns
    -------
    dict with keys "cml", "pws", "asos" → Path (save=True) or xr.Dataset (save=False)
    """
    from datetime import datetime
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt    = datetime.fromisoformat(start_time)
    end_dt      = datetime.fromisoformat(end_time)
    label        = _infer_interval_label(start_dt, end_dt)
    date_suffix  = f"_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}" if include_date else ""
    resample_tag = f"_{resample}" if resample else ""
    results      = {"cml": None, "pws": None, "asos": None}
    sep          = "─" * 55

    return_type = "path" if save else "sample"

    # ── CML ──────────────────────────────────────────────────
    print(f"\n{sep}\n  CML\n{sep}")
    cml_tag  = f"_cmls{len(cml_ids)}" if cml_ids else ""
    cml_path = output_dir / f"openmesh_cml{resample_tag}_{label}{date_suffix}{cml_tag}.nc"
    if save and cml_path.exists() and not replace:
        print(f"  exists  : {cml_path.name}  (skipped — pass replace=True to overwrite)")
        results["cml"] = cml_path
    else:
        try:
            results["cml"] = create_openmesh_sample(
                start_time=start_time, end_time=end_time,
                cml_ids=cml_ids, output_dir=output_dir,
                mode="local", resample=resample, include_date=include_date,
                replace=replace, save=save, return_type=return_type, verbose=verbose,
            )
        except Exception as e:
            print(f"  FAILED  : {e}")

    # ── PWS ──────────────────────────────────────────────────
    print(f"\n{sep}\n  PWS\n{sep}")
    pws_path = output_dir / f"openmesh_wu_pws{resample_tag}_{label}{date_suffix}.nc"
    if save and pws_path.exists() and not replace:
        print(f"  exists  : {pws_path.name}  (skipped — pass replace=True to overwrite)")
        results["pws"] = pws_path
    else:
        try:
            results["pws"] = create_pws_sample(
                start_time=start_time, end_time=end_time,
                output_dir=output_dir, mode="local",
                resample=resample, include_date=include_date,
                replace=replace, save=save, return_type=return_type, verbose=verbose,
            )
        except Exception as e:
            print(f"  FAILED  : {e}")

    # ── ASOS ─────────────────────────────────────────────────
    print(f"\n{sep}\n  ASOS\n{sep}")
    asos_path = output_dir / f"openmesh_asos_ws_{label}{date_suffix}.nc"
    if save and asos_path.exists() and not replace:
        print(f"  exists  : {asos_path.name}  (skipped — pass replace=True to overwrite)")
        results["asos"] = asos_path
    else:
        try:
            results["asos"] = create_asos_sample(
                output_dir=output_dir,
                start_time=start_time, end_time=end_time,
                stations=asos_ids,
                resample=resample, include_date=include_date,
                replace=replace, save=save, return_type=return_type, verbose=verbose,
            )
        except Exception as e:
            print(f"  FAILED  : {e}")

    # ── summary ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Summary:")
    for name, val in results.items():
        if val is None:
            status = "NOT created"
        elif save:
            status = val.name
        else:
            status = repr(val)
        print(f"    {name:<6} → {status}")
    print(sep)

    return results