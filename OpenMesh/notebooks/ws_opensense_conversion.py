"""
ws_opensense_conversion.py
==========================
Convert weather station data (ASOS, PWS, or custom) to OpenSense-format
xr.Dataset and save as compressed NetCDF.

Supported sources
-----------------
  "asos"   — NOAA ASOS 1-minute data fetched via IEM API
  "custom" — any dict of {station_id: pd.DataFrame} with a time index

Pipeline
--------
  1. fetch_asos_stations_nyc()          discover stations
  2. fetch_asos_raw()                   fetch + process to metric DataFrame
  3. to_opensense_dataset()             convert to xr.Dataset (OpenSense v1.0)
  4. save_opensense_dataset()           save as compressed NetCDF

OpenSense output format
-----------------------
  dims    : (id, time)
  coords  : id, time, lat, lon, elev
  data_vars: variable names depend on source / variables requested
  attrs   : title, source, conventions, time_range, start_date, end_date
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    class _tqdm:
        def __init__(self, total=None, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR        = Path(__file__).parent / "data"   # absolute — stable regardless of CWD
SAMPLE_DIR      = DATA_DIR / "samples"
ASOS_META_PATH  = DATA_DIR / "meta" / "ASOS_stations.csv"
IEM_1MIN_URL    = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
IEM_NETWORK_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"

# All variables available from the IEM 1-min API
ASOS_AVAILABLE_VARS = {
    "precip_amount":       ("precip",    "mm",       "1-min precipitation amount"),
    "precip_rate":         ("precip",    "mm/hr",    "Precipitation rate (calculated)"),
    "precip_type":         ("ptype",     "-",        "Raw ASOS precipitation type code"),
    "precip_category":     ("ptype",     "-",        "Simplified precip category"),
    "temperature":         ("tmpf",      "°C",       "Air temperature"),
    "dewpoint":            ("dwpf",      "°C",       "Dewpoint temperature"),
    "wind_speed":          ("sknt",      "m/s",      "Wind speed"),
    "wind_direction":      ("drct",      "degrees",  "Wind direction"),
    "wind_gust":           ("gust_sknt", "m/s",      "Wind gust speed"),
    "wind_gust_direction": ("gust_drct", "degrees",  "Wind gust direction"),
}

# Precipitation type mapping (METAR-based)
PTYPE_MAP = {
    "NP": "dry",
    "R": "rain",  "R+": "rain",  "R-": "rain",
    "S": "snow",  "S+": "snow",  "S-": "snow",
    "P": "precip", "P?": "precip",
    "M": "missing", "M ": "missing",
    "?0": "missing", "?1": "missing", "?2": "missing", "?3": "missing",
}


# ---------------------------------------------------------------------------
# Station discovery
# ---------------------------------------------------------------------------

def fetch_asos_stations_nyc(
    lat_min: float = 40.4,
    lat_max: float = 41.2,
    lon_min: float = -74.5,
    lon_max: float = -73.0,
    networks: list[str] = ("NY_ASOS", "NJ_ASOS", "CT_ASOS"),
    save_path: Union[str, Path, None] = ASOS_META_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch all ASOS stations within a bounding box from multiple state networks.

    Queries the IEM GeoJSON API for each network and filters by bbox.
    Default bbox covers NYC metro: Manhattan, Brooklyn, Queens, Bronx,
    Staten Island, Newark (NJ/KEWR), JFK as the furthest east point.

    Parameters
    ----------
    lat_min, lat_max : float   Latitude bounds  (default 40.4 – 41.0)
    lon_min, lon_max : float   Longitude bounds (default -74.5 – -73.5)
    networks : list of str     IEM network codes to query
    save_path : path-like or None   Where to save CSV (None = skip)
    verbose : bool

    Returns
    -------
    pd.DataFrame  indexed by Station ID, columns: Name, Latitude, Longitude,
                  Elevation, Network
    """
    records = []

    for network in networks:
        url = IEM_NETWORK_URL.format(network=network)
        if verbose:
            print(f"  Querying {network} ...", end=" ")
        try:
            resp     = requests.get(url, timeout=30)
            resp.raise_for_status()
            n_found  = 0
            for f in resp.json()["features"]:
                lon = f["geometry"]["coordinates"][0]
                lat = f["geometry"]["coordinates"][1]
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    p = f["properties"]
                    records.append({
                        "Station ID": p["sid"],
                        "Name":       p["sname"],
                        "Latitude":   lat,
                        "Longitude":  lon,
                        "Elevation":  p.get("elevation", np.nan),
                        "Network":    network,
                    })
                    n_found += 1
            if verbose:
                print(f"{n_found} stations found")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    if not records:
        raise RuntimeError(
            f"No ASOS stations found in bbox "
            f"lat=[{lat_min},{lat_max}] lon=[{lon_min},{lon_max}]"
        )

    df = (
        pd.DataFrame(records)
        .set_index("Station ID")
        .sort_values(["Network", "Latitude"], ascending=[True, False])
    )
    df = df[~df.index.duplicated(keep="first")]

    if verbose:
        print(f"\n  Total : {len(df)} stations")
        print(f"  {'ID':<8} {'Network':<10} {'Name':<35} {'Lat':>7} {'Lon':>8} {'Elev':>6}")
        print(f"  {'─'*72}")
        for sid, row in df.iterrows():
            print(
                f"  {sid:<8} {row['Network']:<10} {row['Name']:<35} "
                f"{row['Latitude']:>7.3f} {row['Longitude']:>8.3f} {row['Elevation']:>6.1f}"
            )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=True, index_label="Station ID")
        if verbose:
            print(f"\n  Saved : {save_path}")

    return df


def load_asos_metadata(path: Union[str, Path] = ASOS_META_PATH) -> pd.DataFrame:
    """Load ASOS station metadata CSV indexed by Station ID."""
    return pd.read_csv(Path(path), index_col="Station ID")


# ---------------------------------------------------------------------------
# ASOS fetch (IEM 1-min API)
# ---------------------------------------------------------------------------

def _map_precip_category(ptype) -> str:
    if pd.isna(ptype) or str(ptype).strip() in ("", "nan", "None"):
        return "missing"
    s = str(ptype).strip()
    if s in PTYPE_MAP:
        return PTYPE_MAP[s]
    u = s.upper()
    if u == "NP":           return "dry"
    if u.startswith("R"):   return "rain"
    if u.startswith("S"):   return "snow"
    if u.startswith("P"):   return "precip"
    return "missing"


def _raw_to_metric(df_raw: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Convert raw IEM DataFrame to metric units with standardized columns."""
    out = pd.DataFrame()
    out["datetime"]   = pd.to_datetime(df_raw["valid(UTC)"])
    out["station_id"] = station_id

    def _num(col): return pd.to_numeric(df_raw[col], errors="coerce") if col in df_raw else None

    if "tmpf"     in df_raw: out["temperature"]         = (_num("tmpf") - 32) * 5 / 9
    if "dwpf"     in df_raw: out["dewpoint"]             = (_num("dwpf") - 32) * 5 / 9
    if "sknt"     in df_raw: out["wind_speed"]           = _num("sknt") * 0.51444
    if "drct"     in df_raw: out["wind_direction"]       = _num("drct")
    if "gust_sknt"in df_raw: out["wind_gust"]            = _num("gust_sknt") * 0.51444
    if "gust_drct"in df_raw: out["wind_gust_direction"]  = _num("gust_drct")
    if "ptype"    in df_raw:
        out["precip_type"]     = df_raw["ptype"].astype(str).replace("nan", None)
        out["precip_category"] = out["precip_type"].apply(_map_precip_category)
    if "precip"   in df_raw:
        mm = _num("precip") * 25.4
        out["precip_amount"] = mm
        out["precip_rate"]   = mm * 60

    return out.sort_values("datetime").reset_index(drop=True)


def fetch_asos_raw(
    start_dt: Union[datetime, str],
    end_dt: Union[datetime, str],
    stations: list[str],
    variables: list[str] = ("precip_amount", "precip_type"),
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch and process ASOS 1-minute data for multiple stations.

    Parameters
    ----------
    stations : list of str   Station IDs, e.g. ["KJFK", "KEWR"]
    start_dt, end_dt : datetime or ISO-8601 str (e.g. "2024-01-15")
    variables : list of str
        Output variable names (from ASOS_AVAILABLE_VARS). Controls which
        API fields are requested and which columns appear in the output.
        Default: precipitation only.
        Full set: all keys of ASOS_AVAILABLE_VARS.
    verbose : bool

    Returns
    -------
    dict  station_id → pd.DataFrame  (time-indexed, metric units)
    """
    if isinstance(start_dt, str):
        start_dt = datetime.fromisoformat(start_dt)
    if isinstance(end_dt, str):
        end_dt = datetime.fromisoformat(end_dt)
    # Map requested output vars → required API fields
    api_vars = set()
    for v in variables:
        if v in ASOS_AVAILABLE_VARS:
            api_vars.add(ASOS_AVAILABLE_VARS[v][0])

    api_var_str = ",".join(sorted(api_vars))
    results: dict[str, pd.DataFrame] = {}

    for sid in stations:
        if verbose:
            print(f"  Fetching {sid} ...", end=" ")
        try:
            params = {
                "station": sid,
                "tz":      "UTC",
                "year1": start_dt.year,  "month1": start_dt.month,  "day1": start_dt.day,
                "year2": end_dt.year,    "month2": end_dt.month,    "day2": end_dt.day,
                "vars":   api_var_str,
                "sample": "1min",
                "what":   "download",
                "delim":  "comma",
            }
            resp = requests.get(IEM_1MIN_URL, params=params, timeout=300)
            resp.raise_for_status()

            if len(resp.text) < 100:
                if verbose: print("no data")
                continue

            df_raw = pd.read_csv(StringIO(resp.text))
            df     = _raw_to_metric(df_raw, sid)

            # Keep only requested output columns
            keep = ["datetime", "station_id"] + [v for v in variables if v in df.columns]
            df   = df[[c for c in keep if c in df.columns]]
            df   = df.set_index("datetime").sort_index()
            df   = df[~df.index.duplicated(keep="first")]

            results[sid] = df
            if verbose:
                print(f"{len(df):,} records")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# OpenSense conversion
# ---------------------------------------------------------------------------

def to_opensense_dataset(
    data: dict[str, pd.DataFrame],
    source: str = "asos",
    variables: Optional[list[str]] = None,
    meta: Optional[pd.DataFrame] = None,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    elev_col: str = "Elevation",
    start_dt: Optional[Union[datetime, str]] = None,
    end_dt: Optional[Union[datetime, str]] = None,
    extra_attrs: Optional[dict] = None,
) -> xr.Dataset:
    """Convert a dict of station DataFrames to an OpenSense-format xr.Dataset.

    Parameters
    ----------
    data : dict  station_id → time-indexed pd.DataFrame
    source : str  label for metadata, e.g. "asos", "pws", "custom"
    variables : list of str or None
        Which DataFrame columns to include. None = all numeric columns.
    meta : pd.DataFrame or None
        Station metadata indexed by station ID (lat/lon/elev).
    lat_col, lon_col, elev_col : str  column names in meta
    start_dt, end_dt : datetime or None  override time range in attrs
    extra_attrs : dict or None  additional global attributes

    Returns
    -------
    xr.Dataset  dims (id, time), coords (id, time, lat, lon, elev)
    """
    if isinstance(start_dt, str):
        start_dt = datetime.fromisoformat(start_dt)
    if isinstance(end_dt, str):
        end_dt = datetime.fromisoformat(end_dt)

    if not data:
        raise ValueError("data dict is empty.")

    all_times = sorted(set().union(*[df.index for df in data.values()]))
    ids, lats, lons, elevs = [], [], [], []
    var_arrays: dict[str, list] = {}

    for sid, df in data.items():
        ids.append(sid)

        # coordinates from metadata
        if meta is not None and sid in meta.index:
            lats.append(float(meta.loc[sid, lat_col])   if lat_col   in meta.columns else np.nan)
            lons.append(float(meta.loc[sid, lon_col])   if lon_col   in meta.columns else np.nan)
            elevs.append(float(meta.loc[sid, elev_col]) if elev_col  in meta.columns else np.nan)
        else:
            lats.append(np.nan); lons.append(np.nan); elevs.append(np.nan)

        # determine which variables to use
        cols = variables if variables is not None else [
            c for c in df.columns
            if c not in ("station_id",) and pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in cols:
            if col not in var_arrays:
                var_arrays[col] = []
            row = df[col].reindex(all_times).values if col in df.columns else np.full(len(all_times), np.nan)
            var_arrays[col].append(row)

    t0 = start_dt or pd.to_datetime(all_times[0])
    t1 = end_dt   or pd.to_datetime(all_times[-1])

    # Build variable metadata from ASOS catalogue if available
    def _var_attrs(col):
        if col in ASOS_AVAILABLE_VARS:
            _, unit, desc = ASOS_AVAILABLE_VARS[col]
            return {"units": unit, "long_name": desc}
        return {}

    data_vars = {
        col: (["id", "time"], np.vstack(rows), _var_attrs(col))
        for col, rows in var_arrays.items()
    }

    attrs = {
        "title":       f"OpenMesh {source.upper()} sample (multi-station)",
        "source":      source,
        "start_date":  t0.strftime("%Y-%m-%d"),
        "end_date":    t1.strftime("%Y-%m-%d"),
        "time_range":  f"{t0.isoformat()} / {t1.isoformat()}",
        "conventions": "OpenSense v1.0",
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    return xr.Dataset(
        data_vars,
        coords={
            "id":   ("id",   ids),
            "time": ("time", all_times),
            "lat":  ("id",   lats,  {"units": "degrees_north"}),
            "lon":  ("id",   lons,  {"units": "degrees_east"}),
            "elev": ("id",   elevs, {"units": "m"}),
        },
        attrs=attrs,
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _compress_encoding(ds: xr.Dataset, level: int = 4) -> dict:
    return {
        name: {"zlib": True, "complevel": level}
        for name, var in ds.data_vars.items()
        if var.dtype.kind in {"f", "i", "u"}
    }


def save_opensense_dataset(
    ds: xr.Dataset,
    output_dir: Union[str, Path] = SAMPLE_DIR,
    filename: Optional[str] = None,
    compression_level: int = 4,
    replace: bool = False,
    verbose: bool = True,
) -> Path:
    """Save an OpenSense xr.Dataset to a compressed NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset
    output_dir : path-like
    filename : str or None   auto-generated from attrs if None
    compression_level : int
    verbose : bool

    Returns
    -------
    Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        source    = ds.attrs.get("source", "ws").upper()
        start_str = ds.attrs.get("start_date", "unknown")
        end_str   = ds.attrs.get("end_date",   "unknown")
        filename  = f"{source}_{start_str}_{end_str}.nc"

    dst_path = output_dir / filename
    if dst_path.exists():
        if not replace:
            if verbose:
                print(f"  Exists : {dst_path.name}  (skipped — pass replace=True to overwrite)")
            return dst_path
        dst_path.unlink()

    ds.to_netcdf(dst_path, encoding=_compress_encoding(ds, compression_level), engine="netcdf4")

    if verbose:
        print(f"  Saved : {dst_path.name}  ({dst_path.stat().st_size / 1e6:.2f} MB)")

    return dst_path


import netCDF4 as nc4
from typing import Union


def save_asos_grouped(
    data: dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    output_dir: Union[str, Path] = DATA_DIR / "raw",
    filename: str = "asos_nyc.nc",
    verbose: bool = True,
) -> Path:
    """Save ASOS data as grouped NetCDF mirroring pws_wu_os.nc structure exactly.
    One group per station, dims (id=1, time), same variable layout as PWS.
    Uses precip_amount/precip_rate instead of rainfall_* (ASOS = all precip types).
    """
    import netCDF4 as nc4

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst_path = output_dir / filename
    if dst_path.exists():
        dst_path.unlink()

    # reference epoch — use earliest timestamp across all stations
    all_times  = pd.DatetimeIndex(
        sorted(set().union(*[df.index.tolist() for df in data.values()]))
    )
    epoch      = all_times[0].strftime("%Y-%m-%d %H:%M:%S")
    time_units = f"minutes since {epoch}"

    with nc4.Dataset(str(dst_path), "w", format="NETCDF4") as root:

        # ── global attributes (mirrors pws_wu_os.nc) ─────────────
        root.title              = "NOAA ASOS 1-min Precipitation Data — NYC"
        root.file_author        = "OpenMesh"
        root.institution        = "NOAA / Iowa Environmental Mesonet (IEM)"
        root.date               = pd.Timestamp.now().strftime("%Y-%m-%d")
        root.source             = "NOAA ASOS 1-min via Iowa Environmental Mesonet"
        root.history            = f"{pd.Timestamp.now().strftime('%Y-%m-%d')}: Converted to OpenSense-PWS-1.0 netCDF4 format."
        root.naming_convention  = "OpenSense-PWS-1.0"
        root.license_restrictions = "Public domain (NOAA)"
        root.reference          = "https://github.com/OpenSenseAction/OS_data_format_conventions/blob/main/netCDF_PWS.adoc"
        root.comment            = (
            "ASOS airport stations in NYC area. "
            "precip_amount/precip_rate used instead of rainfall_* — "
            "ASOS captures all precipitation types (rain, snow, etc.). "
            "Each station stored in separate netCDF4 group. "
            "All timestamps in UTC."
        )
        root.Conventions        = "OpenSense-PWS-v1.0"

        for sid, df in data.items():
            grp = root.createGroup(sid)
            grp.createDimension("id",   1)
            grp.createDimension("time", len(df))

            # ── time ─────────────────────────────────────────────
            t_var           = grp.createVariable("time", "i8", ("time",))
            t_var.long_name = "time_utc"
            t_var.calendar  = "proleptic_gregorian"
            t_var.units     = time_units
            t_var[:]        = nc4.date2num(
                df.index.to_pydatetime(), units=time_units, calendar="proleptic_gregorian"
            )

            # ── id ───────────────────────────────────────────────
            id_var           = grp.createVariable("id", str, ("id",))
            id_var.long_name = "asos_station_identifier"
            id_var[0]        = sid

            # ── coordinates ──────────────────────────────────────
            lat  = float(meta.loc[sid, "Latitude"])  if sid in meta.index else float("nan")
            lon  = float(meta.loc[sid, "Longitude"]) if sid in meta.index else float("nan")
            elev = float(meta.loc[sid, "Elevation"]) if sid in meta.index else float("nan")

            lat_var           = grp.createVariable("lat",  "f8", ("id",), fill_value=float("nan"))
            lat_var.units     = "degrees_in_WGS84_projection"
            lat_var.long_name = "latitude"
            lat_var[0]        = lat

            lon_var           = grp.createVariable("lon",  "f8", ("id",), fill_value=float("nan"))
            lon_var.units     = "degrees_in_WGS84_projection"
            lon_var.long_name = "longitude"
            lon_var[0]        = lon

            elev_var           = grp.createVariable("elev", "f8", ("id",), fill_value=float("nan"))
            elev_var.units     = "metres_above_sea"
            elev_var.long_name = "ground_elevation_above_sea_level"
            elev_var[0]        = elev

            # ── data variables ───────────────────────────────────
            VAR_DEFS = {
                "precip_amount": {"units": "mm",          "long_name": "precip_amount_per_time_unit"},
                "precip_rate":   {"units": "mm hr-1",     "long_name": "precipitation_rate"},
                "temperature":   {"units": "degrees_celsius", "long_name": "air_temperature"},
                "wind_speed":    {"units": "ms-1",        "long_name": "wind_speed"},
                "wind_direction":{"units": "degrees",     "long_name": "wind_direction"},
            }

            for col, vattrs in VAR_DEFS.items():
                if col not in df.columns:
                    continue
                var             = grp.createVariable(
                    col, "f8", ("id", "time"), fill_value=float("nan")
                )
                var.units       = vattrs["units"]
                var.long_name   = vattrs["long_name"]
                var.coordinates = "elev lat lon"
                var[0, :]       = df[col].values.astype("float64")

    if verbose:
        print(f"  Saved grouped : {dst_path}  ({dst_path.stat().st_size / 1e6:.2f} MB)")
        with nc4.Dataset(str(dst_path), "r") as root:
            groups = list(root.groups.keys())
            first  = root.groups[groups[0]]
            print(f"  Groups        : {groups}")
            print(f"  Variables     : {list(first.variables.keys())}")
            print(f"  Dimensions    : { {k: v.size for k, v in first.dimensions.items()} }")

    return dst_path

