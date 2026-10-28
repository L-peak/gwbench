"""Clients for USGS groundwater data.

Two USGS services are used:

* The legacy NWIS Site Service for site metadata and series catalogs.  This is
  still the most convenient way to ask "which wells have long daily records?".
* The modernized Water Data OGC API for observations.  The legacy
  ``/nwis/gwlevels`` endpoint was retired in November 2025; the OGC
  ``field-measurements`` and ``daily`` collections replace it.

Useful parameter codes
----------------------
* 72019 - depth to water level, feet below land surface
* 62610 - groundwater level above NGVD 1929, feet
* 62611 - groundwater level above NAVD 1988, feet
"""

from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NWIS_BASE = "https://waterservices.usgs.gov/nwis"
OGC_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

GW_LEVEL_PARAMETERS = {
    "72019": "Depth to water level, ft below land surface",
    "62610": "Groundwater level above NGVD 1929, ft",
    "62611": "Groundwater level above NAVD 1988, ft",
}


class USGSError(RuntimeError):
    """Raised when a USGS request fails after retries."""


def _request(
    url: str,
    params: dict[str, str] | None = None,
    *,
    timeout: int = 180,
    retries: int = 3,
    backoff: float = 2.0,
) -> requests.Response:
    """GET a USGS endpoint with simple exponential backoff."""
    if "api.waterdata.usgs.gov" in url:
        api_key = os.environ.get("USGS_API_KEY")
        if api_key:
            params = dict(params or {})
            params.setdefault("api_key", api_key)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            last_error = USGSError(
                f"HTTP {response.status_code} for {response.url}: {response.text[:300]}"
            )
        except requests.RequestException as exc:
            last_error = exc

        if attempt < retries:
            sleep_for = backoff * attempt
            logger.warning(
                "USGS request failed (%s/%s): %s; retrying in %.1fs",
                attempt,
                retries,
                last_error,
                sleep_for,
            )
            time.sleep(sleep_for)

    raise USGSError(
        f"USGS request failed after {retries} attempts: {url} params={params}"
    ) from last_error


# ---------------------------------------------------------------------------
# Legacy NWIS site service: metadata and series catalogs
# ---------------------------------------------------------------------------


def parse_rdb(text: str) -> pd.DataFrame:
    """Parse a USGS RDB (tab-delimited) response into a DataFrame."""
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    header = lines[0].split("\t")
    n_cols = len(header)
    rows: list[list[str]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < n_cols:
            parts = parts + [""] * (n_cols - len(parts))
        rows.append(parts[:n_cols])

    df = pd.DataFrame(rows, columns=header)
    return df.map(lambda value: value.strip() if isinstance(value, str) else value)


def fetch_gw_sites(
    state_cd: str | Sequence[str],
    *,
    site_type: str = "GW",
    has_data_type: str = "gw",
    output_data_type: str | None = None,
) -> pd.DataFrame:
    """Fetch the NWIS site file / series catalog for one or more state codes.

    Parameters
    ----------
    state_cd:
        Two-letter state code or a sequence of codes.
    site_type:
        NWIS site type.  ``"GW"`` selects groundwater wells.
    has_data_type:
        Restrict to sites that have this data type.  ``"gw"`` means discrete
        groundwater levels; ``"dv"`` means daily values.
    output_data_type:
        If set, the site service returns a series catalog (one row per
        site/parameter/statistic).  Defaults to ``has_data_type``.
    """
    states = [state_cd] if isinstance(state_cd, str) else list(state_cd)
    frames: list[pd.DataFrame] = []
    output_type = output_data_type or has_data_type

    for state in states:
        params = {
            "format": "rdb",
            "stateCd": state,
            "siteType": site_type,
            "hasDataTypeCd": has_data_type,
            "outputDataTypeCd": output_type,
        }
        response = _request(f"{NWIS_BASE}/site/", params)
        df = parse_rdb(response.text)
        if df.empty:
            logger.warning("No sites returned for state %s", state)
            continue
        df["state_cd"] = state.lower()
        frames.append(df)
        logger.info("Fetched %s rows for %s", len(df), state)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_daily_catalog(
    state_cd: str | Sequence[str], *, site_type: str = "GW"
) -> pd.DataFrame:
    """Series catalog for daily values at groundwater sites."""
    return fetch_gw_sites(
        state_cd,
        site_type=site_type,
        has_data_type="dv",
        output_data_type="dv",
    )


def fetch_field_catalog(
    state_cd: str | Sequence[str], *, site_type: str = "GW"
) -> pd.DataFrame:
    """Series catalog for discrete field measurements at groundwater sites."""
    return fetch_gw_sites(
        state_cd,
        site_type=site_type,
        has_data_type="gw",
        output_data_type="gw",
    )


# ---------------------------------------------------------------------------
# Modernized OGC API: observations
# ---------------------------------------------------------------------------


def _parse_datetime(values: pd.Series) -> pd.Series:
    """Parse NWIS timestamps while preserving local wall-clock time."""
    text = values.astype("string").str.replace(
        r"([+-]\d{2}:?\d{2}|Z)$", "", regex=True
    )
    return pd.to_datetime(text, errors="coerce", format="mixed")


def _site_id(site: str) -> str:
    site = str(site)
    return site if site.startswith("USGS-") else f"USGS-{site}"


def _chunks(items: Sequence[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _column(df: pd.DataFrame, name: str, default=pd.NA) -> pd.Series:
    """Return a column or a correctly-sized Series of ``default``."""
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _features_to_frame(payload: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in payload.get("features") or []:
        props = dict(feature.get("properties") or {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) >= 2:
            props.setdefault("longitude", coordinates[0])
            props.setdefault("latitude", coordinates[1])
        rows.append(props)
    return pd.DataFrame(rows)


def _fetch_ogc_pages(
    collection: str,
    params: dict[str, str],
    *,
    limit: int = 10000,
    max_pages: int = 50,
) -> pd.DataFrame:
    """Fetch all pages of an OGC Features collection."""
    url = f"{OGC_BASE}/collections/{collection}/items"
    query: dict[str, str] | None = {"f": "json", "limit": str(limit), **params}
    frames: list[pd.DataFrame] = []

    for _ in range(max_pages):
        response = _request(url, query)
        payload = response.json()
        frame = _features_to_frame(payload)
        if not frame.empty:
            frames.append(frame)

        next_url = next(
            (
                link.get("href")
                for link in payload.get("links") or []
                if link.get("rel") == "next"
            ),
            None,
        )
        if not next_url:
            break
        url = next_url
        query = None

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_field_measurements(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "site_no": _column(df, "monitoring_location_id", site_id)
            .astype(str)
            .str.replace("USGS-", "", regex=False),
            "site_name": pd.NA,
            "latitude": pd.to_numeric(_column(df, "latitude"), errors="coerce"),
            "longitude": pd.to_numeric(_column(df, "longitude"), errors="coerce"),
            "parameter_cd": _column(df, "parameter_code").astype("string").str.zfill(5),
            "variable_name": _column(df, "observing_procedure"),
            "unit": _column(df, "unit_of_measure"),
            "method_id": _column(df, "observing_procedure_code"),
            "method_desc": _column(df, "observing_procedure"),
            "datetime": _parse_datetime(_column(df, "time")),
            "value": pd.to_numeric(_column(df, "value"), errors="coerce"),
            "qualifiers": _column(df, "qualifier").astype(str),
            "vertical_datum": _column(df, "vertical_datum"),
            "approval_status": _column(df, "approval_status"),
        }
    )
    return out.dropna(subset=["site_no", "datetime", "value"]).reset_index(drop=True)


def fetch_field_measurements(
    sites: Sequence[str],
    start: str,
    end: str,
    *,
    parameter_cd: str | None = "72019",
    pause: float = 0.1,
) -> pd.DataFrame:
    """Fetch discrete field measurements from the OGC field-measurements API."""
    frames: list[pd.DataFrame] = []
    for site in sites:
        site_id = _site_id(site)
        params = {
            "monitoring_location_id": site_id,
            "datetime": f"{start}/{end}",
        }
        if parameter_cd:
            params["parameter_code"] = str(parameter_cd)

        df = _fetch_ogc_pages("field-measurements", params)
        if df.empty:
            logger.info("No field measurements for %s", site_id)
            continue
        frames.append(_normalize_field_measurements(df, site_id))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_daily_values(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "site_no": _column(df, "monitoring_location_id", site_id)
            .astype(str)
            .str.replace("USGS-", "", regex=False),
            "site_name": pd.NA,
            "latitude": pd.to_numeric(_column(df, "y"), errors="coerce"),
            "longitude": pd.to_numeric(_column(df, "x"), errors="coerce"),
            "parameter_cd": _column(df, "parameter_code").astype("string").str.zfill(5),
            "statistic_id": _column(df, "statistic_id").astype("string").str.zfill(5),
            "unit": _column(df, "unit_of_measure"),
            "datetime": pd.to_datetime(_column(df, "time"), errors="coerce"),
            "value": pd.to_numeric(_column(df, "value"), errors="coerce"),
            "qualifiers": _column(df, "qualifier"),
            "approval_status": _column(df, "approval_status"),
        }
    )
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.normalize()
    return out.dropna(subset=["site_no", "datetime", "value"]).reset_index(drop=True)


def fetch_daily_values(
    sites: Sequence[str],
    start: str,
    end: str,
    *,
    parameter_cd: str = "72019",
    statistic_id: str | None = "00001",
    pause: float = 0.1,
    page_limit: int = 10000,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch daily values from the OGC daily collection.

    The CSV representation is used because it is much smaller than GeoJSON for
    multi-decade daily series.  The caller should keep the requested period to
    at most about 27 years so that one 10,000-row page is sufficient.
    """
    frames: list[pd.DataFrame] = []
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    for index, site in enumerate(sites, start=1):
        site_id = _site_id(site)
        cache_file = (
            cache_path / f"{site_id.replace('USGS-', '')}.parquet"
            if cache_path is not None
            else None
        )
        if cache_file is not None and cache_file.exists():
            frames.append(pd.read_parquet(cache_file))
            continue

        params = {
            "f": "csv",
            "monitoring_location_id": site_id,
            "parameter_code": str(parameter_cd),
            "datetime": f"{start}/{end}",
            "limit": str(page_limit),
        }
        if statistic_id:
            params["statistic_id"] = str(statistic_id)

        try:
            response = _request(f"{OGC_BASE}/collections/daily/items", params)
            text = response.text.strip()
            if not text:
                logger.info("No daily values for %s", site_id)
                continue
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    dtype={"statistic_id": "string", "parameter_code": "string"},
                )
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue

            if len(df) >= page_limit:
                logger.warning(
                    "Daily response for %s hit the %s-row page limit; shorten "
                    "the date range or add pagination",
                    site_id,
                    page_limit,
                )
            normalized = _normalize_daily_values(df, site_id)
            if cache_file is not None:
                normalized.to_parquet(cache_file, index=False)
            frames.append(normalized)
        except Exception as exc:  # one bad site should not kill the batch
            logger.warning("Failed to fetch daily values for %s: %s", site_id, exc)

        if index % 25 == 0:
            logger.info("daily values: %s/%s sites", index, len(sites))
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _parse_legacy_dv(payload: dict) -> pd.DataFrame:
    """Parse the legacy NWIS daily-values JSON response."""
    rows: list[dict] = []
    for series in (payload.get("value") or {}).get("timeSeries") or []:
        source = series.get("sourceInfo") or {}
        site_codes = source.get("siteCode") or []
        site_no = site_codes[0].get("value") if site_codes else None
        geo = (source.get("geoLocation") or {}).get("geogLocation") or {}
        variable = series.get("variable") or {}
        variable_codes = variable.get("variableCode") or []
        parameter_cd = (
            variable_codes[0].get("value") if variable_codes else None
        )
        options = (variable.get("options") or {}).get("option") or []
        statistic_id = next(
            (
                option.get("optionCode")
                for option in options
                if option.get("name") == "Statistic"
            ),
            None,
        )
        unit = (variable.get("unit") or {}).get("unitCode")

        for method in series.get("values") or []:
            for observation in method.get("value") or []:
                qualifiers = observation.get("qualifiers") or []
                if "A" in qualifiers:
                    approval = "Approved"
                elif "P" in qualifiers:
                    approval = "Provisional"
                else:
                    approval = pd.NA
                rows.append(
                    {
                        "site_no": site_no,
                        "site_name": source.get("siteName"),
                        "latitude": geo.get("latitude"),
                        "longitude": geo.get("longitude"),
                        "parameter_cd": parameter_cd,
                        "statistic_id": statistic_id,
                        "unit": unit,
                        "datetime": observation.get("dateTime"),
                        "value": observation.get("value"),
                        "qualifiers": "|".join(qualifiers),
                        "approval_status": approval,
                    }
                )
    return pd.DataFrame(rows)


def _normalize_legacy_daily(df: pd.DataFrame, site_id: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "site_no": _column(df, "site_no", site_id).astype(str),
            "site_name": _column(df, "site_name"),
            "latitude": pd.to_numeric(_column(df, "latitude"), errors="coerce"),
            "longitude": pd.to_numeric(_column(df, "longitude"), errors="coerce"),
            "parameter_cd": _column(df, "parameter_cd").astype("string").str.zfill(5),
            "statistic_id": _column(df, "statistic_id").astype("string").str.zfill(5),
            "unit": _column(df, "unit"),
            "datetime": pd.to_datetime(_column(df, "datetime"), errors="coerce"),
            "value": pd.to_numeric(_column(df, "value"), errors="coerce"),
            "qualifiers": _column(df, "qualifiers").astype(str),
            "approval_status": _column(df, "approval_status"),
        }
    )
    out["datetime"] = out["datetime"].dt.normalize()
    return out.dropna(subset=["site_no", "datetime", "value"]).reset_index(
        drop=True
    )


def fetch_daily_values_legacy(
    sites: Sequence[str],
    start: str,
    end: str,
    *,
    parameter_cd: str = "72019",
    statistic_id: str | None = "00001",
    chunk_size: int = 20,
    pause: float = 0.5,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch daily values from the legacy NWIS dv endpoint in site batches.

    The legacy endpoint accepts many sites per request, which is much friendlier
    to the USGS rate limit than one request per site.  It is still available as
    of late 2026, but the modern OGC ``daily`` collection is the long-term
    replacement.
    """
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    site_list = [str(site) for site in sites]
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for site in site_list:
        cache_file = cache_path / f"{site}.parquet" if cache_path else None
        if cache_file is not None and cache_file.exists():
            frames.append(pd.read_parquet(cache_file))
        else:
            missing.append(site)

    for index, chunk in enumerate(_chunks(missing, chunk_size), start=1):
        params = {
            "format": "json",
            "sites": ",".join(chunk),
            "parameterCd": str(parameter_cd),
            "startDT": start,
            "endDT": end,
        }
        if statistic_id:
            params["statCd"] = str(statistic_id)

        try:
            response = _request(f"{NWIS_BASE}/dv/", params)
            parsed = _parse_legacy_dv(response.json())
            if parsed.empty:
                logger.info("No daily values in chunk %s", index)
                continue
            normalized = _normalize_legacy_daily(parsed, chunk[0])
            if cache_path is not None:
                for site, group in normalized.groupby("site_no"):
                    group.to_parquet(cache_path / f"{site}.parquet", index=False)
            frames.append(normalized)
            logger.info(
                "legacy dv chunk %s: %s sites, %s rows",
                index,
                normalized["site_no"].nunique(),
                len(normalized),
            )
        except Exception as exc:  # keep going; cached sites are preserved
            logger.warning("Failed legacy dv chunk %s: %s", index, exc)
        time.sleep(pause)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_gwlevels(
    sites: Sequence[str],
    start: str,
    end: str,
    *,
    parameter_cd: str | None = None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`fetch_field_measurements`."""
    logger.warning("fetch_gwlevels is deprecated; use fetch_field_measurements")
    return fetch_field_measurements(
        sites, start, end, parameter_cd=parameter_cd
    )
