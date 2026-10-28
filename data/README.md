# Data provenance and license

## Raw data

All raw groundwater observations are from the U.S. Geological Survey National
Water Information System (NWIS):

- Service: USGS Water Data / NWIS
- Parameter code: 72019 (depth to water level, feet below land surface)
- Statistic code: 00001 (daily maximum)
- Period: 2000-2024
- States: California, Nebraska, Texas
- Retrieval date: 2026-10-27

USGS data are in the public domain. Users should cite the USGS NWIS service
and the retrieval date.

## Processed panels

The `data/processed/` directory contains derived panels:

- `panel_full.parquet` - 279 wells, complete daily grid
- `panel_core.parquet` - 89 wells, observed coverage >= 70%
- `panel_core_clean.parquet` - 74 wells, core minus wells with >5 large jumps

These derived panels are released under CC BY 4.0 as part of the gwbench
release. They are provided for reproducibility; the authoritative source is
USGS NWIS.
