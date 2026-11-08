# MS-GNNs: AIS Preprocessing and Graph Construction

This repository provides the **data preprocessing** and **multi-scale graph construction** code used in research on vessel estimated time of arrival (ETA) prediction with MS-GNNs. **It does not include the complete MS-GNNs prediction model or its training and evaluation code.**

## Contents

- `src/ais_preprocessing.py` — AIS cleaning, voyage segmentation, Singapore-bound voyage extraction, and voyage statistics.
- `src/graph_construction.py` — Raster-based A* trajectory completion and multi-scale self-organizing map (SOM) construction.
- `requirements.txt` — Python dependencies.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**1. Preprocess AIS data** (Japan or Australia to Singapore):

```bash
python src/ais_preprocessing.py \
  --input data/ais.parquet \
  --region JP \
  --output-dir outputs/JP
```

Set `--region AU` for Australia. The input Parquet file requires the columns `mmsi`, `last_known_timestamp`, `lat`, `lng`, and `last_known_speed`.

**2. Construct multi-scale graphs:**

```bash
python src/graph_construction.py \
  --voyages outputs/JP/JP_2SG.parquet \
  --raster data/map_data.npy \
  --bounds 120 -5 155 46 \
  --som-sizes 4 6 9 12 \
  --output-dir outputs/JP_graph
```

The `--bounds` values above are illustrative; replace them with the geographic extent appropriate for your study. The raster must be a georeferenced-by-configuration 2D NumPy array with `0` representing navigable water. Check its geographic extent and, if necessary, set `--global-bounds`.

## Data availability

AIS observations and the land/water raster are **not included** in this repository. Users must provide data they are authorized to use. The code supports preprocessing and graph construction only; it cannot independently reproduce the paper's ETA prediction results.
