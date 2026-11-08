"""Raster-based A* completion and multi-scale SOM construction.

Adapted from vessel_traffic_network_JP.ipynb. Expects a separately supplied
water/land raster (.npy), where 0 is traversable water. Does not include the
original external map_data.npy asset. Coordinates are always (row, col).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rasterio.transform import from_bounds, rowcol


def crop_global_raster(raster: np.ndarray, bounds: tuple[float, float, float, float],
                       global_bounds: tuple[float, float, float, float] =
                       (-180.0, -90.0, 180.0, 83.6651126)):
    """Crop global raster to (lon_min, lat_min, lon_max, lat_max).

    Global extent follows the original notebook. Verify your actual raster's
    georeferencing before use; a bare .npy file carries no spatial metadata.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    transform = from_bounds(*global_bounds, raster.shape[1], raster.shape[0])
    top, left = rowcol(transform, lon_min, lat_max)
    bottom, right = rowcol(transform, lon_max, lat_min)
    r0, r1 = max(0, min(top, bottom)), min(raster.shape[0], max(top, bottom) + 1)
    c0, c1 = max(0, min(left, right)), min(raster.shape[1], max(left, right) + 1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError('Requested bounds do not intersect supplied raster')
    return raster[r0:r1, c0:c1], transform, (r0, c0)


def observations_to_grid(df: pd.DataFrame, transform, origin: tuple[int, int]) -> pd.DataFrame:
    """Map lon/lat to raster-local (row,col), preserving original record order."""
    rows, cols = rowcol(transform, df['lng'].to_numpy(), df['lat'].to_numpy())
    result = df.copy()
    result['row'] = np.asarray(rows, dtype=int) - origin[0]
    result['col'] = np.asarray(cols, dtype=int) - origin[1]
    return result


def keep_water_observations(df: pd.DataFrame, water_grid: np.ndarray) -> pd.DataFrame:
    r, c = df['row'].to_numpy(), df['col'].to_numpy()
    valid = ((r >= 0) & (r < water_grid.shape[0]) &
             (c >= 0) & (c < water_grid.shape[1]))
    mask = np.zeros(len(df), dtype=bool)
    mask[valid] = water_grid[r[valid], c[valid]] == 0
    return df.loc[mask].copy()


def build_water_graph(grid: np.ndarray, connectivity: int = 8) -> nx.Graph:
    """Build an unweighted water-cell graph with 4- or 8-neighbor connectivity.

    The notebook used all eight neighbors and weight=1, including diagonals.
    """
    if connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    deltas = [(0, 1), (1, 0)]
    if connectivity == 8:
        deltas += [(1, 1), (1, -1)]
    graph = nx.Graph()
    nrow, ncol = grid.shape
    for row, col in zip(*np.where(grid == 0)):
        row, col = int(row), int(col)
        graph.add_node((row, col))
        for dr, dc in deltas:
            rr, cc = row + dr, col + dc
            if 0 <= rr < nrow and 0 <= cc < ncol and grid[rr, cc] == 0:
                graph.add_edge((row, col), (rr, cc), weight=1)
    return graph


def remove_consecutive_duplicates(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    clean = []
    for point in points:
        if not clean or point != clean[-1]:
            clean.append(point)
    return clean


def complete_trajectory(points: list[tuple[int, int]], graph: nx.Graph) -> list[tuple[int, int]]:
    """Complete large grid gaps with NetworkX A* (unit-cost path).

    If a path is unavailable, the original consecutive valid points are kept,
    matching the notebook's fallback. Uses Chebyshev distance as an admissible
    heuristic for unit-cost 8-connected movement (unlike Manhattan distance).
    """
    points = remove_consecutive_duplicates([p for p in points if p in graph])
    if len(points) < 2:
        return points
    result = [points[0]]
    def chebyshev(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    for end in points[1:]:
        start = result[-1]
        if chebyshev(start, end) > 1:
            try:
                path = nx.astar_path(graph, start, end, heuristic=chebyshev, weight='weight')
                result.extend(path[1:])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                result.append(end)
        else:
            result.append(end)
    return result


def complete_all_voyages(df: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    """Output ordered (row,col) points with original voyage identifiers."""
    df = df.sort_values(['mmsi', 'trip', 'last_known_timestamp'])
    records = []
    for (mmsi, trip), g in df.groupby(['mmsi', 'trip'], sort=False):
        points = list(zip(g['row'].astype(int), g['col'].astype(int)))
        for order, (r, c) in enumerate(complete_trajectory(points, graph)):
            records.append({'mmsi': str(mmsi), 'trip': int(trip),
                            'point_order': order, 'row': r, 'col': c})
    return pd.DataFrame(records, columns=['mmsi', 'trip', 'point_order', 'row', 'col'])


def fit_som(points: pd.DataFrame, size: int, iterations: int = 3000,
            seed: int = 42) -> tuple[pd.DataFrame, object]:
    """Learn a size×size self-organizing map from unique completed grid points.

    Uses per-coordinate min/max normalization, sigma=size/3, learning_rate=.5,
    and MiniSom random initialization/training as in the source notebook.
    """
    if points.empty:
        raise ValueError('Cannot train SOM without water-grid trajectory points')
    unique = points[['row', 'col']].drop_duplicates().reset_index(drop=True).copy()
    data = unique[['row', 'col']].to_numpy(dtype=float)
    lo, hi = data.min(axis=0), data.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scaled = (data - lo) / span
    from minisom import MiniSom
    som = MiniSom(x=size, y=size, input_len=2, sigma=size / 3,
                  learning_rate=0.5, random_seed=seed)
    som.random_weights_init(scaled)
    som.train_random(scaled, num_iteration=iterations)
    winners = [som.winner(point) for point in scaled]
    unique['som_cluster'] = [f'{r}-{c}' for r, c in winners]
    return unique, som


def main() -> None:
    p = argparse.ArgumentParser(description='A* raster path completion followed by multi-scale SOM.')
    p.add_argument('--voyages', type=Path, required=True, help='Voyage AIS parquet')
    p.add_argument('--raster', type=Path, required=True, help='Land/water raster .npy, 0=water')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--bounds', nargs=4, type=float, required=True,
                   metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'))
    p.add_argument('--global-bounds', nargs=4, type=float,
                   default=(-180, -90, 180, 83.6651126))
    p.add_argument('--som-sizes', nargs='+', type=int, default=[4, 6, 9, 12])
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raster = np.load(args.raster, allow_pickle=False)
    if raster.ndim != 2:
        raise ValueError('Expected a 2-dimensional land/water raster')
    grid, transform, origin = crop_global_raster(raster, tuple(args.bounds), tuple(args.global_bounds))
    ais = pd.read_parquet(args.voyages)
    for col in ('mmsi', 'trip', 'last_known_timestamp', 'lat', 'lng'):
        if col not in ais:
            raise ValueError(f'Missing required column: {col}')
    ais = keep_water_observations(observations_to_grid(ais, transform, origin), grid)
    graph = build_water_graph(grid)
    completed = complete_all_voyages(ais, graph)
    completed.to_parquet(args.output_dir / 'completed_trajectories.parquet', index=False)
    unique_points = completed[['row', 'col']].drop_duplicates()
    for size in args.som_sizes:
        labels, som = fit_som(unique_points, size=size, iterations=args.iterations, seed=args.seed)
        labels.to_parquet(args.output_dir / f'som_{size}x{size}_labels.parquet', index=False)
        np.save(args.output_dir / f'som_{size}x{size}_weights.npy', som.get_weights())
    metadata = {'bounds': args.bounds, 'global_bounds': args.global_bounds,
                'som_sizes': args.som_sizes, 'iterations': args.iterations,
                'seed': args.seed, 'raster_shape': list(raster.shape),
                'cropped_raster_shape': list(grid.shape),
                'graph_nodes': graph.number_of_nodes(), 'graph_edges': graph.number_of_edges(),
                'trajectory_rows': len(completed)}
    (args.output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
