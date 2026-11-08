"""AIS voyage preprocessing for the MS-GNNs experimental pipeline.

Adapted from AIS_preprocess.ipynb. No commercial AIS observations are bundled.
Port rectangles and segmentation rules are illustrative of the supplied notebook;
review them against your licensed data and research protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

TIME = 'last_known_timestamp'
SHIP = 'mmsi'
SPEED = 'last_known_speed'
REQUIRED = {SHIP, TIME, 'lat', 'lng', SPEED}

# Axis-aligned rectangles are (lat_min, lat_max, lng_min, lng_max).
SINGAPORE_TERMINALS = {
    'PSA_Pasir_Panjang': [(1.2533, 1.2943, 103.7486, 103.7966)],
    'Tanjong_Pagar_Keppel': [(1.2600, 1.2728, 103.8392, 103.8519)],
}
ORIGIN_PORTS = {
    'JP': {
        'Tokyo': [(35.5859, 35.6133, 139.7567, 139.8039)],
        'Yokohama': [(35.4248, 35.4470, 139.6686, 139.6912)],
        'Nagoya': [(35.0142, 35.0509, 136.7753, 136.8440)],
        'Kobe': [(34.6472, 34.6982, 135.2043, 135.2982)],
        'Osaka': [(34.6281, 34.6463, 135.4230, 135.4443)],
    },
    'AU': {
        'Sydney Botany': [(-33.9864, -33.9573, 151.1937, 151.2292)],
        'Melbourne': [(-37.8214, -37.8056, 144.9057, 144.9348),
                      (-37.8530, -37.8353, 144.8963, 144.9183)],
        'Adelaide': [(-34.7729, -34.7654, 138.4876, 138.4964)],
        'Brisbane': [(-27.3818, -27.3508, 153.1597, 153.1906)],
        'Perth Fremantle': [(-32.0530, -32.0358, 115.7293, 115.7535)],
    },
}


def inside_rectangles(df: pd.DataFrame, rectangles: list[tuple]) -> pd.Series:
    """Vectorized test against the union of axis-aligned geographic rectangles."""
    result = pd.Series(False, index=df.index)
    for lat_lo, lat_hi, lon_lo, lon_hi in rectangles:
        result |= (df['lat'].between(lat_lo, lat_hi) & df['lng'].between(lon_lo, lon_hi))
    return result


def prepare_ais(df: pd.DataFrame, gap_hours: float = 15.0) -> pd.DataFrame:
    """Sort AIS records and assign a new trip after a >gap_hours observation gap.

    This is a gap-based trip *segment*, not necessarily a completed ocean voyage.
    """
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required AIS columns: {sorted(missing)}')
    result = df.copy().dropna(subset=[SHIP, TIME, 'lat', 'lng'])
    result[TIME] = pd.to_datetime(result[TIME], errors='coerce', utc=True)
    result = result.dropna(subset=[TIME]).sort_values([SHIP, TIME]).reset_index(drop=True)
    gaps = result.groupby(SHIP, sort=False)[TIME].diff()
    result['trip'] = gaps.gt(pd.Timedelta(hours=gap_hours)).groupby(result[SHIP]).cumsum().add(1).astype('int64')
    return result


def select_singapore_arrivals(df: pd.DataFrame, speed_threshold: float = 0.1) -> pd.DataFrame:
    """Keep segments reaching Singapore terminal rectangles, ending at last low speed.

    Mirrors the notebook's terminal-region and low-speed filtering intention.
    Each segment is handled independently to avoid cross-vessel shift artifacts.
    """
    df = df.copy()
    terminal_rects = [rect for areas in SINGAPORE_TERMINALS.values() for rect in areas]
    df['In_Singapore_Ports'] = inside_rectangles(df, terminal_rects)
    segments = []
    for _, group in df.groupby([SHIP, 'trip'], sort=False):
        group = group.sort_values(TIME).reset_index(drop=True)
        inside = group['In_Singapore_Ports']
        enters = inside & ~inside.shift(fill_value=False)
        if not enters.any():
            continue
        arrival_pos = int(enters.to_numpy().nonzero()[0][0])
        # Only consider first detected terminal visit (exclude subsequent departures).
        remaining = group.iloc[arrival_pos:]
        leaves = (~remaining['In_Singapore_Ports']).to_numpy().nonzero()[0]
        endpoint = arrival_pos + int(leaves[0]) if len(leaves) else len(group)
        voyage = group.iloc[:endpoint].copy()
        in_terminal = voyage['In_Singapore_Ports']
        speeds = pd.to_numeric(voyage[SPEED], errors='coerce')
        # Notebook rejects groups without a recorded exact-zero and <threshold port speed.
        if not (speeds[in_terminal].eq(0).any() and speeds[in_terminal].lt(speed_threshold).any()):
            continue
        low_speed_indices = (speeds.lt(speed_threshold) | speeds.eq(0)).to_numpy().nonzero()[0]
        if len(low_speed_indices):
            voyage = voyage.iloc[:int(low_speed_indices[-1]) + 1].copy()
        if not voyage.empty:
            segments.append(voyage)
    return pd.concat(segments, ignore_index=True) if segments else df.iloc[0:0].copy()


def extract_origin_voyages(df: pd.DataFrame, region: str, speed_threshold: float = 0.1) -> pd.DataFrame:
    """Extract trips from each specified origin rectangle to Singapore.

    A trip may match multiple origin ports, as in the original per-port extraction.
    For each port retain records starting with the first in-port observation.
    """
    if region not in ORIGIN_PORTS:
        raise ValueError(f'Unknown origin region {region!r}; choose JP or AU')
    output = []
    grouped = list(df.groupby([SHIP, 'trip'], sort=False))
    for port, bounds in ORIGIN_PORTS[region].items():
        for _, group in grouped:
            group = group.sort_values(TIME).reset_index(drop=True)
            in_port = inside_rectangles(group, bounds)
            if not in_port.any():
                continue
            start = int(in_port.to_numpy().nonzero()[0][0])
            segment = group.iloc[start:].copy().reset_index(drop=True)
            segment['Departure Port'] = port
            # Align to the first low-speed point only when it is inside this port.
            slow = pd.to_numeric(segment[SPEED], errors='coerce').lt(speed_threshold)
            slow_positions = slow.to_numpy().nonzero()[0]
            if len(slow_positions) and bool(inside_rectangles(segment, bounds).iloc[int(slow_positions[0])]):
                segment = segment.iloc[int(slow_positions[0]):].copy()
            output.append(segment)
    return pd.concat(output, ignore_index=True) if output else df.iloc[0:0].copy()


def voyage_statistics(df: pd.DataFrame, speed_threshold: float = 0.1) -> pd.DataFrame:
    """Estimate departure/arrival timestamps and sailing duration from speed records.

    This reproduces the notebook's speed-based rule; it is not berth-verified ATA.
    """
    results = []
    for (port, mmsi, trip), group in df.groupby(['Departure Port', SHIP, 'trip'], sort=False):
        group = group.sort_values(TIME)
        speeds = pd.to_numeric(group[SPEED], errors='coerce')
        departures = group.loc[speeds.gt(speed_threshold), TIME]
        arrivals = group.loc[speeds.le(speed_threshold), TIME]
        departure = departures.iloc[0] if len(departures) else pd.NaT
        arrival = arrivals.iloc[-1] if len(arrivals) else pd.NaT
        duration = (arrival - departure).total_seconds() / 3600 if pd.notna(departure) and pd.notna(arrival) else float('nan')
        results.append({'Departure Port': port, SHIP: mmsi, 'trip': trip,
                        'departure_time': departure, 'arrival_time': arrival,
                        'voyage_duration_hours': duration,
                        'voyage_duration_days': duration / 24})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description='Preprocess licensed AIS data to Singapore-bound voyages.')
    parser.add_argument('--input', required=True, type=Path, help='Input AIS parquet file')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--region', choices=['JP', 'AU'], required=True)
    parser.add_argument('--gap-hours', type=float, default=15.0)
    parser.add_argument('--speed-threshold', type=float, default=0.1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ais = prepare_ais(pd.read_parquet(args.input), args.gap_hours)
    ais.to_parquet(args.output_dir / 'ais_with_trip.parquet', index=False)
    arrivals = select_singapore_arrivals(ais, args.speed_threshold)
    arrivals.to_parquet(args.output_dir / 'ais_to_singapore.parquet', index=False)
    voyages = extract_origin_voyages(arrivals, args.region, args.speed_threshold)
    voyages.to_parquet(args.output_dir / f'{args.region}_2SG.parquet', index=False)
    voyage_statistics(voyages, args.speed_threshold).to_parquet(
        args.output_dir / f'{args.region}_2SG_stats.parquet', index=False)
    print(f'Prepared {len(voyages)} AIS observations in {args.output_dir}')


if __name__ == '__main__':
    main()
