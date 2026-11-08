"""Run with: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from ais_preprocessing import prepare_ais, inside_rectangles, extract_origin_voyages
from graph_construction import (build_water_graph, complete_trajectory,
                                fit_som, remove_consecutive_duplicates)


class PipelineTests(unittest.TestCase):
    def test_trip_gap(self):
        data = pd.DataFrame({'mmsi': [1, 1, 1, 2],
                             'last_known_timestamp': ['2021-01-01 00:00', '2021-01-01 02:00',
                                                      '2021-01-02 02:00', '2021-01-01 01:00'],
                             'lat': [1, 1, 1, 2], 'lng': [103, 103, 103, 104],
                             'last_known_speed': [0, 2, 3, 0]})
        trips = prepare_ais(data)
        self.assertEqual(trips.loc[trips.mmsi == 1, 'trip'].tolist(), [1, 1, 2])

    def test_port_rectangles(self):
        df = pd.DataFrame({'lat': [35.60, 0], 'lng': [139.78, 0]})
        result = inside_rectangles(df, [(35.5859, 35.6133, 139.7567, 139.8039)])
        self.assertEqual(result.tolist(), [True, False])

    def test_astar_connectivity(self):
        water = np.zeros((4, 4), dtype=np.uint8)
        water[1, 1] = 1
        g = build_water_graph(water)
        path = complete_trajectory([(0, 0), (3, 3)], g)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 3))
        self.assertTrue(all(a in g and b in g and g.has_edge(a, b)
                            for a, b in zip(path[:-1], path[1:])))
        self.assertEqual(remove_consecutive_duplicates([(0, 0), (0, 0)]), [(0, 0)])

    @unittest.skipUnless(__import__("importlib").util.find_spec("minisom"), "minisom not installed")
    def test_som(self):
        pts = pd.DataFrame({'row': [0, 0, 2, 2], 'col': [0, 2, 0, 2]})
        clustered, som = fit_som(pts, size=2, iterations=20, seed=1)
        self.assertEqual(len(clustered), 4)
        self.assertEqual(som.get_weights().shape, (2, 2, 2))


if __name__ == '__main__':
    unittest.main()
