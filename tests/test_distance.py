import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from gpx_importer.importer import Parser as GpxParser
from src.jobs.import_data import ImportConfiguration, SERIES_INDEX_ATTR, write_hdf5_cache


PLUGIN_DIR = Path(__file__).resolve().parents[1]
# At the equator, this longitude increment is a 3 m great-circle step.
THREE_METRES_DEG = math.degrees(3 / 6_371_008.8)


def nmea(body):
    checksum = 0
    for byte in body.encode("ascii"):
        checksum ^= byte
    return f"${body}*{checksum:02X}\r\n".encode("ascii")


class GnssDistanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
    def parse_gpx(self, segments):
        body = []
        for points in segments:
            body.append("<trkseg>")
            for second, lat, lon, alt in points:
                elevation = "" if alt is None else f"<ele>{alt}</ele>"
                body.append(
                    f'<trkpt lat="{lat}" lon="{lon}">{elevation}'
                    f'<time>2026-09-23T12:00:{second:02d}Z</time></trkpt>'
                )
            body.append("</trkseg>")
        path = self.folder / "track.gpx"
        path.write_text('<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk>' + "".join(body) + "</trk></gpx>")
        parser = GpxParser(PLUGIN_DIR / "importer.json")
        return parser, path, parser.parse(path)
    def assert_distances(self, result, horizontal, spatial):
        for name, expected in (("distance_2d_m", horizontal), ("distance_3d_m", spatial)):
            series = result["series"][f"GNSS/{name}"]
            np.testing.assert_allclose(series["values"], expected, atol=1e-6)
            self.assertEqual(series["unit"], "m")
            self.assertTrue(series["visible"])
            self.assertTrue(np.all(np.diff(series["time"]) > 0))
    def test_gpx_slope_and_return_accumulate_travelled_distance(self):
        _, _, result = self.parse_gpx([[
            (0, 0, 0, 10),
            (1, 0, THREE_METRES_DEG, 14),
            (2, 0, 0, 10),
        ]])
        self.assert_distances(result, [0, 3, 6], [0, 5, 10])
    def test_altitude_only_motion_does_not_change_2d(self):
        _, _, result = self.parse_gpx([[
            (0, 60, 24, 10), (1, 60, 24, 14), (2, 60, 24, 11),
        ]])
        self.assert_distances(result, [0, 0, 0], [0, 4, 7])
    def test_missing_altitude_does_not_create_a_drop_to_zero(self):
        for elevations in ((1000, None, 1004), (None, None, None)):
            with self.subTest(elevations=elevations):
                _, _, result = self.parse_gpx([[
                    (i, 0, i * THREE_METRES_DEG, alt)
                    for i, alt in enumerate(elevations)
                ]])
                self.assert_distances(result, [0, 3, 6], [0, 3, 6])
    def test_gpx_segments_do_not_add_a_jump(self):
        _, _, result = self.parse_gpx([
            [(0, 0, 0, 0), (1, 0, THREE_METRES_DEG, 4)],
            [(2, 0, 90, 1000), (3, 0, 90 + THREE_METRES_DEG, 1004)],
        ])
        self.assert_distances(result, [0, 3, 3, 6], [0, 5, 5, 10])
    def test_single_point_starts_at_zero(self):
        _, _, result = self.parse_gpx([[(0, 60, 24, 100)]])
        self.assert_distances(result, [0], [0])
    def test_dateline_crossing_uses_short_distance(self):
        _, _, result = self.parse_gpx([[
            (0, 0, 180 - THREE_METRES_DEG / 2, 10),
            (1, 0, -180 + THREE_METRES_DEG / 2, 14),
        ]])
        self.assert_distances(result, [0, 3], [0, 5])
    def test_unsorted_duplicate_epochs_use_last_position(self):
        _, _, result = self.parse_gpx([[
            (2, 0, 2 * THREE_METRES_DEG, 18),
            (0, 0, 0, 10),
            (1, 50, 50, 1000),
            (1, 0, THREE_METRES_DEG, 14),
        ]])
        self.assert_distances(result, [0, 3, 6], [0, 5, 10])
    def test_gpx_distances_are_saved_in_h5_and_navigator_index(self):
        parser, path, result = self.parse_gpx([[
            (0, 0, 0, 10), (1, 0, THREE_METRES_DEG, 14),
        ]])
        self.assert_cache_distances(parser, path, result)
    def assert_cache_distances(self, parser, path, result):
        cache = path.with_suffix(".h5")
        write_hdf5_cache(
            output_path=cache, input_path=path,
            selected_config=ImportConfiguration(parser.config_path, parser.config, parser),
            import_result=result,
        )
        with h5py.File(cache, "r") as handle:
            index = json.loads(handle.attrs[SERIES_INDEX_ATTR])
            for name, expected in (("distance_2d_m", [0, 3]), ("distance_3d_m", [0, 5])):
                group = handle[f"series/GNSS/{name}"]
                np.testing.assert_allclose(group["value"][:], expected, atol=1e-6)
                np.testing.assert_array_equal(
                    group["time_unix"][:], result["series"][f"GNSS/{name}"]["time"],
                )
                self.assertEqual(group.attrs["unit"], "m")
                self.assertTrue(any(entry["navigator_name"] == name for entry in index))
