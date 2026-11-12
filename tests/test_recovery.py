import tempfile
import unittest
from pathlib import Path

import numpy as np

from gpx_importer.importer import Parser as GpxParser


PLUGIN_DIR = Path(__file__).resolve().parents[1]


class GnssRecoveryTests(unittest.TestCase):
    def test_gpx_outlier_is_removed_and_counted(self):
        points = [
            ("2026-09-23T05:27:55Z", 10),
            ("2000-01-01T05:30:30Z", 99),
            ("2026-09-23T05:27:56Z", 11),
        ]
        body = "".join(
            f'<trkpt lat="60" lon="24"><ele>{alt}</ele><time>{time}</time></trkpt>'
            for time, alt in points
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.gpx"
            path.write_text(f'<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>{body}</trkseg></trk></gpx>')
            result = GpxParser(PLUGIN_DIR / "importer.json").parse(path)
        self.assertEqual(result["metadata"]["track_point_count"], 3)
        self.assertEqual(result["metadata"]["skipped_track_point_count"], 1)
        self.assertEqual(result["metadata"]["timestamp_outlier_track_point_count"], 1)
        np.testing.assert_array_equal(result["series"]["GNSS/altitude_m"]["values"], [10, 11])
