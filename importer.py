"""GPX 1.0/1.1 track parser producing standard GNSS series."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.contracts.importer import DataParser, ProgressCallback, ProgressReporter, add_gnss_distance_series, filter_gnss_time_outliers


class Parser(DataParser):
    """Read timestamped GPX track points without an external GPX dependency."""

    name = "GPX GNSS track"
    priority = 90
    extensions = frozenset({".gpx"})
    format_name = "gnss.gpx"

    def can_parse(self, input_path: str | Path) -> bool:
        path = Path(input_path)
        try:
            with path.open("rb") as file:
                sample = file.read(8192).lstrip().lower()
        except OSError:
            return False
        return b"<gpx" in sample

    def parse(
        self,
        input_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        path = Path(input_path)
        file_size = max(1, path.stat().st_size)
        latitudes: list[tuple[float, float]] = []
        longitudes: list[tuple[float, float]] = []
        altitudes: list[tuple[float, float]] = []
        segment_ids: dict[float, int] = {}
        segment_id = 0
        track_points = 0
        skipped_points = 0
        progress_reporter = ProgressReporter(progress_callback)

        # iterparse keeps even very long recordings bounded in memory.  Clear
        # each completed point after its attributes and child elements are read.
        with path.open("rb") as file:
            for _event, element in ElementTree.iterparse(file, events=("end",)):
                if _local_name(element.tag) == "trkseg":
                    segment_id += 1
                    element.clear()
                    continue
                if _local_name(element.tag) != "trkpt":
                    continue
                track_points += 1
                progress_reporter.report(min(0.99, file.tell() / file_size))
                timestamp = _point_timestamp(element)
                latitude = _finite_float(element.attrib.get("lat"))
                longitude = _finite_float(element.attrib.get("lon"))
                if timestamp is None or latitude is None or longitude is None:
                    skipped_points += 1
                    element.clear()
                    continue
                if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
                    skipped_points += 1
                    element.clear()
                    continue

                latitudes.append((timestamp, latitude))
                longitudes.append((timestamp, longitude))
                segment_ids[timestamp] = segment_id
                altitude = _child_float(element, "ele")
                if altitude is not None:
                    altitudes.append((timestamp, altitude))
                element.clear()

        if not latitudes:
            raise ValueError(f"GPX file contains no timestamped track points: {path.name}")

        output: dict[str, dict[str, Any]] = {}
        self.add_series(output, "latitude_deg", "GNSS", *zip(*latitudes), source="GPX track point")
        self.add_series(output, "longitude_deg", "GNSS", *zip(*longitudes), source="GPX track point")
        if altitudes:
            self.add_series(output, "altitude_m", "GNSS", *zip(*altitudes), source="GPX track point")
        skipped_timestamps = filter_gnss_time_outliers(output)
        add_gnss_distance_series(self, output, segment_ids=segment_ids)
        skipped_time_points = skipped_timestamps.get("GNSS/latitude_deg", 0)
        progress_reporter.finish()
        return {
            "format": self.format_name,
            "series": output,
            "metadata": {
                "track_point_count": track_points,
                "skipped_track_point_count": skipped_points + skipped_time_points,
                "timestamp_outlier_track_point_count": skipped_time_points,
            },
        }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _point_timestamp(element: ElementTree.Element) -> float | None:
    for child in element:
        if _local_name(child.tag) != "time" or not child.text:
            continue
        try:
            parsed = datetime.fromisoformat(child.text.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _child_float(element: ElementTree.Element, name: str) -> float | None:
    for child in element:
        if _local_name(child.tag) == name:
            return _finite_float(child.text)
    return None


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
