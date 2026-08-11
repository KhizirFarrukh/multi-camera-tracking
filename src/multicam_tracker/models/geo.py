"""Geographic primitives."""

from __future__ import annotations

import math

from pydantic import Field

from multicam_tracker.models.base import MCTBaseModel

__all__ = ["EARTH_RADIUS_METERS", "GeoPoint"]

EARTH_RADIUS_METERS = 6_371_008.8
"""IUGG mean Earth radius. Chosen over the equatorial radius because camera
topologies span a few kilometres in arbitrary directions, where the mean radius
minimises worst-case error."""


class GeoPoint(MCTBaseModel):
    """A WGS84 coordinate."""

    lat: float = Field(ge=-90.0, le=90.0, description="WGS84 latitude in degrees")
    lon: float = Field(ge=-180.0, le=180.0, description="WGS84 longitude in degrees")

    def haversine_distance_to(self, other: GeoPoint) -> float:
        """Return the great-circle distance to ``other`` in metres.

        Uses the haversine formula, which is numerically well behaved for the
        short distances between neighbouring cameras -- unlike the spherical law
        of cosines, which loses precision as the separation approaches zero.
        Antimeridian crossings need no special handling: the formula depends on
        the cosine of the longitude difference, which is periodic.

        The Earth is modelled as a sphere, so this runs roughly 0.3% off a
        geodesic calculation at worst. That is far inside the uncertainty of any
        travel-time estimate built on top of it.

        Args:
            other: The destination point.

        Returns:
            Distance in metres. Zero for identical coordinates.
        """
        lat1, lon1 = math.radians(self.lat), math.radians(self.lon)
        lat2, lon2 = math.radians(other.lat), math.radians(other.lon)

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        chord = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        # min(1.0, ...) guards against a chord marginally above 1 from rounding,
        # which would make asin() raise for two effectively identical points.
        return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(chord)))
