"""Unit tests for :mod:`multicam_tracker.models.geo`.

Reference distances come from published great-circle values rather than from
this module's own output, so the tests measure correctness rather than
self-consistency.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multicam_tracker.models import GeoPoint

pytestmark = pytest.mark.unit

TOLERANCE = 0.005
"""0.5%, per the stage spec. Absorbs the spherical-Earth approximation."""

LONDON = GeoPoint(lat=51.5074, lon=-0.1278)
PARIS = GeoPoint(lat=48.8566, lon=2.3522)
LONDON_PARIS_METERS = 343_500.0
"""Published great-circle distance, ~343.5 km."""

EQUATOR_DEGREE_METERS = 111_195.0
"""One degree of longitude at the equator on a sphere of mean Earth radius."""


def test_geo_point__valid_coordinates__construct() -> None:
    """The happy path."""
    point = GeoPoint(lat=40.7128, lon=-74.0060)

    assert point.lat == pytest.approx(40.7128)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(90.0, 0.0), (-90.0, 0.0), (0.0, 180.0), (0.0, -180.0)],
    ids=["north-pole", "south-pole", "antimeridian+", "antimeridian-"],
)
def test_geo_point__coordinates_at_the_bounds__are_accepted(lat: float, lon: float) -> None:
    """Boundary: the poles and both sides of the antimeridian are real places."""
    assert GeoPoint(lat=lat, lon=lon).lat == pytest.approx(lat)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(90.1, 0.0), (-90.1, 0.0), (0.0, 180.1), (0.0, -180.1)],
    ids=["lat-above", "lat-below", "lon-above", "lon-below"],
)
def test_geo_point__coordinates_out_of_range__are_rejected(lat: float, lon: float) -> None:
    """Out-of-range values are almost always transposed lat/lon."""
    with pytest.raises(ValidationError):
        GeoPoint(lat=lat, lon=lon)


def test_haversine__identical_points__is_zero() -> None:
    """The formula must not return a small positive artefact for a zero distance."""
    assert LONDON.haversine_distance_to(LONDON) == 0.0


def test_haversine__known_reference_pair__matches_the_published_distance() -> None:
    """London to Paris, within the 0.5% tolerance the spec allows."""
    measured = LONDON.haversine_distance_to(PARIS)

    assert measured == pytest.approx(LONDON_PARIS_METERS, rel=TOLERANCE)


def test_haversine__one_degree_at_the_equator__matches_the_expected_arc() -> None:
    """A second independent reference, chosen for its closed-form expected value."""
    measured = GeoPoint(lat=0.0, lon=0.0).haversine_distance_to(GeoPoint(lat=0.0, lon=1.0))

    assert measured == pytest.approx(EQUATOR_DEGREE_METERS, rel=TOLERANCE)


def test_haversine__antimeridian_crossing__takes_the_short_way() -> None:
    """179.5E to 179.5W is one degree apart, not 359 -- the formula is periodic."""
    east = GeoPoint(lat=0.0, lon=179.5)
    west = GeoPoint(lat=0.0, lon=-179.5)

    assert east.haversine_distance_to(west) == pytest.approx(EQUATOR_DEGREE_METERS, rel=TOLERANCE)


def test_haversine__antipodal_points__is_half_the_circumference() -> None:
    """Boundary: the maximum possible distance on the sphere."""
    measured = GeoPoint(lat=0.0, lon=0.0).haversine_distance_to(GeoPoint(lat=0.0, lon=180.0))

    assert measured == pytest.approx(20_015_000.0, rel=TOLERANCE)


def test_haversine__pole_to_pole__is_half_the_circumference() -> None:
    """A second antipodal case, along a meridian rather than the equator."""
    measured = GeoPoint(lat=90.0, lon=0.0).haversine_distance_to(GeoPoint(lat=-90.0, lon=0.0))

    assert measured == pytest.approx(20_015_000.0, rel=TOLERANCE)


def test_haversine__is_symmetric_for_a_reference_pair() -> None:
    """Distance is a metric; direction cannot change it."""
    assert LONDON.haversine_distance_to(PARIS) == pytest.approx(PARIS.haversine_distance_to(LONDON))


def test_haversine__nearby_camera_pair__is_accurate_at_city_scale() -> None:
    """The realistic case: two cameras a few hundred metres apart."""
    a = GeoPoint(lat=40.7195, lon=-74.0021)
    b = GeoPoint(lat=40.7203, lon=-73.9925)

    measured = a.haversine_distance_to(b)

    assert 700.0 < measured < 900.0
