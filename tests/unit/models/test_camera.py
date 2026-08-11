"""Unit tests for :mod:`multicam_tracker.models.camera`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multicam_tracker.models import Camera, CameraLink
from tests.fixtures.factories import make_camera, make_link

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_camera__valid_input__constructs() -> None:
    """The happy path, with every optional field supplied."""
    camera = Camera(
        camera_id="cam_03",
        name="Main St & 5th Ave",
        lat=40.7128,
        lon=-74.0060,
        heading_degrees=182.5,
        clock_offset_ms=-250,
        enabled=True,
        notes="Pole-mounted, faces south.",
    )

    assert camera.camera_id == "cam_03"
    assert camera.heading_degrees == pytest.approx(182.5)


def test_camera__optional_fields_omitted__uses_defaults() -> None:
    """A camera needs only an id, a name, and a position."""
    camera = Camera(camera_id="cam_01", name="Corner", lat=0.0, lon=0.0)

    assert camera.heading_degrees is None
    assert camera.clock_offset_ms == 0
    assert camera.enabled is True
    assert camera.notes is None


@pytest.mark.parametrize("lat", [90.0, -90.0, 0.0], ids=["north-pole", "south-pole", "equator"])
def test_camera__latitude_at_the_bounds__is_accepted(lat: float) -> None:
    """Boundary: the poles are valid latitudes."""
    assert make_camera(lat=lat).lat == pytest.approx(lat)


@pytest.mark.parametrize("lat", [90.1, -90.1], ids=["above", "below"])
def test_camera__latitude_out_of_range__is_rejected(lat: float) -> None:
    """A latitude outside [-90, 90] is not a coordinate."""
    with pytest.raises(ValidationError):
        make_camera(lat=lat)


@pytest.mark.parametrize(
    "lon", [180.0, -180.0, 0.0], ids=["antimeridian+", "antimeridian-", "prime"]
)
def test_camera__longitude_at_the_bounds__is_accepted(lon: float) -> None:
    """Boundary: both sides of the antimeridian are valid."""
    assert make_camera(lon=lon).lon == pytest.approx(lon)


@pytest.mark.parametrize("lon", [180.1, -180.1], ids=["above", "below"])
def test_camera__longitude_out_of_range__is_rejected(lon: float) -> None:
    """A longitude outside [-180, 180] is not a coordinate."""
    with pytest.raises(ValidationError):
        make_camera(lon=lon)


@pytest.mark.parametrize("heading", [0.0, 359.9], ids=["zero", "just-under-360"])
def test_camera__heading_inside_the_half_open_range__is_accepted(heading: float) -> None:
    """Boundary: the range is [0, 360)."""
    assert make_camera(heading_degrees=heading).heading_degrees == pytest.approx(heading)


def test_camera__heading_of_exactly_360__is_rejected() -> None:
    """360 aliases 0; allowing both would make two headings compare unequal."""
    with pytest.raises(ValidationError):
        make_camera(heading_degrees=360.0)


def test_camera__negative_heading__is_rejected() -> None:
    """Headings are expressed in [0, 360), not as signed bearings."""
    with pytest.raises(ValidationError):
        make_camera(heading_degrees=-1.0)


@pytest.mark.parametrize(
    "camera_id",
    ["cam_03", "cam-03", "camera03", "c", "a1_b2-c3"],
    ids=["underscore", "hyphen", "alnum", "single-char", "mixed"],
)
def test_camera__id_matching_the_pattern__is_accepted(camera_id: str) -> None:
    """Lowercase alphanumerics, underscores, and hyphens are the allowed alphabet."""
    assert make_camera(camera_id=camera_id).camera_id == camera_id


@pytest.mark.parametrize(
    "camera_id",
    ["Cam_03", "CAM03", "cam 03", "cam.03", "cam/03", "cam@03", ""],
    ids=["mixed-case", "upper", "space", "dot", "slash", "at", "empty"],
)
def test_camera__id_violating_the_pattern__is_rejected(camera_id: str) -> None:
    """Uppercase or punctuation would let two spellings denote one camera."""
    with pytest.raises(ValidationError):
        make_camera(camera_id=camera_id)


def test_camera__empty_name__is_rejected() -> None:
    """A blank label is useless in the UI and almost always a data error."""
    with pytest.raises(ValidationError):
        make_camera(name="   ")


def test_camera__unknown_extra_field__is_rejected() -> None:
    """extra='forbid' catches a renamed or misspelled key at construction."""
    with pytest.raises(ValidationError) as excinfo:
        Camera(camera_id="cam_01", name="Corner", lat=0.0, lon=0.0, latitude=1.0)

    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())


def test_camera__negative_clock_offset__is_accepted() -> None:
    """A camera running fast needs a negative correction."""
    assert make_camera(clock_offset_ms=-1500).clock_offset_ms == -1500


# ---------------------------------------------------------------------------
# CameraLink
# ---------------------------------------------------------------------------


def test_camera_link__valid_input__constructs() -> None:
    """The happy path."""
    link = CameraLink(
        from_camera_id="cam_01",
        to_camera_id="cam_02",
        min_travel_time_sec=45.0,
        max_travel_time_sec=300.0,
        distance_meters=815.0,
        bidirectional=True,
    )

    assert link.min_travel_time_sec == pytest.approx(45.0)
    assert link.bidirectional is True


def test_camera_link__zero_minimum_travel_time__is_accepted() -> None:
    """Boundary: adjacent cameras with overlapping fields of view."""
    assert make_link(min_travel_time_sec=0.0).min_travel_time_sec == 0.0


def test_camera_link__negative_minimum_travel_time__is_rejected() -> None:
    """Time does not run backwards between cameras."""
    with pytest.raises(ValidationError):
        make_link(min_travel_time_sec=-1.0)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(60.0, 60.0), (300.0, 45.0)],
    ids=["equal", "inverted"],
)
def test_camera_link__max_not_above_min__is_rejected_naming_both(
    minimum: float, maximum: float
) -> None:
    """A transposed pair in a topology file is the common cause, so name both."""
    with pytest.raises(ValidationError) as excinfo:
        make_link(min_travel_time_sec=minimum, max_travel_time_sec=maximum)

    message = str(excinfo.value)
    assert str(minimum) in message
    assert str(maximum) in message


def test_camera_link__self_link__is_rejected() -> None:
    """A hop is a transition between cameras; a self-link would loop forever."""
    with pytest.raises(ValidationError, match="must differ"):
        make_link(from_camera_id="cam_01", to_camera_id="cam_01")


def test_camera_link__invalid_camera_id__is_rejected() -> None:
    """Endpoints obey the same identifier rules as Camera.camera_id."""
    with pytest.raises(ValidationError):
        make_link(to_camera_id="Cam_02")


def test_camera_link__negative_distance__is_rejected() -> None:
    """Distance is a magnitude."""
    with pytest.raises(ValidationError):
        make_link(distance_meters=-5.0)


def test_camera_link__distance_omitted__defaults_to_none() -> None:
    """Distance is optional: a topology may be authored from travel times alone."""
    link = CameraLink(
        from_camera_id="cam_01",
        to_camera_id="cam_02",
        min_travel_time_sec=10.0,
        max_travel_time_sec=20.0,
    )

    assert link.distance_meters is None
    assert link.bidirectional is True
