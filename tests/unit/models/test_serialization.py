"""Serialization round-trip tests, including property-based coverage.

Round-tripping is the property every other stage depends on silently: sightings
go to Postgres and come back, trajectories go out over the API, fixtures load
from JSON. If the round trip is lossy anywhere, the loss shows up as an
inexplicable behaviour change between an in-memory object and a stored one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from multicam_tracker.models import (
    Camera,
    CameraLink,
    CoverageGap,
    GeoPoint,
    MatchCandidate,
    MatchMethod,
    MCTBaseModel,
    ObjectClass,
    ReviewStatus,
    Sighting,
    Target,
    TimeWindow,
    Trajectory,
)
from tests.fixtures.factories import (
    BASE_INSTANT,
    make_camera,
    make_link,
    make_match_candidate,
    make_sighting,
    make_target,
    make_trajectory,
    unit_vector,
)

pytestmark = pytest.mark.unit

MODEL_INSTANCES: list[tuple[type[MCTBaseModel], MCTBaseModel]] = [
    (Camera, make_camera(heading_degrees=182.5, notes="Faces south.")),
    (Camera, make_camera(camera_id="cam_bare", heading_degrees=None, notes=None)),
    (CameraLink, make_link()),
    (
        Sighting,
        make_sighting(
            plate_text_raw="ABC 1234",
            plate_text_normalized="ABC1234",
            plate_confidence=0.91,
            embedding=unit_vector(seed=11),
            embedding_model_version="osnet_x1_0@v3",
            thumbnail_path="storage/thumbnails/cam_01/1.jpg",
        ),
    ),
    (Sighting, make_sighting()),
    (Target, make_target()),
    (Target, make_target(plate_query=None, reference_embeddings=[unit_vector(seed=12)])),
    (MatchCandidate, make_match_candidate()),
    (
        MatchCandidate,
        make_match_candidate(
            match_method=MatchMethod.EMBEDDING,
            plate_edit_distance=None,
            embedding_similarity=0.93,
            review_status=ReviewStatus.PENDING_REVIEW,
        ),
    ),
    (Trajectory, make_trajectory()),
    (GeoPoint, GeoPoint(lat=40.7128, lon=-74.0060)),
    (TimeWindow, TimeWindow(start_utc=BASE_INSTANT, end_utc=BASE_INSTANT + timedelta(minutes=5))),
    (
        CoverageGap,
        CoverageGap(
            from_sighting_id="s1",
            to_sighting_id="s2",
            elapsed_sec=1800.0,
            expected_max_sec=340.0,
            reason="exceeded the plausible window",
        ),
    ),
]


def _model_id(value: object) -> str:
    """Return a readable parametrize id.

    Args:
        value: A model class or instance from :data:`MODEL_INSTANCES`.

    Returns:
        The class name, or an empty string for instances.
    """
    return value.__name__ if isinstance(value, type) else ""


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("model_type", "instance"), MODEL_INSTANCES, ids=_model_id)
def test_serialization__every_model__round_trips_to_an_equal_object(
    model_type: type[MCTBaseModel], instance: MCTBaseModel
) -> None:
    """to_json_dict -> from_json_dict is lossless for every domain model."""
    assert model_type.from_json_dict(instance.to_json_dict()) == instance


@pytest.mark.parametrize(("model_type", "instance"), MODEL_INSTANCES, ids=_model_id)
def test_serialization__every_model__survives_real_json_text(
    model_type: type[MCTBaseModel], instance: MCTBaseModel
) -> None:
    """The dict is not merely JSON-shaped: it encodes and decodes as JSON text."""
    decoded = json.loads(json.dumps(instance.to_json_dict()))

    assert model_type.from_json_dict(decoded) == instance


@pytest.mark.parametrize(("model_type", "instance"), MODEL_INSTANCES, ids=_model_id)
def test_serialization__every_model__emits_only_json_native_types(
    model_type: type[MCTBaseModel], instance: MCTBaseModel
) -> None:
    """No datetime, enum, or Decimal leaks into the payload."""

    def assert_native(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert_native(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                assert_native(item, f"{path}[{index}]")
        else:
            assert value is None or isinstance(value, str | int | float | bool), (
                f"{path} is {type(value).__name__}, not a JSON-native type"
            )

    assert_native(instance.to_json_dict(), model_type.__name__)


# ---------------------------------------------------------------------------
# Field-level guarantees
# ---------------------------------------------------------------------------


def test_serialization__datetimes__use_a_z_suffix_and_millisecond_precision() -> None:
    """The wire format is fixed: exactly three decimals and a Z, never an offset."""
    payload = make_sighting().to_json_dict()

    for field in ("timestamp_utc", "raw_timestamp", "created_at"):
        assert payload[field].endswith("Z"), field
        assert "+00:00" not in payload[field], field
        assert len(payload[field].split(".")[-1]) == len("500Z"), field


def test_serialization__non_utc_input__serializes_as_utc() -> None:
    """Whatever offset came in, what goes out is UTC."""
    from datetime import timezone

    tashkent = timezone(timedelta(hours=5))
    moment = datetime(2026, 8, 10, 19, 22, 11, 500000, tzinfo=tashkent)
    sighting = make_sighting(timestamp_utc=moment, raw_timestamp=moment, created_at=moment)

    assert sighting.to_json_dict()["timestamp_utc"] == "2026-08-10T14:22:11.500Z"


def test_serialization__enums__become_plain_strings() -> None:
    """Enum members must not serialize as objects or as 'ObjectClass.CAR'."""
    payload = make_sighting(object_class=ObjectClass.TRUCK).to_json_dict()

    assert payload["object_class"] == "truck"
    assert isinstance(payload["object_class"], str)


def test_serialization__embeddings__become_plain_float_lists() -> None:
    """Vectors cross the wire as numbers, not as a nested object or a string."""
    payload = make_sighting(embedding=unit_vector(seed=13)).to_json_dict()

    assert isinstance(payload["embedding"], list)
    assert len(payload["embedding"]) == 512
    assert all(isinstance(component, float) for component in payload["embedding"])


def test_serialization__none_valued_optionals__survive_as_none() -> None:
    """Optional fields must not be dropped, or the round trip changes their meaning."""
    payload = make_sighting().to_json_dict()

    for field in ("plate_text_raw", "plate_text_normalized", "plate_confidence", "embedding"):
        assert field in payload, field
        assert payload[field] is None, field


def test_serialization__nested_models__round_trip_inside_their_parent() -> None:
    """A trajectory carries sightings and hops; both must survive intact."""
    trajectory = make_trajectory()
    payload = trajectory.to_json_dict()

    assert isinstance(payload["sightings"][0], dict)
    assert Trajectory.from_json_dict(payload).sightings[0] == trajectory.sightings[0]


def test_serialization__unknown_key_in_a_payload__is_rejected_on_load() -> None:
    """Deserialization is as strict as construction; a stale key is not ignored."""
    payload = make_camera().to_json_dict()
    payload["retired"] = True

    with pytest.raises(ValueError, match=r"Extra inputs"):
        Camera.from_json_dict(payload)


# ---------------------------------------------------------------------------
# Property-based
# ---------------------------------------------------------------------------

_EMBEDDINGS = [None, unit_vector(seed=21), unit_vector(seed=22)]
_HYPOTHESIS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def sightings(draw: st.DrawFn) -> Sighting:
    """Generate an arbitrary valid :class:`Sighting`.

    The three timestamp fields are generated together so the cross-field
    consistency invariant holds by construction rather than by rejection
    sampling.

    Args:
        draw: Hypothesis draw function.

    Returns:
        A validated sighting.
    """
    raw = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2035, 1, 1),
            timezones=st.just(UTC),
        )
    ).replace(microsecond=draw(st.integers(min_value=0, max_value=999)) * 1000)

    offset_ms = draw(st.integers(min_value=-60_000, max_value=60_000))

    x1 = draw(st.integers(min_value=0, max_value=4000))
    y1 = draw(st.integers(min_value=0, max_value=4000))
    plate = draw(
        st.one_of(st.none(), st.text(alphabet="ABCDEFGH0123456789", min_size=1, max_size=8))
    )

    return Sighting(
        camera_id=draw(st.sampled_from(["cam_01", "cam_02", "cam_03"])),
        raw_timestamp=raw,
        clock_offset_applied_ms=offset_ms,
        timestamp_utc=raw + timedelta(milliseconds=offset_ms),
        object_class=draw(st.sampled_from(list(ObjectClass))),
        detection_confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        bbox=[
            x1,
            y1,
            x1 + draw(st.integers(min_value=1, max_value=500)),
            y1 + draw(st.integers(min_value=1, max_value=500)),
        ],
        frame_index=draw(st.integers(min_value=0, max_value=10**6)),
        plate_text_raw=plate,
        plate_text_normalized=plate,
        plate_confidence=(
            None
            if plate is None
            else draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
        ),
        embedding=draw(st.sampled_from(_EMBEDDINGS)),
        source_id=draw(st.text(alphabet="abcdefg_0123456789", min_size=1, max_size=20)),
        created_at=raw,
    )


@_HYPOTHESIS
@given(sighting=sightings())
def test_serialization__arbitrary_valid_sighting__round_trips_unchanged(
    sighting: Sighting,
) -> None:
    """Property: the round trip is lossless across the whole valid input space."""
    assert Sighting.from_json_dict(sighting.to_json_dict()) == sighting


coordinates = st.builds(
    GeoPoint,
    lat=st.floats(min_value=-90.0, max_value=90.0, allow_nan=False),
    lon=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False),
)


@_HYPOTHESIS
@given(a=coordinates, b=coordinates)
def test_haversine__is_symmetric(a: GeoPoint, b: GeoPoint) -> None:
    """Property: distance does not depend on which point you start from."""
    assert a.haversine_distance_to(b) == pytest.approx(b.haversine_distance_to(a), rel=1e-9)


@_HYPOTHESIS
@given(a=coordinates, b=coordinates, c=coordinates)
def test_haversine__satisfies_the_triangle_inequality(
    a: GeoPoint, b: GeoPoint, c: GeoPoint
) -> None:
    """Property: a detour is never shorter than the direct route.

    The absolute slack absorbs floating-point error near degenerate triangles,
    where the two sides sum to the third within a rounding error.
    """
    direct = a.haversine_distance_to(c)
    via_b = a.haversine_distance_to(b) + b.haversine_distance_to(c)

    assert direct <= via_b + 1e-6


@_HYPOTHESIS
@given(point=coordinates)
def test_haversine__distance_to_self__is_zero(point: GeoPoint) -> None:
    """Property: identity holds everywhere, not only at the reference points."""
    assert point.haversine_distance_to(point) == 0.0
