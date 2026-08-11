"""Unit tests for :mod:`multicam_tracker.models.sighting`.

The sighting is the atomic record of the system, so its rejection paths are
tested as thoroughly as its acceptance paths. Every invariant asserted here is
one a later stage gets to assume without re-checking.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from multicam_tracker.models import ObjectClass, Sighting
from tests.fixtures.factories import BASE_INSTANT, make_sighting, unit_vector

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_sighting__fully_populated__constructs() -> None:
    """Every optional field supplied at once."""
    sighting = make_sighting(
        "cam_03",
        plate_text_raw="ABC 1234",
        plate_text_normalized="ABC1234",
        plate_confidence=0.91,
        embedding=unit_vector(seed=1),
        embedding_model_version="osnet_x1_0@v3",
        thumbnail_path="storage/thumbnails/cam_03/1284.jpg",
    )

    assert sighting.has_plate is True
    assert sighting.has_embedding is True
    assert sighting.object_class is ObjectClass.CAR


def test_sighting__minimal_input__constructs_without_plate_or_embedding() -> None:
    """Most sightings carry neither: the plate was unreadable and re-id is off."""
    sighting = make_sighting()

    assert sighting.has_plate is False
    assert sighting.has_embedding is False
    assert sighting.plate_confidence is None


def test_sighting__object_class_omitted__defaults_to_unknown() -> None:
    """The detector must be able to decline to classify."""
    sighting = Sighting(
        camera_id="cam_01",
        timestamp_utc=BASE_INSTANT,
        raw_timestamp=BASE_INSTANT,
        detection_confidence=0.5,
        bbox=[0, 0, 10, 10],
        frame_index=0,
        source_id="clip",
        created_at=BASE_INSTANT,
    )

    assert sighting.object_class is ObjectClass.UNKNOWN


def test_sighting__id_omitted__is_generated_and_unique() -> None:
    """Ingestion does not have to mint identifiers by hand."""
    assert make_sighting().sighting_id != make_sighting().sighting_id


# ---------------------------------------------------------------------------
# bbox
# ---------------------------------------------------------------------------


def test_sighting__minimal_positive_area_bbox__is_accepted() -> None:
    """Boundary: a one-pixel box is degenerate but not invalid."""
    assert make_sighting(bbox=[0, 0, 1, 1]).bbox == [0, 0, 1, 1]


@pytest.mark.parametrize(
    ("bbox", "reason"),
    [
        ([300, 100, 200, 400], "x2 > x1"),
        ([100, 100, 100, 400], "x2 > x1"),
        ([100, 400, 300, 200], "y2 > y1"),
        ([100, 100, 300, 100], "y2 > y1"),
    ],
    ids=["x-inverted", "x-degenerate", "y-inverted", "y-degenerate"],
)
def test_sighting__bbox_without_positive_area__is_rejected(bbox: list[int], reason: str) -> None:
    """A zero- or negative-area box crops to nothing and would fail far downstream."""
    with pytest.raises(ValidationError, match=reason):
        make_sighting(bbox=bbox)


@pytest.mark.parametrize(
    "bbox",
    [[-1, 0, 10, 10], [0, -1, 10, 10], [0, 0, -10, 10], [0, 0, 10, -10]],
    ids=["x1", "y1", "x2", "y2"],
)
def test_sighting__bbox_with_a_negative_coordinate__is_rejected(bbox: list[int]) -> None:
    """Pixel coordinates start at the origin."""
    with pytest.raises(ValidationError, match="non-negative"):
        make_sighting(bbox=bbox)


@pytest.mark.parametrize(
    "bbox", [[], [0, 0], [0, 0, 10], [0, 0, 10, 10, 10]], ids=["empty", "two", "three", "five"]
)
def test_sighting__bbox_of_the_wrong_length__is_rejected(bbox: list[int]) -> None:
    """A bbox is exactly [x1, y1, x2, y2]."""
    with pytest.raises(ValidationError):
        make_sighting(bbox=bbox)


# ---------------------------------------------------------------------------
# Confidences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.0, 1.0], ids=["zero", "one"])
def test_sighting__detection_confidence_at_the_bounds__is_accepted(confidence: float) -> None:
    """Boundary: the closed interval [0, 1] is the full probability range."""
    assert make_sighting(detection_confidence=confidence).detection_confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01], ids=["below", "above"])
def test_sighting__detection_confidence_out_of_range__is_rejected(confidence: float) -> None:
    """A confidence outside [0, 1] is not a probability."""
    with pytest.raises(ValidationError):
        make_sighting(detection_confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0], ids=["zero", "one"])
def test_sighting__plate_confidence_at_the_bounds__is_accepted(confidence: float) -> None:
    """The same closed interval applies to the OCR confidence."""
    sighting = make_sighting(plate_text_normalized="ABC1234", plate_confidence=confidence)

    assert sighting.plate_confidence == confidence


@pytest.mark.parametrize("confidence", [-0.01, 1.01], ids=["below", "above"])
def test_sighting__plate_confidence_out_of_range__is_rejected(confidence: float) -> None:
    """Out-of-range OCR confidence usually means an unnormalized score."""
    with pytest.raises(ValidationError):
        make_sighting(plate_text_normalized="ABC1234", plate_confidence=confidence)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_sighting__normalized_embedding__is_accepted() -> None:
    """A unit vector of the configured dimension passes."""
    assert make_sighting(embedding=unit_vector(seed=5)).has_embedding is True


def test_sighting__embedding_of_the_wrong_dimension__names_expected_and_actual() -> None:
    """A dimension mismatch usually means the wrong extractor produced the vector."""
    with pytest.raises(ValidationError) as excinfo:
        make_sighting(embedding=unit_vector(seed=5, dim=256))

    message = str(excinfo.value)
    assert "256" in message
    assert "512" in message


def test_sighting__unnormalized_embedding__is_rejected_and_not_renormalized() -> None:
    """The stored value must never differ from what the extractor produced."""
    scaled = [component * 3.0 for component in unit_vector(seed=6)]

    with pytest.raises(ValidationError, match="not L2-normalized"):
        make_sighting(embedding=scaled)


def test_sighting__embedding_at_the_normalization_tolerance_boundary__is_accepted() -> None:
    """Boundary: 1e-3 of slack absorbs float32 round-off from a real extractor."""
    scaled = [component * 1.001 for component in unit_vector(seed=7)]

    assert make_sighting(embedding=scaled).has_embedding is True


# ---------------------------------------------------------------------------
# Timestamp consistency
# ---------------------------------------------------------------------------


def test_sighting__timestamp_consistent_with_a_positive_offset__is_accepted() -> None:
    """The redundant trio must agree: raw + offset == corrected."""
    raw = BASE_INSTANT
    sighting = make_sighting(
        raw_timestamp=raw,
        clock_offset_applied_ms=2000,
        timestamp_utc=raw + timedelta(milliseconds=2000),
    )

    assert sighting.timestamp_utc == raw + timedelta(seconds=2)


def test_sighting__timestamp_consistent_with_a_negative_offset__is_accepted() -> None:
    """A camera running fast is corrected backwards."""
    raw = BASE_INSTANT
    sighting = make_sighting(
        raw_timestamp=raw,
        clock_offset_applied_ms=-1500,
        timestamp_utc=raw - timedelta(milliseconds=1500),
    )

    assert sighting.timestamp_utc == raw - timedelta(milliseconds=1500)


def test_sighting__timestamp_inconsistent_with_the_offset__is_rejected() -> None:
    """Inconsistency here would make stage 09's drift audit meaningless."""
    with pytest.raises(ValidationError, match="does not equal"):
        make_sighting(
            raw_timestamp=BASE_INSTANT,
            clock_offset_applied_ms=2000,
            timestamp_utc=BASE_INSTANT + timedelta(milliseconds=9000),
        )


def test_sighting__timestamp_off_by_exactly_the_tolerance__is_accepted() -> None:
    """Boundary: 1 ms of disagreement is inside the permitted tolerance."""
    sighting = make_sighting(
        raw_timestamp=BASE_INSTANT,
        clock_offset_applied_ms=0,
        timestamp_utc=BASE_INSTANT + timedelta(milliseconds=1),
    )

    assert sighting.timestamp_utc == BASE_INSTANT + timedelta(milliseconds=1)


def test_sighting__timestamp_off_by_more_than_the_tolerance__is_rejected() -> None:
    """One notch past the boundary fails, so the boundary is real."""
    with pytest.raises(ValidationError, match="does not equal"):
        make_sighting(
            raw_timestamp=BASE_INSTANT,
            clock_offset_applied_ms=0,
            timestamp_utc=BASE_INSTANT + timedelta(milliseconds=2),
        )


@pytest.mark.parametrize(
    "field", ["timestamp_utc", "raw_timestamp", "created_at"], ids=["corrected", "raw", "created"]
)
def test_sighting__naive_datetime_in_any_field__is_rejected(field: str) -> None:
    """No model accepts a naive datetime anywhere."""
    with pytest.raises(ValidationError, match="naive datetime"):
        make_sighting(**{field: BASE_INSTANT.replace(tzinfo=None)})


# ---------------------------------------------------------------------------
# Plate coupling
# ---------------------------------------------------------------------------


def test_sighting__normalized_plate_without_confidence__is_rejected() -> None:
    """Matching weights every read by its confidence; an unweighted read has no place."""
    with pytest.raises(ValidationError, match="plate_confidence is required"):
        make_sighting(plate_text_normalized="ABC1234", plate_confidence=None)


def test_sighting__raw_plate_without_normalized__is_accepted() -> None:
    """A read too garbled to normalize is still worth keeping for audit."""
    sighting = make_sighting(plate_text_raw="A?C 1|34")

    assert sighting.plate_text_raw == "A?C 1|34"
    assert sighting.has_plate is False


def test_sighting__confidence_without_a_plate__is_accepted() -> None:
    """The asymmetry is deliberate: only the reverse direction is contradictory."""
    assert make_sighting(plate_confidence=0.4).has_plate is False


# ---------------------------------------------------------------------------
# Computed properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("with_plate", "with_embedding"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["neither", "plate-only", "embedding-only", "both"],
)
def test_sighting__has_plate_and_has_embedding__cover_all_four_combinations(
    with_plate: bool, with_embedding: bool
) -> None:
    """All four states occur in practice and each drives a different match path."""
    overrides: dict[str, object] = {}
    if with_plate:
        overrides |= {"plate_text_normalized": "ABC1234", "plate_confidence": 0.9}
    if with_embedding:
        overrides["embedding"] = unit_vector(seed=8)

    sighting = make_sighting(**overrides)

    assert sighting.has_plate is with_plate
    assert sighting.has_embedding is with_embedding


# ---------------------------------------------------------------------------
# Misc field constraints
# ---------------------------------------------------------------------------


def test_sighting__negative_frame_index__is_rejected() -> None:
    """Frame indices start at zero."""
    with pytest.raises(ValidationError):
        make_sighting(frame_index=-1)


def test_sighting__empty_source_id__is_rejected() -> None:
    """Without a source id a sighting cannot be traced back to its footage."""
    with pytest.raises(ValidationError):
        make_sighting(source_id="  ")


def test_sighting__unknown_extra_field__is_rejected() -> None:
    """A renamed contract field must fail loudly rather than be dropped."""
    with pytest.raises(ValidationError) as excinfo:
        make_sighting(timestamp="2026-08-10T14:22:11.500Z")

    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())
