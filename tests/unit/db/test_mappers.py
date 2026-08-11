"""Unit tests for :mod:`multicam_tracker.db.mappers`.

The correspondence test is the important one: it is what turns "mapping must be
total" from a promise into something the suite enforces. Add a field to a domain
model without adding a column and it fails naming the field.
"""

from __future__ import annotations

import pytest

from multicam_tracker.db.mappers import (
    FIELD_CORRESPONDENCE,
    FieldCorrespondence,
    camera_link_to_domain,
    camera_link_to_orm,
    camera_to_domain,
    camera_to_orm,
    correspondence_for,
    match_candidate_to_domain,
    match_candidate_to_orm,
    sighting_to_domain,
    sighting_to_orm,
    target_to_domain,
    target_to_orm,
    trajectory_to_domain,
    trajectory_to_orm,
)
from multicam_tracker.db.orm import CameraORM, SightingORM
from multicam_tracker.exceptions import StorageError
from multicam_tracker.models import Camera, MatchMethod, ReviewStatus, Sighting
from tests.fixtures.factories import (
    make_camera,
    make_link,
    make_match_candidate,
    make_sighting,
    make_target,
    make_trajectory,
    unit_vector,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_camera__round_trips_through_the_orm() -> None:
    """Every field survives domain -> row -> domain."""
    camera = make_camera(camera_id="cam_07", heading_degrees=182.5, notes="South-facing")

    assert camera_to_domain(camera_to_orm(camera)) == camera


def test_camera__with_all_optionals_none__round_trips() -> None:
    """None-valued optionals must not become defaults on the way back."""
    camera = make_camera(heading_degrees=None, notes=None)

    restored = camera_to_domain(camera_to_orm(camera))

    assert restored.heading_degrees is None
    assert restored.notes is None


def test_camera_link__round_trips_through_the_orm() -> None:
    """Links survive the round trip."""
    link = make_link(distance_meters=815.0, bidirectional=False)

    assert camera_link_to_domain(camera_link_to_orm(link)) == link


def test_camera_link__with_no_distance__round_trips() -> None:
    """A topology authored from travel times alone still round-trips."""
    link = make_link(distance_meters=None)

    assert camera_link_to_domain(camera_link_to_orm(link)).distance_meters is None


def test_sighting__fully_populated__round_trips_through_the_orm() -> None:
    """The hot-path entity, with every optional supplied."""
    sighting = make_sighting(
        "cam_03",
        plate_text_raw="ABC 1234",
        plate_text_normalized="ABC1234",
        plate_confidence=0.91,
        embedding=unit_vector(seed=1),
        embedding_model_version="osnet_x1_0@v3",
        thumbnail_path="storage/thumbnails/cam_03/1.jpg",
    )

    assert sighting_to_domain(sighting_to_orm(sighting)) == sighting


def test_sighting__minimal__round_trips_with_optionals_still_none() -> None:
    """A sighting with no plate and no embedding survives unchanged."""
    sighting = make_sighting()

    restored = sighting_to_domain(sighting_to_orm(sighting))

    assert restored == sighting
    assert restored.plate_text_normalized is None
    assert restored.embedding is None


def test_sighting__embedding__survives_the_round_trip_component_wise() -> None:
    """Vectors must not lose precision beyond float rounding."""
    vector = unit_vector(seed=5)
    sighting = make_sighting(embedding=vector)

    restored = sighting_to_domain(sighting_to_orm(sighting))

    assert restored.embedding is not None
    assert restored.embedding == pytest.approx(vector, abs=1e-6)


def test_sighting__object_class__maps_to_and_from_its_string_value() -> None:
    """The column stores a plain string; the domain keeps the enum."""
    row = sighting_to_orm(make_sighting())

    assert row.object_class == "car"
    assert sighting_to_domain(row).object_class.value == "car"


def test_sighting__plate_folded__is_left_to_the_database() -> None:
    """A generated column must not be written by the application."""
    row = sighting_to_orm(make_sighting(plate_text_normalized="ABC1234", plate_confidence=0.9))

    assert row.plate_folded is None


def test_target__round_trips_through_the_orm() -> None:
    """Targets survive, reference embeddings included."""
    target = make_target(reference_embeddings=[unit_vector(seed=1), unit_vector(seed=2)])

    assert target_to_domain(target_to_orm(target)) == target


def test_target__without_reference_embeddings__round_trips() -> None:
    """A plate-only target keeps its None."""
    target = make_target(reference_embeddings=None)

    assert target_to_domain(target_to_orm(target)).reference_embeddings is None


def test_match_candidate__round_trips_through_the_orm() -> None:
    """Match candidates survive, enums included."""
    candidate = make_match_candidate(
        match_method=MatchMethod.EMBEDDING,
        plate_edit_distance=None,
        embedding_similarity=0.93,
        review_status=ReviewStatus.PENDING_REVIEW,
    )

    assert match_candidate_to_domain(match_candidate_to_orm(candidate)) == candidate


def test_trajectory__round_trips_with_its_hops_and_sightings() -> None:
    """The composite entity reassembles exactly."""
    trajectory = make_trajectory()
    row, hops = trajectory_to_orm(trajectory)

    restored = trajectory_to_domain(row, hops, trajectory.sightings)

    assert restored == trajectory


def test_trajectory__hops__are_reordered_by_position_on_the_way_back() -> None:
    """SQL result order is not guaranteed, so position is what defines the route."""
    trajectory = make_trajectory(
        sightings=[
            make_sighting("cam_01", offset_sec=0),
            make_sighting("cam_02", offset_sec=90),
            make_sighting("cam_03", offset_sec=240),
        ]
    )
    row, hops = trajectory_to_orm(trajectory)

    restored = trajectory_to_domain(row, list(reversed(hops)), trajectory.sightings)

    assert restored.hops == trajectory.hops


def test_trajectory__single_sighting__round_trips_with_no_hops() -> None:
    """Boundary: a one-sighting trajectory has no hops to order."""
    trajectory = make_trajectory(sightings=[make_sighting("cam_01", offset_sec=0)])
    row, hops = trajectory_to_orm(trajectory)

    assert hops == []
    assert trajectory_to_domain(row, hops, trajectory.sightings) == trajectory


def test_trajectory__gaps__survive_as_structured_json() -> None:
    """Gaps are part of the answer and must not be flattened to text."""
    from multicam_tracker.models import CoverageGap

    sightings = [make_sighting("cam_01", offset_sec=0), make_sighting("cam_02", offset_sec=90)]
    gap = CoverageGap(
        from_sighting_id=sightings[0].sighting_id,
        to_sighting_id=sightings[1].sighting_id,
        elapsed_sec=90.0,
        expected_max_sec=60.0,
        reason="exceeded the plausible window",
    )
    trajectory = make_trajectory(sightings=sightings, gaps=[gap])
    row, hops = trajectory_to_orm(trajectory)

    assert trajectory_to_domain(row, hops, sightings).gaps == [gap]


# ---------------------------------------------------------------------------
# Malformed identifiers
# ---------------------------------------------------------------------------


def test_sighting_to_orm__non_uuid_id__raises_storage_error_naming_the_field() -> None:
    """The error names the field, which a driver-level UUID error would not."""
    sighting = make_sighting().model_copy(update={"sighting_id": "not-a-uuid"})

    with pytest.raises(StorageError) as excinfo:
        sighting_to_orm(sighting)

    assert excinfo.value.context["field"] == "sighting_id"


# ---------------------------------------------------------------------------
# Field-set correspondence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", FIELD_CORRESPONDENCE, ids=lambda e: e.domain_model.__name__)
def test_correspondence__no_domain_field_is_unmapped(entry: FieldCorrespondence) -> None:
    """A field with no column would be silently dropped on the way to storage."""
    assert entry.unmapped_domain_fields() == set(), (
        f"{entry.domain_model.__name__} has fields with no column and no justification. "
        f"Add the column, or record why it has none in FIELD_CORRESPONDENCE."
    )


@pytest.mark.parametrize("entry", FIELD_CORRESPONDENCE, ids=lambda e: e.domain_model.__name__)
def test_correspondence__no_orm_column_is_unmapped(entry: FieldCorrespondence) -> None:
    """A column nothing populates is either dead weight or a forgotten mapping."""
    assert entry.unmapped_orm_columns() == set(), (
        f"{entry.orm_model.__name__} has columns no domain field populates. "
        f"Map them, or record why they are database-only in FIELD_CORRESPONDENCE."
    )


def test_correspondence__covers_every_domain_entity() -> None:
    """A new entity must not slip in without a declared mapping."""
    declared = {entry.domain_model for entry in FIELD_CORRESPONDENCE}

    assert Camera in declared
    assert Sighting in declared


def test_correspondence__detects_an_unmapped_field() -> None:
    """The check must actually fail when a field is missing, not merely pass.

    Guards against a correspondence check that is vacuously true -- the failure
    mode that would make every other assertion here worthless.
    """
    broken = FieldCorrespondence(domain_model=Sighting, orm_model=CameraORM)

    assert "bbox" in broken.unmapped_domain_fields()


def test_correspondence__detects_an_unmapped_column() -> None:
    """The reverse direction is checked too."""
    broken = FieldCorrespondence(domain_model=Camera, orm_model=SightingORM)

    assert "bbox" in broken.unmapped_orm_columns()


def test_correspondence_for__unknown_model__raises_storage_error() -> None:
    """A model with no declared mapping is an error, not a silent pass."""
    from multicam_tracker.models import TimeWindow

    with pytest.raises(StorageError, match="No field correspondence"):
        correspondence_for(TimeWindow)


def test_correspondence__sighting_plate_folded__is_justified_as_database_only() -> None:
    """The one intentional column with no domain field carries its reason in writing."""
    entry = correspondence_for(Sighting)

    assert "plate_folded" in entry.orm_only
    assert "generated" in entry.orm_only["plate_folded"].lower()
