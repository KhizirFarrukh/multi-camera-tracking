"""Moving the topology between its authoring format and its runtime home.

The YAML file is what a human edits and reviews in a diff; the database is what
the running system queries. Sync pushes one into the other.

Deletion is the delicate part. A camera removed from the file has usually been
decommissioned -- but its sightings are evidence, and evidence outlives the
hardware that recorded it. So pruning is opt-in, and a camera with sightings is
reported rather than removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from multicam_tracker.db.repositories.protocols import (
    CameraLinkRepository,
    CameraRepository,
    SightingRepository,
)
from multicam_tracker.exceptions import TopologyError
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import TimeWindow
from multicam_tracker.topology.graph import Topology, TopologyEdge

__all__ = ["SyncReport", "load_topology_from_db", "sync_topology_to_db"]

logger = get_logger(__name__)

_ALL_TIME = TimeWindow(
    start_utc=datetime(1970, 1, 1, tzinfo=UTC),
    end_utc=datetime(2200, 1, 1, tzinfo=UTC),
)
"""A window wide enough to mean "ever", used to ask whether a camera has any
sightings at all. The repository protocol offers no bare count, and inventing
one for a single administrative check would widen an interface every later stage
depends on."""


@dataclass(frozen=True)
class SyncReport:
    """What one sync changed."""

    cameras_written: int = 0
    links_written: int = 0
    cameras_deleted: list[str] = field(default_factory=list)
    cameras_retained: list[str] = field(default_factory=list)
    """Cameras absent from the file that were kept because pruning was off."""


def sync_topology_to_db(
    topology: Topology,
    camera_repo: CameraRepository,
    link_repo: CameraLinkRepository,
    *,
    sighting_repo: SightingRepository | None = None,
    prune_missing: bool = False,
) -> SyncReport:
    """Upsert a file-defined topology into the database.

    Idempotent: every write is an upsert keyed on the natural identifier, so
    running it twice changes nothing the second time.

    Only the *declared* direction of each edge is written. A bidirectional link
    is stored once with ``bidirectional=True`` and re-expanded on load, so the
    database holds what the author wrote rather than a derived duplicate that
    would drift if the original were edited.

    Args:
        topology: The graph to persist.
        camera_repo: Camera persistence.
        link_repo: Link persistence.
        sighting_repo: Needed only when ``prune_missing`` is set, to check
            whether a camera about to be deleted still holds evidence.
        prune_missing: Delete cameras present in the database but absent from
            the topology. Off by default: deleting evidence-bearing
            infrastructure should be a decision, not a side effect of a sync.

    Returns:
        A summary of what changed.

    Raises:
        TopologyError: If pruning is requested without a ``sighting_repo``, or
            if any camera due for deletion still has sightings. The offending
            camera ids are listed, and nothing is deleted -- a partial prune
            would leave the operator unsure what survived.
    """
    if prune_missing and sighting_repo is None:
        raise TopologyError(
            "Pruning requires a sighting repository so cameras holding evidence "
            "can be identified before anything is deleted",
            {"prune_missing": True},
        )

    declared_ids = {camera.camera_id for camera in topology.list_cameras()}
    existing_ids = {camera.camera_id for camera in camera_repo.list_enabled()}

    for camera in topology.list_cameras():
        camera_repo.upsert(camera)

    declared_edges = [
        edge for edge in topology.list_edges() if not edge.reversed_from_bidirectional
    ]
    for edge in declared_edges:
        link_repo.upsert(edge.link)

    removable = sorted(existing_ids - declared_ids)
    deleted: list[str] = []
    retained: list[str] = []

    # The `is not None` is redundant with the guard at the top, but it is what
    # narrows the type for the checker without an assert in production code.
    if prune_missing and removable and sighting_repo is not None:
        blocked = [
            camera_id
            for camera_id in removable
            if sighting_repo.find_by_camera_and_window(camera_id, _ALL_TIME)
        ]
        if blocked:
            raise TopologyError(
                "Cannot remove cameras that still have sightings; their recorded "
                "evidence would be orphaned",
                {"camera_ids": blocked},
            )
        for camera_id in removable:
            if camera_repo.delete(camera_id):
                deleted.append(camera_id)
    else:
        retained = removable

    report = SyncReport(
        cameras_written=len(declared_ids),
        links_written=len(declared_edges),
        cameras_deleted=deleted,
        cameras_retained=retained,
    )
    logger.info(
        "topology_synced",
        cameras_written=report.cameras_written,
        links_written=report.links_written,
        cameras_deleted=len(report.cameras_deleted),
        cameras_retained=len(report.cameras_retained),
    )
    return report


def load_topology_from_db(
    camera_repo: CameraRepository, link_repo: CameraLinkRepository
) -> Topology:
    """Build a topology from the database, the runtime source of truth.

    Bidirectional links are re-expanded here exactly as the file loader expands
    them, so a graph loaded from the database is equal to the one loaded from
    the YAML that produced it.

    Args:
        camera_repo: Camera persistence.
        link_repo: Link persistence.

    Returns:
        The graph.

    Raises:
        TopologyError: If the stored rows do not form a valid graph, which would
            mean something wrote to the tables without going through sync.
    """
    from multicam_tracker.models import CameraLink

    cameras = camera_repo.list_enabled()
    known = {camera.camera_id for camera in cameras}

    edges: list[TopologyEdge] = []
    for link in link_repo.list_all():
        if link.from_camera_id not in known or link.to_camera_id not in known:
            # A link to a disabled camera is not corruption: the camera row
            # still exists, it is just out of service. Skipping keeps the
            # runtime graph consistent with list_enabled().
            continue
        edges.append(TopologyEdge(link))
        if link.bidirectional:
            edges.append(
                TopologyEdge(
                    CameraLink(
                        from_camera_id=link.to_camera_id,
                        to_camera_id=link.from_camera_id,
                        min_travel_time_sec=link.min_travel_time_sec,
                        max_travel_time_sec=link.max_travel_time_sec,
                        distance_meters=link.distance_meters,
                        bidirectional=True,
                    ),
                    reversed_from_bidirectional=True,
                )
            )

    return Topology(cameras, edges)
