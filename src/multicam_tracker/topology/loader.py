"""Loading, validating, and deriving the camera topology from YAML.

The YAML file is the *authoring* format: a human writes it, reviews it in a diff,
and reasons about it. The database is the runtime source of truth
(:mod:`multicam_tracker.topology.sync` moves one to the other).

Two classes of problem are treated differently on purpose.

**Errors** are things that make the graph meaningless: a link to a camera that
does not exist, two definitions of the same edge, a window whose maximum is
below its minimum. These raise, because loading a broken graph would silently
produce plausible-looking wrong answers rather than failing.

**Warnings** are things that are suspicious but legitimate: a camera with no
links (newly installed, not yet surveyed), or a link implying 250 km/h (a
motorway, or a typo). These load and are reported, because refusing to start
over a possibly-correct oddity is worse than surfacing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.logging_config import get_logger
from multicam_tracker.models import Camera, CameraLink, GeoPoint
from multicam_tracker.topology.graph import Topology, TopologyEdge

__all__ = [
    "SpeedModel",
    "TopologyLoadResult",
    "TopologyWarning",
    "derive_travel_window",
    "load_topology",
    "load_topology_result",
]

logger = get_logger(__name__)

SECONDS_PER_HOUR = 3600.0
METERS_PER_KILOMETER = 1000.0


@dataclass(frozen=True)
class SpeedModel:
    """How straight-line distance becomes a plausible travel-time window.

    Read from the ``defaults`` block of the topology file rather than from
    application settings, because it describes one particular road network. A
    dense city centre and a motorway corridor need different numbers, and both
    may be authored by the same deployment.

    Args:
        min_speed_kph: Slowest plausible average speed. Sets the *upper* travel
            time -- the slower a vehicle may go, the longer it may plausibly
            take.
        max_speed_kph: Fastest plausible average speed. Sets the *lower* travel
            time.
        winding_factor: Multiplier converting great-circle distance to road
            distance. Roads do not run in straight lines; 1.3 is a common
            planning figure for urban grids.
        additive_slack_sec: Added to the upper bound for stops, signals, and
            queueing -- delays that do not scale with distance and would
            otherwise make short links absurdly tight.
    """

    min_speed_kph: float = 10.0
    max_speed_kph: float = 90.0
    winding_factor: float = 1.3
    additive_slack_sec: float = 60.0

    def __post_init__(self) -> None:
        """Validate the model.

        Raises:
            TopologyError: If any factor is non-positive, or the speed range is
                inverted.
        """
        for name, value in (
            ("min_speed_kph", self.min_speed_kph),
            ("max_speed_kph", self.max_speed_kph),
            ("winding_factor", self.winding_factor),
        ):
            if value <= 0:
                raise TopologyError(
                    "Speed model factor must be positive", {"field": name, "value": value}
                )
        if self.additive_slack_sec < 0:
            raise TopologyError(
                "Speed model slack cannot be negative",
                {"field": "additive_slack_sec", "value": self.additive_slack_sec},
            )
        if self.max_speed_kph <= self.min_speed_kph:
            raise TopologyError(
                "max_speed_kph must exceed min_speed_kph",
                {"min_speed_kph": self.min_speed_kph, "max_speed_kph": self.max_speed_kph},
            )


@dataclass(frozen=True)
class TopologyWarning:
    """A suspicious but non-fatal finding from loading a topology."""

    kind: str
    """``isolated_camera`` or ``implausible_speed``."""

    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyLoadResult:
    """A loaded topology together with everything questionable about it."""

    topology: Topology
    warnings: list[TopologyWarning] = field(default_factory=list)


def derive_travel_window(
    origin: Camera, destination: Camera, model: SpeedModel
) -> tuple[float, float, float]:
    """Derive a travel-time window from the distance between two cameras.

    ::

        road_distance = great_circle_distance * winding_factor
        min_travel = road_distance / max_speed
        max_travel = road_distance / min_speed + slack

    Two cameras at identical coordinates yield ``(0.0, slack)`` rather than a
    zero-width window: co-located cameras genuinely have no minimum transit,
    but a window of zero width would accept nothing at all.

    Args:
        origin: Source camera.
        destination: Destination camera.
        model: The speed model to apply.

    Returns:
        ``(min_travel_time_sec, max_travel_time_sec, straight_line_meters)``.
        The reported distance is the true great-circle distance, not the
        winding-adjusted estimate -- one is a fact, the other an assumption.

    Raises:
        TopologyError: If the derived window is degenerate, which can only
            happen with a zero slack on co-located cameras.
    """
    straight_line_m = GeoPoint(lat=origin.lat, lon=origin.lon).haversine_distance_to(
        GeoPoint(lat=destination.lat, lon=destination.lon)
    )
    road_distance_m = straight_line_m * model.winding_factor

    max_speed_mps = model.max_speed_kph * METERS_PER_KILOMETER / SECONDS_PER_HOUR
    min_speed_mps = model.min_speed_kph * METERS_PER_KILOMETER / SECONDS_PER_HOUR

    min_travel = road_distance_m / max_speed_mps
    max_travel = road_distance_m / min_speed_mps + model.additive_slack_sec

    if not math.isfinite(max_travel) or max_travel <= min_travel:
        raise TopologyError(
            "Derived travel window is degenerate; increase additive_slack_sec",
            {
                "from_camera_id": origin.camera_id,
                "to_camera_id": destination.camera_id,
                "min_travel_time_sec": min_travel,
                "max_travel_time_sec": max_travel,
            },
        )

    return min_travel, max_travel, straight_line_m


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read and parse the topology file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.

    Raises:
        TopologyError: If the file is missing, unreadable, not valid YAML, or
            does not parse to a non-empty mapping. Raised rather than letting a
            raw parser exception escape, so callers only catch one hierarchy.
    """
    if not path.is_file():
        raise TopologyError("Topology file not found", {"path": str(path)})

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TopologyError(
            "Topology file could not be parsed",
            {"path": str(path), "reason": str(exc)},
        ) from exc

    if raw is None:
        raise TopologyError("Topology file is empty", {"path": str(path)})
    if not isinstance(raw, dict):
        raise TopologyError(
            "Topology file must contain a YAML mapping",
            {"path": str(path), "parsed_type": type(raw).__name__},
        )
    return raw


def _build_camera(entry: Any, index: int) -> Camera:
    """Validate one camera entry.

    Args:
        entry: The raw mapping from the file.
        index: Its position, reported in errors.

    Returns:
        The validated camera.

    Raises:
        TopologyError: If the entry is malformed.
    """
    if not isinstance(entry, dict):
        raise TopologyError(
            "Camera entry must be a mapping",
            {"index": index, "parsed_type": type(entry).__name__},
        )
    try:
        return Camera.model_validate(entry)
    except ValidationError as exc:
        raise TopologyError(
            "Invalid camera entry",
            {"index": index, "camera_id": entry.get("camera_id"), "reason": str(exc)},
        ) from exc


def _link_endpoints(entry: dict[str, Any], index: int) -> tuple[str, str]:
    """Extract and check the endpoint ids of a link entry.

    Args:
        entry: The raw mapping.
        index: Its position, reported in errors.

    Returns:
        ``(from_camera_id, to_camera_id)``.

    Raises:
        TopologyError: If either endpoint is missing or they are equal.
    """
    origin = entry.get("from_camera_id")
    destination = entry.get("to_camera_id")

    if not isinstance(origin, str) or not isinstance(destination, str):
        raise TopologyError(
            "Link entry needs both from_camera_id and to_camera_id",
            {"index": index, "entry": entry},
        )
    if origin == destination:
        raise TopologyError(
            "Self-link is not a transition between cameras",
            {"index": index, "camera_id": origin},
        )
    return origin, destination


def _build_link(
    entry: Any,
    index: int,
    cameras: dict[str, Camera],
    model: SpeedModel,
) -> tuple[CameraLink, bool]:
    """Validate one link entry, deriving its window when it omits one.

    Args:
        entry: The raw mapping from the file.
        index: Its position, reported in errors.
        cameras: Cameras already loaded, used to resolve endpoints.
        model: Speed model for derivation.

    Returns:
        ``(link, derived)``.

    Raises:
        TopologyError: If the entry is malformed, names an unknown camera,
            supplies only one of the two travel times, or declares an inverted
            window.
    """
    if not isinstance(entry, dict):
        raise TopologyError(
            "Link entry must be a mapping",
            {"index": index, "parsed_type": type(entry).__name__},
        )

    origin_id, destination_id = _link_endpoints(entry, index)

    for camera_id, role in ((origin_id, "from_camera_id"), (destination_id, "to_camera_id")):
        if camera_id not in cameras:
            raise TopologyError(
                "Link references an unknown camera",
                {
                    "index": index,
                    "link": f"{origin_id} -> {destination_id}",
                    "field": role,
                    "missing_camera_id": camera_id,
                },
            )

    minimum = entry.get("min_travel_time_sec")
    maximum = entry.get("max_travel_time_sec")
    derived = minimum is None and maximum is None

    if derived:
        minimum, maximum, straight_line_m = derive_travel_window(
            cameras[origin_id], cameras[destination_id], model
        )
        distance = entry.get("distance_meters", straight_line_m)
    else:
        if minimum is None or maximum is None:
            raise TopologyError(
                "A link must supply both travel times or neither; "
                "a half-specified window cannot be completed unambiguously",
                {
                    "index": index,
                    "link": f"{origin_id} -> {destination_id}",
                    "min_travel_time_sec": minimum,
                    "max_travel_time_sec": maximum,
                },
            )
        if float(maximum) <= float(minimum):
            raise TopologyError(
                "max_travel_time_sec must be strictly greater than min_travel_time_sec",
                {
                    "index": index,
                    "link": f"{origin_id} -> {destination_id}",
                    "min_travel_time_sec": float(minimum),
                    "max_travel_time_sec": float(maximum),
                },
            )
        distance = entry.get("distance_meters")

    try:
        link = CameraLink(
            from_camera_id=origin_id,
            to_camera_id=destination_id,
            min_travel_time_sec=float(minimum),
            max_travel_time_sec=float(maximum),
            distance_meters=float(distance) if distance is not None else None,
            bidirectional=bool(entry.get("bidirectional", True)),
        )
    except ValidationError as exc:
        raise TopologyError(
            "Invalid link entry",
            {"index": index, "link": f"{origin_id} -> {destination_id}", "reason": str(exc)},
        ) from exc

    return link, derived


def _expand(link: CameraLink, derived: bool) -> list[TopologyEdge]:
    """Expand one declared link into its directed edges.

    Args:
        link: The declared link.
        derived: Whether its window was computed.

    Returns:
        One edge for a one-way link, two for a bidirectional one. The reverse
        edge carries the same window: without a directional survey there is no
        basis for asserting the return trip is faster or slower.
    """
    edges = [TopologyEdge(link, derived=derived)]
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
                derived=derived,
                reversed_from_bidirectional=True,
            )
        )
    return edges


def _collect_warnings(topology: Topology, implausible_speed_kph: float) -> list[TopologyWarning]:
    """Report suspicious but legitimate findings.

    Args:
        topology: The constructed graph.
        implausible_speed_kph: Implied speed above which a link is flagged.

    Returns:
        The warnings, in a stable order.
    """
    warnings: list[TopologyWarning] = []

    for camera in topology.list_cameras():
        if not topology.neighbors(camera.camera_id) and not topology.incoming(camera.camera_id):
            warnings.append(
                TopologyWarning(
                    kind="isolated_camera",
                    message=(
                        f"Camera '{camera.camera_id}' has no links; nothing can be "
                        f"reached from it and it can never appear mid-trajectory"
                    ),
                    context={"camera_id": camera.camera_id},
                )
            )

    for edge in topology.list_edges():
        link = edge.link
        if link.distance_meters is None or link.min_travel_time_sec <= 0:
            continue
        implied_kph = (
            (link.distance_meters / link.min_travel_time_sec)
            * SECONDS_PER_HOUR
            / METERS_PER_KILOMETER
        )
        if implied_kph > implausible_speed_kph:
            warnings.append(
                TopologyWarning(
                    kind="implausible_speed",
                    message=(
                        f"Link {link.from_camera_id} -> {link.to_camera_id} implies "
                        f"{implied_kph:.0f} km/h at its minimum travel time, above the "
                        f"{implausible_speed_kph:.0f} km/h sanity limit"
                    ),
                    context={
                        "link": f"{link.from_camera_id} -> {link.to_camera_id}",
                        "implied_kph": round(implied_kph, 1),
                        "limit_kph": implausible_speed_kph,
                    },
                )
            )

    return warnings


def load_topology_result(
    path: Path | str, *, implausible_speed_kph: float | None = None
) -> TopologyLoadResult:
    """Load a topology file, returning the graph and any warnings.

    Args:
        path: Path to the YAML file.
        implausible_speed_kph: Sanity limit for implied speeds. Defaults to the
            configured value.

    Returns:
        The loaded topology and its warnings.

    Raises:
        TopologyError: On any fatal validation failure.
    """
    from multicam_tracker.config import get_settings

    resolved_limit = (
        implausible_speed_kph
        if implausible_speed_kph is not None
        else get_settings().topology.implausible_speed_kph
    )

    document = _read_yaml(Path(path))
    model = _speed_model_from(document.get("defaults"))

    camera_entries = document.get("cameras") or []
    if not isinstance(camera_entries, list) or not camera_entries:
        raise TopologyError("Topology file must declare at least one camera", {"path": str(path)})

    cameras: dict[str, Camera] = {}
    for index, entry in enumerate(camera_entries):
        camera = _build_camera(entry, index)
        if camera.camera_id in cameras:
            raise TopologyError(
                "Duplicate camera_id in topology",
                {"camera_id": camera.camera_id, "index": index},
            )
        cameras[camera.camera_id] = camera

    link_entries = document.get("links") or []
    if not isinstance(link_entries, list):
        raise TopologyError(
            "The 'links' section must be a list",
            {"path": str(path), "parsed_type": type(link_entries).__name__},
        )

    edges: list[TopologyEdge] = []
    for index, entry in enumerate(link_entries):
        link, derived = _build_link(entry, index, cameras, model)
        edges.extend(_expand(link, derived))

    topology = Topology(cameras.values(), edges)
    warnings = _collect_warnings(topology, resolved_limit)

    for warning in warnings:
        logger.warning("topology_warning", kind=warning.kind, detail=warning.message)

    logger.info(
        "topology_loaded",
        path=str(path),
        cameras=len(topology),
        edges=len(topology.list_edges()),
        warnings=len(warnings),
    )
    return TopologyLoadResult(topology=topology, warnings=warnings)


def load_topology(path: Path | str | None = None) -> Topology:
    """Load a topology file.

    Args:
        path: Path to the YAML file. Defaults to the configured topology file.

    Returns:
        The validated graph. Warnings are logged; use
        :func:`load_topology_result` to inspect them programmatically.

    Raises:
        TopologyError: On any fatal validation failure.
    """
    from multicam_tracker.config import get_settings

    resolved = Path(path) if path is not None else get_settings().topology.topology_file
    return load_topology_result(resolved).topology


def _speed_model_from(defaults: Any) -> SpeedModel:
    """Build the speed model from the file's ``defaults`` block.

    Args:
        defaults: The raw block, or ``None`` when absent.

    Returns:
        The speed model, using built-in defaults for anything unspecified.

    Raises:
        TopologyError: If the block is not a mapping or contains unknown keys.
            Unknown keys are rejected because a misspelled ``winding_factor``
            would silently leave every derived window computed from the default.
    """
    if defaults is None:
        return SpeedModel()
    if not isinstance(defaults, dict):
        raise TopologyError(
            "The 'defaults' block must be a mapping",
            {"parsed_type": type(defaults).__name__},
        )

    known = {"min_speed_kph", "max_speed_kph", "winding_factor", "additive_slack_sec"}
    unknown = set(defaults) - known
    if unknown:
        raise TopologyError(
            "Unknown keys in the 'defaults' block",
            {"unknown_keys": sorted(unknown), "known_keys": sorted(known)},
        )

    return SpeedModel(**{key: float(value) for key, value in defaults.items()})
