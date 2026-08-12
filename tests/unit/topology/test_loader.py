"""Unit tests for :mod:`multicam_tracker.topology.loader`.

Split along the line the module itself draws: things that make the graph
meaningless must raise, things that are merely suspicious must load and warn.
Getting that boundary wrong in either direction is expensive -- a topology that
refuses to load blocks the whole system, and one that swallows a broken link
produces confident wrong routes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multicam_tracker.exceptions import TopologyError
from multicam_tracker.models import Camera
from multicam_tracker.topology import SpeedModel, derive_travel_window, load_topology
from multicam_tracker.topology.loader import load_topology_result

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REPO_CONFIG = Path(__file__).resolve().parents[3] / "config" / "topology.yaml"


def _camera(camera_id: str, lat: float = 0.0, lon: float = 0.0) -> Camera:
    """Build a camera at a given position.

    Args:
        camera_id: The identifier.
        lat: Latitude.
        lon: Longitude.

    Returns:
        The camera.
    """
    return Camera(camera_id=camera_id, name=camera_id, lat=lat, lon=lon)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_load_topology__valid_fixture__loads_expected_counts() -> None:
    """Four cameras, and three directed edges after bidirectional expansion."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    assert len(topology) == 4
    assert len(topology.list_edges()) == 3


def test_load_topology__bidirectional_link__expands_into_two_directed_edges() -> None:
    """Downstream code never has to check the reverse direction itself."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    assert topology.has_link("cam_a", "cam_b") is True
    assert topology.has_link("cam_b", "cam_a") is True


def test_load_topology__one_way_link__expands_into_exactly_one_edge() -> None:
    """A one-way street must not become a two-way route by accident."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    assert topology.has_link("cam_b", "cam_c") is True
    assert topology.has_link("cam_c", "cam_b") is False


def test_load_topology__reverse_edge__carries_the_same_window() -> None:
    """Without a directional survey there is no basis for an asymmetric window."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    forward = topology.get_link("cam_a", "cam_b")
    backward = topology.get_link("cam_b", "cam_a")

    assert forward is not None
    assert backward is not None
    assert backward.min_travel_time_sec == forward.min_travel_time_sec
    assert backward.max_travel_time_sec == forward.max_travel_time_sec


def test_load_topology__shipped_example_config__loads_cleanly() -> None:
    """The documented example must actually be a valid topology."""
    result = load_topology_result(REPO_CONFIG)

    assert len(result.topology) == 6
    assert result.topology.has_link("cam_04", "cam_05") is True
    assert result.topology.has_link("cam_05", "cam_04") is False, "one-way avenue"


# ---------------------------------------------------------------------------
# Fatal validation failures
# ---------------------------------------------------------------------------


def test_load_topology__duplicate_camera_id__raises_naming_the_id() -> None:
    """Two rows for one camera would make every lookup ambiguous."""
    with pytest.raises(TopologyError) as excinfo:
        load_topology(FIXTURES / "topology_invalid_duplicate_camera.yaml")

    assert excinfo.value.context["camera_id"] == "cam_a"


def test_load_topology__unknown_camera_in_link__names_the_link_and_the_camera() -> None:
    """Both are needed: which link is broken, and what it is missing."""
    with pytest.raises(TopologyError) as excinfo:
        load_topology(FIXTURES / "topology_invalid_unknown_camera.yaml")

    assert excinfo.value.context["missing_camera_id"] == "cam_ghost"
    assert "cam_a -> cam_ghost" in excinfo.value.context["link"]


def test_load_topology__self_link__is_rejected() -> None:
    """A hop is a transition between cameras; a self-link would loop forever."""
    with pytest.raises(TopologyError, match="Self-link"):
        load_topology(FIXTURES / "topology_invalid_self_link.yaml")


def test_load_topology__duplicate_link_for_one_pair__is_rejected() -> None:
    """Two windows for one edge leave no way to choose between them."""
    with pytest.raises(TopologyError, match="Duplicate link"):
        load_topology(FIXTURES / "topology_invalid_duplicate_link.yaml")


def test_load_topology__inverted_window__names_both_values() -> None:
    """A transposed pair is the usual cause, and is hard to spot without both."""
    with pytest.raises(TopologyError) as excinfo:
        load_topology(FIXTURES / "topology_invalid_window.yaml")

    assert excinfo.value.context["min_travel_time_sec"] == 300.0
    assert excinfo.value.context["max_travel_time_sec"] == 45.0


def test_load_topology__half_specified_window__is_rejected() -> None:
    """Completing a half-written window would mean inventing a constraint."""
    with pytest.raises(TopologyError, match="both travel times or neither"):
        load_topology(FIXTURES / "topology_invalid_half_window.yaml")


def test_load_topology__empty_file__raises() -> None:
    """An empty topology is almost certainly a truncated or unwritten file."""
    with pytest.raises(TopologyError, match="empty"):
        load_topology(FIXTURES / "topology_invalid_empty.yaml")


def test_load_topology__malformed_yaml__raises_topology_error_not_a_parser_error() -> None:
    """Callers catch one hierarchy; a raw yaml.YAMLError would escape it."""
    with pytest.raises(TopologyError, match="could not be parsed"):
        load_topology(FIXTURES / "topology_invalid_malformed.yaml")


def test_load_topology__missing_file__raises() -> None:
    """A mistyped path must not look like an empty network."""
    with pytest.raises(TopologyError, match="not found"):
        load_topology(FIXTURES / "no_such_topology.yaml")


def test_load_topology__no_cameras__raises(tmp_path: Path) -> None:
    """Boundary: a file with links but no cameras cannot describe a graph."""
    path = tmp_path / "topology.yaml"
    path.write_text("cameras: []\nlinks: []\n", encoding="utf-8")

    with pytest.raises(TopologyError, match="at least one camera"):
        load_topology(path)


def test_load_topology__unknown_key_in_defaults__is_rejected(tmp_path: Path) -> None:
    """A misspelled factor would silently leave every derived window at default."""
    path = tmp_path / "topology.yaml"
    path.write_text(
        "defaults:\n  windng_factor: 1.5\ncameras:\n"
        "  - camera_id: cam_a\n    name: A\n    lat: 0.0\n    lon: 0.0\nlinks: []\n",
        encoding="utf-8",
    )

    with pytest.raises(TopologyError, match="Unknown keys"):
        load_topology(path)


# ---------------------------------------------------------------------------
# Warnings: suspicious but legitimate
# ---------------------------------------------------------------------------


def test_load_topology__isolated_camera__warns_but_loads() -> None:
    """A newly installed, not-yet-surveyed camera must not block startup."""
    result = load_topology_result(FIXTURES / "topology_valid.yaml")

    isolated = [w for w in result.warnings if w.kind == "isolated_camera"]
    assert [w.context["camera_id"] for w in isolated] == ["cam_iso"]
    assert "cam_iso" in result.topology


def test_load_topology__implausible_implied_speed__warns_but_loads(tmp_path: Path) -> None:
    """A 700 km/h link is probably a typo, but it might be a motorway."""
    path = tmp_path / "topology.yaml"
    path.write_text(
        "cameras:\n"
        "  - camera_id: cam_a\n    name: A\n    lat: 0.0\n    lon: 0.0\n"
        "  - camera_id: cam_b\n    name: B\n    lat: 0.0\n    lon: 0.1\n"
        "links:\n"
        "  - from_camera_id: cam_a\n    to_camera_id: cam_b\n"
        "    min_travel_time_sec: 10.0\n    max_travel_time_sec: 60.0\n"
        "    distance_meters: 20000.0\n    bidirectional: false\n",
        encoding="utf-8",
    )

    result = load_topology_result(path, implausible_speed_kph=200.0)

    speeds = [w for w in result.warnings if w.kind == "implausible_speed"]
    assert len(speeds) == 1
    assert speeds[0].context["implied_kph"] > 200.0
    assert result.topology.has_link("cam_a", "cam_b")


def test_load_topology__plausible_speed__does_not_warn() -> None:
    """The sanity check must not fire on ordinary links."""
    result = load_topology_result(FIXTURES / "topology_valid.yaml")

    assert [w for w in result.warnings if w.kind == "implausible_speed"] == []


# ---------------------------------------------------------------------------
# Travel-time derivation
# ---------------------------------------------------------------------------


def test_derive_travel_window__uses_the_configured_speeds() -> None:
    """min comes from the max speed and max from the min speed, plus slack."""
    model = SpeedModel(
        min_speed_kph=10.0, max_speed_kph=100.0, winding_factor=1.0, additive_slack_sec=0.0
    )
    origin, destination = _camera("cam_a"), _camera("cam_b", lon=0.01)

    minimum, maximum, straight_line = derive_travel_window(origin, destination, model)

    assert minimum == pytest.approx(straight_line / (100_000 / 3600), rel=1e-9)
    assert maximum == pytest.approx(straight_line / (10_000 / 3600), rel=1e-9)


def test_derive_travel_window__applies_the_winding_factor() -> None:
    """Road distance exceeds straight-line distance by exactly the factor."""
    straight = SpeedModel(winding_factor=1.0, additive_slack_sec=0.0)
    winding = SpeedModel(winding_factor=1.3, additive_slack_sec=0.0)
    origin, destination = _camera("cam_a"), _camera("cam_b", lon=0.01)

    straight_min, _, _ = derive_travel_window(origin, destination, straight)
    winding_min, _, _ = derive_travel_window(origin, destination, winding)

    assert winding_min == pytest.approx(straight_min * 1.3, rel=1e-9)


def test_derive_travel_window__includes_the_additive_slack_in_the_upper_bound() -> None:
    """Slack covers signals and stops, which do not scale with distance."""
    without = SpeedModel(additive_slack_sec=0.0)
    with_slack = SpeedModel(additive_slack_sec=90.0)
    origin, destination = _camera("cam_a"), _camera("cam_b", lon=0.01)

    _, bare_max, _ = derive_travel_window(origin, destination, without)
    _, padded_max, _ = derive_travel_window(origin, destination, with_slack)

    assert padded_max == pytest.approx(bare_max + 90.0, rel=1e-9)


def test_derive_travel_window__reports_the_straight_line_distance_not_the_winded_one() -> None:
    """One is a measured fact, the other an assumption; the record keeps the fact."""
    model = SpeedModel(winding_factor=2.0)
    origin, destination = _camera("cam_a"), _camera("cam_b", lon=0.01)

    _, _, reported = derive_travel_window(origin, destination, model)

    # 0.01 degrees of longitude at the equator is ~1112 m. The winding factor of
    # 2.0 must not appear in the reported distance.
    assert reported == pytest.approx(1112.0, abs=5.0)


def test_derive_travel_window__identical_coordinates__yields_a_finite_non_zero_window() -> None:
    """Boundary: co-located cameras must not produce a zero-width window."""
    model = SpeedModel(additive_slack_sec=60.0)
    origin, destination = _camera("cam_a"), _camera("cam_b")

    minimum, maximum, distance = derive_travel_window(origin, destination, model)

    assert distance == 0.0
    assert minimum == 0.0
    assert maximum == pytest.approx(60.0)


def test_derive_travel_window__zero_slack_on_identical_coordinates__is_rejected() -> None:
    """The degenerate case is reported rather than producing an unusable window."""
    model = SpeedModel(additive_slack_sec=0.0)

    with pytest.raises(TopologyError, match="degenerate"):
        derive_travel_window(_camera("cam_a"), _camera("cam_b"), model)


def test_load_topology__explicit_times__are_not_overwritten_by_derivation(
    tmp_path: Path,
) -> None:
    """An authored window is a measurement and outranks the model."""
    path = tmp_path / "topology.yaml"
    path.write_text(
        "defaults:\n  max_speed_kph: 200.0\n"
        "cameras:\n"
        "  - camera_id: cam_a\n    name: A\n    lat: 0.0\n    lon: 0.0\n"
        "  - camera_id: cam_b\n    name: B\n    lat: 0.0\n    lon: 0.5\n"
        "links:\n"
        "  - from_camera_id: cam_a\n    to_camera_id: cam_b\n"
        "    min_travel_time_sec: 111.0\n    max_travel_time_sec: 222.0\n"
        "    bidirectional: false\n",
        encoding="utf-8",
    )

    topology = load_topology(path)
    link = topology.get_link("cam_a", "cam_b")

    assert link is not None
    assert link.min_travel_time_sec == 111.0
    assert link.max_travel_time_sec == 222.0
    assert topology.is_derived("cam_a", "cam_b") is False


def test_load_topology__omitted_times__are_derived_and_flagged(tmp_path: Path) -> None:
    """Provenance is auditable: every constraint says where it came from."""
    path = tmp_path / "topology.yaml"
    path.write_text(
        "cameras:\n"
        "  - camera_id: cam_a\n    name: A\n    lat: 0.0\n    lon: 0.0\n"
        "  - camera_id: cam_b\n    name: B\n    lat: 0.0\n    lon: 0.01\n"
        "links:\n"
        "  - from_camera_id: cam_a\n    to_camera_id: cam_b\n    bidirectional: false\n",
        encoding="utf-8",
    )

    topology = load_topology(path)
    link = topology.get_link("cam_a", "cam_b")

    assert link is not None
    assert link.min_travel_time_sec > 0
    assert link.distance_meters is not None
    assert topology.is_derived("cam_a", "cam_b") is True


def test_load_topology__derived_bidirectional_link__flags_both_directions(
    tmp_path: Path,
) -> None:
    """The synthesized reverse edge inherits the provenance of its source."""
    path = tmp_path / "topology.yaml"
    path.write_text(
        "cameras:\n"
        "  - camera_id: cam_a\n    name: A\n    lat: 0.0\n    lon: 0.0\n"
        "  - camera_id: cam_b\n    name: B\n    lat: 0.0\n    lon: 0.01\n"
        "links:\n"
        "  - from_camera_id: cam_a\n    to_camera_id: cam_b\n    bidirectional: true\n",
        encoding="utf-8",
    )

    topology = load_topology(path)

    assert topology.is_derived("cam_a", "cam_b") is True
    assert topology.is_derived("cam_b", "cam_a") is True


def test_is_derived__absent_edge__is_false() -> None:
    """An edge that does not exist was not derived; it simply is not there."""
    topology = load_topology(FIXTURES / "topology_valid.yaml")

    assert topology.is_derived("cam_c", "cam_b") is False


# ---------------------------------------------------------------------------
# Speed model validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_speed_kph": 0.0},
        {"max_speed_kph": -5.0},
        {"winding_factor": 0.0},
        {"additive_slack_sec": -1.0},
    ],
    ids=["zero-min-speed", "negative-max-speed", "zero-winding", "negative-slack"],
)
def test_speed_model__invalid_factor__is_rejected(kwargs: dict[str, float]) -> None:
    """A non-positive factor produces infinite or negative travel times."""
    with pytest.raises(TopologyError):
        SpeedModel(**kwargs)


def test_speed_model__inverted_speed_range__is_rejected() -> None:
    """A max below the min would invert the derived window."""
    with pytest.raises(TopologyError, match="max_speed_kph"):
        SpeedModel(min_speed_kph=90.0, max_speed_kph=10.0)
