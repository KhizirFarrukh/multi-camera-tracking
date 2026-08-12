"""Camera topology: the graph, its constraints, and the queries over it.

This package is what turns an unbounded search ("where else did this vehicle
appear?") into a bounded one ("which cameras could it have reached between
10:02 and 10:07?"). Its correctness sets both the false-positive rate and the
cost of every query stages 06-08 make.

The core query path is pure: :mod:`~multicam_tracker.topology.graph`,
:mod:`~multicam_tracker.topology.plausibility`,
:mod:`~multicam_tracker.topology.reachability`, and
:mod:`~multicam_tracker.topology.coverage` touch no database and no filesystem.
Only :mod:`~multicam_tracker.topology.loader` reads files and only
:mod:`~multicam_tracker.topology.sync` reaches the database.
"""

from __future__ import annotations

from multicam_tracker.topology.coverage import (
    describe_gap,
    find_isolated_cameras,
    graph_diameter_sec,
    shortest_transit_sec,
)
from multicam_tracker.topology.graph import Topology, TopologyEdge
from multicam_tracker.topology.loader import (
    SpeedModel,
    TopologyLoadResult,
    TopologyWarning,
    derive_travel_window,
    load_topology,
    load_topology_result,
)
from multicam_tracker.topology.plausibility import (
    PlausibilityReason,
    PlausibilityResult,
    is_same_pass,
    is_transition_plausible,
    plausibility_score,
)
from multicam_tracker.topology.reachability import (
    ReachabilityResult,
    ReachableCamera,
    reachable_from,
    reachable_within,
)
from multicam_tracker.topology.sync import (
    SyncReport,
    load_topology_from_db,
    sync_topology_to_db,
)

__all__ = [
    "PlausibilityReason",
    "PlausibilityResult",
    "ReachabilityResult",
    "ReachableCamera",
    "SpeedModel",
    "SyncReport",
    "Topology",
    "TopologyEdge",
    "TopologyLoadResult",
    "TopologyWarning",
    "derive_travel_window",
    "describe_gap",
    "find_isolated_cameras",
    "graph_diameter_sec",
    "is_same_pass",
    "is_transition_plausible",
    "load_topology",
    "load_topology_from_db",
    "load_topology_result",
    "plausibility_score",
    "reachable_from",
    "reachable_within",
    "shortest_transit_sec",
    "sync_topology_to_db",
]
