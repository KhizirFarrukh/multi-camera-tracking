# Camera topology

The topology is what turns an unbounded question — *where else did this vehicle
appear?* — into a bounded one: *which cameras could it have reached between
10:02 and 10:07?*

That matters twice over. It sets the **false-positive rate**, because a
too-permissive window admits sightings the vehicle could not physically have
produced. And it sets the **query cost**, because a bounded arrival window turns
a full-table scan into an indexed range scan.

Topology bugs are quiet. They do not crash; they produce plausible-looking
routes that are wrong.

---

## Authoring vs. runtime

| | Format | Role |
|---|---|---|
| `config/topology.yaml` | YAML | **Authoring.** A human edits it and reviews it in a diff. |
| `cameras` / `camera_links` tables | Postgres | **Runtime.** What the system queries. |

`sync_topology_to_db()` pushes one into the other. Edit the file, review, sync.

---

## File format

```yaml
defaults:            # optional — the speed model, see below
  min_speed_kph: 10.0
  max_speed_kph: 90.0
  winding_factor: 1.3
  additive_slack_sec: 60.0

cameras:
  - camera_id: cam_01        # ^[a-z0-9_-]+$, stable once in use
    name: Broadway & Canal St
    lat: 40.7195
    lon: -74.0021
    heading_degrees: 90.0    # optional, [0, 360)
    clock_offset_ms: 0       # optional, applied at ingest (stage 09)
    enabled: true            # optional
    notes: ...               # optional

links:
  - from_camera_id: cam_01
    to_camera_id: cam_02
    min_travel_time_sec: 45.0    # both, or neither
    max_travel_time_sec: 300.0
    distance_meters: 815.0       # optional
    bidirectional: true          # optional, defaults to true
```

**Supply both travel times or neither.** A half-specified window cannot be
completed unambiguously, so it is rejected rather than guessed at.

**`bidirectional: false` means exactly one directed edge.** A one-way street
declared bidirectional invites the system to propose a route no vehicle could
drive. The example config includes one on purpose.

---

## The derivation model

A link that omits both travel times has them derived:

```
road_distance = great_circle_distance × winding_factor
min_travel    = road_distance / max_speed
max_travel    = road_distance / min_speed + additive_slack_sec
```

The pairing is the part worth pausing on: the **fastest** speed sets the
**lower** travel-time bound, and the **slowest** speed sets the **upper** one.

| Factor | Default | What it is for |
|---|---|---|
| `min_speed_kph` | 10.0 | Slowest plausible average. Congested-urban; below this a vehicle is stopped, which the slack covers instead. |
| `max_speed_kph` | 90.0 | Fastest plausible average. Set it **above** the speed limit — it bounds what is possible, not what is legal, and a speeding vehicle is exactly the case worth catching. |
| `winding_factor` | 1.3 | Straight line → road distance. Scales every derived window linearly, so measure it if accuracy matters. |
| `additive_slack_sec` | 60.0 | Signals, queueing, brief stops. These do not scale with distance, so without a fixed allowance short links get absurdly tight windows. |

Two cameras at identical coordinates derive to `(0.0, additive_slack_sec)` —
co-located cameras genuinely have no minimum transit, but a zero-width window
would accept nothing at all.

**Every derived edge is flagged.** `topology.is_derived(from_id, to_id)` says
whether a constraint was measured by a human or inferred by the model. Prefer
measuring a handful of real transits over tuning the factors.

---

## Errors vs. warnings

The loader treats these differently on purpose.

**Errors** make the graph meaningless, and raise `TopologyError`:

- duplicate `camera_id`
- a link naming a camera that does not exist
- a self-link
- two definitions of the same ordered pair
- `max_travel_time_sec <= min_travel_time_sec`
- only one of the two travel times
- empty file, malformed YAML, unknown key in `defaults`

**Warnings** are suspicious but legitimate, and load anyway:

- an **isolated camera** — newly installed, not yet surveyed
- an **implausible implied speed** — a typo, or a motorway

Refusing to start over a possibly-correct oddity is worse than surfacing it.
`load_topology_result()` returns the warnings; `load_topology()` logs them.

---

## Query API

All pure — no database, no filesystem, no clock.

```python
topology.has_link("cam_01", "cam_02")  # -> bool, direction-sensitive
topology.neighbors("cam_01")  # -> list[CameraLink], outgoing
topology.get_link("cam_01", "cam_02")  # -> CameraLink | None

is_transition_plausible(topology, "cam_01", "cam_02", elapsed_sec)
# -> PlausibilityResult(plausible, reason, margin_sec)

plausibility_score(topology, "cam_01", "cam_02", elapsed_sec)
# -> float in [0, 1]

reachable_from(topology, "cam_01", departure, max_horizon_sec)
# -> list[ReachableCamera]  (single hop)

reachable_within(topology, "cam_01", departure, max_horizon_sec, max_hops)
# -> ReachabilityResult    (multi-hop, with a truncation flag)
```

### Conventions that later stages depend on

**Travel windows are inclusive at both ends.** Elapsed time exactly equal to
`min_travel_time_sec` is plausible, and so is exactly `max_travel_time_sec`. An
exclusive bound would make a verdict depend on floating-point equality.

**Negative elapsed time returns `too_fast`, it does not raise.** Out-of-order
timestamps mean bad data, but path reconstruction evaluates thousands of
candidate pairs and one nonsense pair must not abort the search.

**`plausibility_score` never reaches zero for a linked pair.** It decays
exponentially with the margin (`0.5 ** (margin / half_life)`). The contract
requires an implausible hop to be *penalised, not discarded*, and a hard zero
would discard it. An unlinked pair returns a configured floor rather than
raising — an undeclared route is unexplained, not proven impossible.

**Multi-hop windows compound min-with-min and max-with-max.** The earliest
possible arrival is every leg driven at its fastest; the latest is every leg at
its slowest.

**Arrival windows are not clipped to the horizon.** The horizon decides
*whether* a camera is included — its earliest arrival must fall inside — but the
window reports the full plausible range, because a caller narrowing a database
query needs the true upper bound.

**A truncated reachability result is a lower bound.** Everything listed is
genuinely reachable, but something reachable may be missing. Check
`result.truncated` before treating absence as proof of absence.

---

## Same-camera repeats

A camera that sees a vehicle in three consecutive sampled frames observed **one**
transit. A vehicle that circles the block and returns four minutes later made
**two**. `is_same_pass(elapsed_sec)` draws the line at
`topology.min_redetection_gap_sec` (default 60s), strictly: a gap of exactly
that value is a *separate* transit, matching the half-open convention used for
time windows everywhere else.

Every later stage reads the gap from here rather than picking its own.

---

## Syncing

```python
sync_topology_to_db(topology, camera_repo, link_repo)
```

Idempotent — every write is an upsert keyed on the natural identifier. Only the
**declared** direction of a bidirectional link is stored; the reverse is
re-expanded on load, so the database holds what the author wrote rather than a
derived duplicate that would drift if the original were edited.

**Deletion is opt-in.** A camera present in the database but absent from the
file is *retained* by default. Pass `prune_missing=True` with a `sighting_repo`
to remove it — and a camera that still has sightings raises `TopologyError`
listing it, deleting nothing at all. Evidence outlives the hardware that
recorded it, and a partial prune would leave the operator unsure what survived.

---

## Operational settings

The speed model lives in the topology file because it describes one road
network. These live in application settings because they describe how *this
deployment* queries the graph:

| Setting | Default | Effect |
|---|---|---|
| `MCT_TOPOLOGY__TOPOLOGY_FILE` | `config/topology.yaml` | Where to load from |
| `MCT_TOPOLOGY__IMPLAUSIBLE_SPEED_KPH` | 200.0 | Implied speed that triggers a warning |
| `MCT_TOPOLOGY__MIN_REDETECTION_GAP_SEC` | 60.0 | One pass vs. two transits |
| `MCT_TOPOLOGY__PLAUSIBILITY_DECAY_HALF_LIFE_SEC` | 120.0 | How fast the score decays outside the window |
| `MCT_TOPOLOGY__UNLINKED_PLAUSIBILITY_SCORE` | 0.1 | Floor for an undeclared pair |
| `MCT_TOPOLOGY__MAX_VISITED_NODES` | 10000 | Expansion cap for multi-hop search |
