"""Camera and camera-link models.

Together these form the topology graph that stage 04 queries: cameras are the
nodes, links carry the plausible travel-time window between them, and stage 08
uses those windows to reject transitions that could not physically have
happened.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from multicam_tracker.models.base import MCTBaseModel

__all__ = ["CAMERA_ID_PATTERN", "Camera", "CameraLink"]

CAMERA_ID_PATTERN = r"^[a-z0-9_\-]+$"
"""Lowercase alphanumerics, underscores, and hyphens.

Constrained because camera ids appear in file paths, URL segments, log fields,
and database keys. Allowing mixed case would let ``Cam_03`` and ``cam_03``
denote the same physical camera in different places, and every join against
them would silently under-match.
"""


class Camera(MCTBaseModel):
    """A fixed camera at a known location."""

    camera_id: str = Field(
        pattern=CAMERA_ID_PATTERN,
        min_length=1,
        description="Stable human-readable identifier, e.g. 'cam_03'",
    )
    name: str = Field(min_length=1, description="Human label, e.g. 'Main St & 5th Ave'")
    lat: float = Field(ge=-90.0, le=90.0, description="WGS84 latitude")
    lon: float = Field(ge=-180.0, le=180.0, description="WGS84 longitude")
    heading_degrees: float | None = Field(
        default=None,
        ge=0.0,
        lt=360.0,
        description="Direction the camera faces, in [0, 360). 360 is rejected as it aliases 0.",
    )
    clock_offset_ms: int = Field(
        default=0,
        description="Milliseconds added to this camera's raw timestamps (drift, stage 09)",
    )
    enabled: bool = True
    notes: str | None = None


class CameraLink(MCTBaseModel):
    """A directed transition between two cameras with a plausible travel time.

    The window is the pruning mechanism at the heart of path reconstruction. A
    candidate sighting at ``to_camera_id`` is only plausible if the time since
    the sighting at ``from_camera_id`` falls inside
    ``[min_travel_time_sec, max_travel_time_sec]``.
    """

    from_camera_id: str = Field(pattern=CAMERA_ID_PATTERN, min_length=1)
    to_camera_id: str = Field(pattern=CAMERA_ID_PATTERN, min_length=1)
    min_travel_time_sec: float = Field(
        ge=0.0, description="Fastest plausible transit time in seconds"
    )
    max_travel_time_sec: float = Field(
        gt=0.0, description="Slowest plausible transit time before the link is implausible"
    )
    distance_meters: float | None = Field(default=None, ge=0.0)
    bidirectional: bool = True

    @model_validator(mode="after")
    def _window_is_ordered(self) -> CameraLink:
        """Reject a travel-time window that is inverted or degenerate.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``max_travel_time_sec`` is not strictly greater than
                ``min_travel_time_sec``. Both values appear in the message
                because a topology file with the two transposed is the common
                cause and is otherwise tedious to spot.
        """
        if self.max_travel_time_sec <= self.min_travel_time_sec:
            msg = (
                f"max_travel_time_sec ({self.max_travel_time_sec}) must be strictly greater "
                f"than min_travel_time_sec ({self.min_travel_time_sec}) for link "
                f"{self.from_camera_id} -> {self.to_camera_id}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _not_a_self_link(self) -> CameraLink:
        """Reject a link from a camera to itself.

        Returns:
            The validated instance.

        Raises:
            ValueError: If both endpoints name the same camera. A hop is defined
                as a transition *between* cameras, so a self-link would let path
                reconstruction loop on one node forever.
        """
        if self.from_camera_id == self.to_camera_id:
            msg = f"from_camera_id and to_camera_id must differ; both are '{self.from_camera_id}'"
            raise ValueError(msg)
        return self
