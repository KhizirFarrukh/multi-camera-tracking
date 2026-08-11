"""multicam-tracker: multi-camera object tracking and path reconstruction.

Given a target identifier (primarily a vehicle license plate, with visual
re-identification as fallback), the system ingests video from multiple cameras,
detects vehicles, records ``Sighting`` rows, and reconstructs the target's
movement path across cameras using synchronized timestamps and a camera
topology graph.

The public surface of this package is deliberately small at this stage: the
foundation modules (:mod:`~multicam_tracker.config`,
:mod:`~multicam_tracker.exceptions`, :mod:`~multicam_tracker.logging_config`,
:mod:`~multicam_tracker.clock`) are the only things that exist until stage 02
introduces the domain models.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
