"""The unit every vision stage consumes.

A :class:`Frame` is one decoded image plus everything needed to say *when and
where* it was captured. Everything downstream -- detection, OCR, embedding
extraction -- takes frames and nothing else, which is what lets stages 11-13 be
tested against generated frames with no media files at all.

**Memory ownership.** The array a frame carries is **read-only to consumers**.
A decoder hands out a view into a buffer it will reuse for the next frame, so a
consumer that writes into ``frame.image`` is corrupting the next frame as well
as its own, and a consumer that keeps a reference is holding a buffer that
changes underneath it. Both failures produce images that are subtly wrong rather
than obviously broken.

Two things enforce that rather than merely asking for it:

* the array is marked non-writeable, so an accidental write raises instead of
  silently corrupting;
* :meth:`Frame.owned_copy` is the one supported way to get an array a consumer
  may keep and modify.

The metadata is deliberately small. A frame is created thousands of times a
minute, and anything copied per frame is paid for thousands of times a minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import IngestError

__all__ = ["Frame", "as_readonly"]

_COLOUR_CHANNELS = 3


def as_readonly(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a non-writeable view of an image.

    Args:
        image: The decoded image.

    Returns:
        A view whose ``WRITEABLE`` flag is off. A view rather than a copy: the
        point is to make accidental writes fail, not to pay for a copy of every
        frame that passes through.
    """
    view = image.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class Frame:
    """One decoded image with its capture time and provenance."""

    image: npt.NDArray[np.uint8]
    """BGR, ``(height, width, 3)``, **read-only**. Call :meth:`owned_copy` for an
    array you may keep or modify."""

    frame_index: int
    """Position within the source, from zero. Never renumbered by sampling: a
    sampler that yielded 0, 1, 2 for frames 0, 30, 60 would make every later
    reference to "frame 30" mean something else."""

    raw_timestamp: datetime
    """Capture time as the source reported it, before any clock correction."""

    timestamp_utc: datetime
    """Capture time after the camera's clock offset is applied (stage 09)."""

    source_id: str
    camera_id: str
    is_keyframe: bool = False

    def __post_init__(self) -> None:
        """Validate the image and enforce the read-only convention.

        Raises:
            IngestError: If the image is not a three-channel 8-bit array. A
                grayscale or float array reaching a detector produces a
                confident wrong answer rather than an error, so it is rejected
                at the boundary.
        """
        image = self.image
        if image.ndim != _COLOUR_CHANNELS or image.shape[2] != _COLOUR_CHANNELS:
            raise IngestError(
                "Frame image must be a three-channel BGR array",
                {"shape": list(image.shape), "source_id": self.source_id},
            )
        if image.dtype != np.uint8:
            raise IngestError(
                "Frame image must be 8-bit; a float or 16-bit array reaching a "
                "detector yields a confident wrong answer rather than an error",
                {"dtype": str(image.dtype), "source_id": self.source_id},
            )
        if image.flags.writeable:
            # frozen dataclass: assign through __dict__ as the generated
            # __init__ does.
            object.__setattr__(self, "image", as_readonly(image))

    @property
    def height(self) -> int:
        """Return the image height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Return the image width in pixels."""
        return int(self.image.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(width, height)``, the order the config and the UI use."""
        return (self.width, self.height)

    def owned_copy(self) -> npt.NDArray[np.uint8]:
        """Return a writeable copy of the image.

        The one supported way to get an array a consumer may keep or modify.
        Everything else is a view into a buffer the decoder will reuse.

        Returns:
            A fresh writeable array.
        """
        return np.array(self.image, dtype=np.uint8, copy=True)

    def replacing_image(self, image: npt.NDArray[np.uint8]) -> Frame:
        """Return the same frame carrying a different image.

        Used by preprocessing, which changes the pixels and must not change
        anything else -- least of all the timestamps, which are the frame's
        whole claim about when it was captured.

        Args:
            image: The processed image.

        Returns:
            A new frame with identical metadata.
        """
        return Frame(
            image=image,
            frame_index=self.frame_index,
            raw_timestamp=self.raw_timestamp,
            timestamp_utc=self.timestamp_utc,
            source_id=self.source_id,
            camera_id=self.camera_id,
            is_keyframe=self.is_keyframe,
        )

    def __repr__(self) -> str:
        """Return a representation that does not print the whole image."""
        return (
            f"Frame(camera_id={self.camera_id!r}, frame_index={self.frame_index}, "
            f"timestamp_utc={self.timestamp_utc.isoformat()}, "
            f"size={self.width}x{self.height}, keyframe={self.is_keyframe})"
        )
