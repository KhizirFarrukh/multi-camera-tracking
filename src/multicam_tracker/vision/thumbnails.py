"""Cutting the vehicle out of the frame and writing it somewhere findable.

A thumbnail is the only part of a sighting a human can check. Everything else --
the plate string, the embedding, the confidence -- is a claim the system makes
about an image nobody kept. So this is what a reviewer looks at when asked
whether the system got it right, and it is what stage 12 reads the plate from.

Three decisions worth stating:

**Crops carry padding.** A crop cut exactly to the detector's box loses whatever
the box was a pixel tight on, and the plate sits at the very bottom edge of a
vehicle box more often than anywhere else. A tenth of the box size on each side
costs almost nothing and recovers those.

**Boxes are clamped, never rejected.** A vehicle entering frame has a box that
runs past the edge -- that is not an error, it is what entering frame looks
like. The crop is clipped to what exists. A box that clips to nothing still
produces a readable image rather than raising, because the failure a reviewer
needs to see is "there is nothing here", and an exception three layers down does
not say that.

**Paths cannot collide.** Two vehicles crossing one camera in the same second is
ordinary, not exotic, and a path scheme keyed on camera and timestamp alone
silently overwrites one with the other -- leaving two sighting rows pointing at
one image, of which one is wrong. The sighting id is therefore part of the
filename, which makes collisions impossible rather than unlikely.

Thumbnails are subject to the retention TTL from stage 03
(:class:`~multicam_tracker.config.RetentionSettings`), and the purge job in
stage 19 is what enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from multicam_tracker.config import Settings
from multicam_tracker.exceptions import VisionError
from multicam_tracker.logging_config import get_logger

__all__ = [
    "ThumbnailWriter",
    "crop_with_padding",
    "encode_jpeg",
    "resize_image",
]

logger = get_logger(__name__)

_MIN_CROP_SIDE = 1
"""A crop is never smaller than one pixel on a side; see :func:`crop_with_padding`."""


def _require_cv2() -> Any:
    """Return the OpenCV module, or explain what is missing.

    Imported lazily rather than at module import so that the tracker, the
    ranking, and everything that does not write an image stay importable in the
    core install. Stage 10 made the same choice for the decoders and for the
    same reason.

    Returns:
        The ``cv2`` module.

    Raises:
        VisionError: If OpenCV is not installed, naming the extra that provides
            it.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise VisionError(
            "OpenCV is required to encode thumbnails; install the 'vision' extra",
            {"package": "opencv-python", "extra": "vision"},
        ) from exc
    return cv2


def crop_with_padding(
    image: npt.NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
    *,
    padding_fraction: float = 0.0,
) -> npt.NDArray[np.uint8]:
    """Return an owned copy of the box region, padded and clamped to the frame.

    Args:
        image: The source frame.
        bbox: ``(x1, y1, x2, y2)`` in that frame's coordinates.
        padding_fraction: Context to add on each side, as a fraction of the
            box's width and height.

    Returns:
        A writeable copy. A copy rather than a view because the decoder reuses
        its buffer for the next frame, so a view kept by a track would change
        underneath it (see :class:`~multicam_tracker.ingest.frame.Frame`).

        A box that clamps to nothing -- zero area, or entirely outside the frame
        -- yields a one-pixel image and a warning rather than an exception. The
        caller gets something valid to write, and the event is counted where
        somebody will see it.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    pad_x = round((x2 - x1) * padding_fraction)
    pad_y = round((y2 - y1) * padding_fraction)

    left = max(0, min(x1 - pad_x, width))
    top = max(0, min(y1 - pad_y, height))
    right = max(0, min(x2 + pad_x, width))
    bottom = max(0, min(y2 + pad_y, height))

    if right - left < _MIN_CROP_SIDE or bottom - top < _MIN_CROP_SIDE:
        logger.warning(
            "degenerate_crop",
            bbox=list(bbox),
            frame_size=[width, height],
            detail="box clamped to nothing; emitting a one-pixel image",
        )
        left = min(left, max(0, width - _MIN_CROP_SIDE))
        top = min(top, max(0, height - _MIN_CROP_SIDE))
        right = left + _MIN_CROP_SIDE
        bottom = top + _MIN_CROP_SIDE

    return np.array(image[top:bottom, left:right], dtype=np.uint8, copy=True)


def resize_image(image: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
    """Return the image resized to exact dimensions.

    Args:
        image: The image to resize.
        width: Target width.
        height: Target height.

    Returns:
        The resized image. Aspect ratio is not preserved: a thumbnail store
        whose images are all the same size is what makes a review grid render
        and a batch of crops stack into a tensor, and the box the crop came from
        is recorded separately for anyone who needs the true proportions.

    Raises:
        VisionError: If OpenCV is not installed, or the target size is not
            positive.
    """
    if width <= 0 or height <= 0:
        raise VisionError(
            "Thumbnail dimensions must be positive",
            {"width": width, "height": height},
        )
    cv2 = _require_cv2()
    resized: npt.NDArray[np.uint8] = cv2.resize(
        image, (width, height), interpolation=cv2.INTER_AREA
    )
    return resized


def encode_jpeg(image: npt.NDArray[np.uint8], quality: int) -> bytes:
    """Return the image encoded as JPEG bytes.

    Args:
        image: The image to encode.
        quality: JPEG quality, 1-100.

    Returns:
        The encoded bytes.

    Raises:
        VisionError: If OpenCV is not installed, or the encoder refuses the
            image.
    """
    cv2 = _require_cv2()
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise VisionError(
            "JPEG encoding failed",
            {"shape": list(image.shape), "quality": quality},
        )
    return bytes(buffer.tobytes())


@dataclass
class ThumbnailWriter:
    """Writes vehicle crops to the configured store.

    Args:
        root: Directory the relative paths are resolved against. Sighting rows
            store the relative path, so the store can be moved or mounted
            elsewhere without rewriting the database.
        width: Standard thumbnail width.
        height: Standard thumbnail height.
        padding_fraction: Context kept around each box.
        jpeg_quality: Encoder quality.
    """

    root: Path
    width: int = 224
    height: int = 224
    padding_fraction: float = 0.1
    jpeg_quality: int = 85

    @classmethod
    def from_settings(cls, settings: Settings) -> ThumbnailWriter:
        """Build the writer from configuration.

        Args:
            settings: Loaded settings.

        Returns:
            A writer pointed at the configured thumbnail directory.
        """
        detection = settings.detection
        return cls(
            root=settings.storage.thumbnail_dir,
            width=detection.thumbnail_width,
            height=detection.thumbnail_height,
            padding_fraction=detection.crop_padding_fraction,
            jpeg_quality=detection.thumbnail_jpeg_quality,
        )

    def relative_path(self, camera_id: str, timestamp_utc: datetime, sighting_id: str) -> Path:
        """Return the store-relative path for one sighting's thumbnail.

        Laid out by camera and date because that is how the purge job scans it
        and how a human browses it: "everything cam_03 saw on the 14th" is one
        directory rather than a query.

        Args:
            camera_id: Which camera.
            timestamp_utc: The sighting's corrected capture time.
            sighting_id: The sighting's identifier, which is what makes the path
                collision-free.

        Returns:
            A relative path such as
            ``cam_03/2026/08/10/142211_500_3f2a1c9b.jpg``.
        """
        stamp = timestamp_utc.strftime("%H%M%S")
        millis = f"{timestamp_utc.microsecond // 1000:03d}"
        suffix = sighting_id.replace("-", "")[:8]
        return Path(
            camera_id,
            f"{timestamp_utc.year:04d}",
            f"{timestamp_utc.month:02d}",
            f"{timestamp_utc.day:02d}",
            f"{stamp}_{millis}_{suffix}.jpg",
        )

    def write(
        self,
        image: npt.NDArray[np.uint8],
        bbox: tuple[int, int, int, int],
        *,
        camera_id: str,
        timestamp_utc: datetime,
        sighting_id: str,
        already_cropped: bool = False,
    ) -> str:
        """Crop, resize, encode, and write one thumbnail.

        Args:
            image: The source frame, or an already-cropped region.
            bbox: The vehicle box in the source frame. Ignored when
                ``already_cropped`` is set.
            camera_id: Which camera.
            timestamp_utc: The sighting's corrected capture time.
            sighting_id: The sighting's identifier.
            already_cropped: Set when ``image`` is the retained crop from the
                track rather than a whole frame. The track kept the crop for
                exactly this, and cropping a crop again would pad it twice.

        Returns:
            The store-relative path, as a string, for
            :attr:`~multicam_tracker.models.sighting.Sighting.thumbnail_path`.

        Raises:
            VisionError: If OpenCV is unavailable or the image cannot be
                encoded.
        """
        region = (
            image
            if already_cropped
            else crop_with_padding(image, bbox, padding_fraction=self.padding_fraction)
        )
        encoded = encode_jpeg(resize_image(region, self.width, self.height), self.jpeg_quality)

        relative = self.relative_path(camera_id, timestamp_utc, sighting_id)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)

        logger.debug(
            "thumbnail_written",
            camera_id=camera_id,
            sighting_id=sighting_id,
            path=str(relative),
            bytes=len(encoded),
        )
        return relative.as_posix()
