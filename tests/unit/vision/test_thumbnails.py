"""Cropping, encoding, and the path scheme that cannot collide.

The collision case is the one that would otherwise reach production: two
vehicles crossing one camera in the same second is ordinary, and a scheme keyed
on camera and timestamp alone silently overwrites one with the other, leaving
two sighting rows pointing at one image of which one is wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from multicam_tracker.config import Settings
from multicam_tracker.exceptions import VisionError
from multicam_tracker.vision import ThumbnailWriter, crop_with_padding, encode_jpeg, resize_image

pytestmark = pytest.mark.unit

MOMENT = datetime(2026, 8, 10, 14, 22, 11, 500_000, tzinfo=UTC)


def _has_opencv() -> bool:
    """Return whether OpenCV is importable.

    Returns:
        ``True`` when it imports.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


requires_opencv = pytest.mark.skipif(not _has_opencv(), reason="OpenCV is not installed")


def gradient_frame(width: int = 160, height: int = 120) -> np.ndarray:
    """Return a frame whose pixel values encode their own position.

    That is what lets a crop assertion check it took the right region rather
    than merely a region of the right size.

    Args:
        width: Frame width.
        height: Frame height.

    Returns:
        A BGR image.
    """
    rows = np.arange(height, dtype=np.uint8)[:, None]
    columns = np.arange(width, dtype=np.uint8)[None, :]
    return np.stack(
        [
            np.broadcast_to(rows, (height, width)),
            np.broadcast_to(columns, (height, width)),
            np.zeros((height, width), dtype=np.uint8),
        ],
        axis=2,
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def test_crop_with_padding__no_padding__matches_the_box_exactly() -> None:
    """The anchor case: a crop is the box, and its content proves which region."""
    frame = gradient_frame()

    crop = crop_with_padding(frame, (40, 30, 70, 50))

    assert crop.shape == (20, 30, 3)
    assert crop[0, 0, 0] == 30
    assert crop[0, 0, 1] == 40


def test_crop_with_padding__with_padding__expands_by_the_configured_fraction() -> None:
    """A crop cut exactly to the box loses whatever the box was a pixel tight on."""
    frame = gradient_frame()

    crop = crop_with_padding(frame, (40, 30, 80, 60), padding_fraction=0.1)

    # 40x30 box, padded by 4 and 3 on each side.
    assert crop.shape == (30 + 6, 40 + 8, 3)


def test_crop_with_padding__box_past_the_frame_edge__is_clamped_and_still_valid() -> None:
    """A vehicle entering frame has a box past the edge; that is not an error."""
    frame = gradient_frame()

    crop = crop_with_padding(frame, (140, 100, 200, 160))

    assert crop.shape == (20, 20, 3)
    assert crop.size > 0


def test_crop_with_padding__zero_area_box__yields_a_one_pixel_image_without_raising() -> None:
    """The reviewer needs to see "there is nothing here", not an exception three layers down."""
    crop = crop_with_padding(gradient_frame(), (50, 50, 50, 50))

    assert crop.shape == (1, 1, 3)


def test_crop_with_padding__box_entirely_outside_the_frame__does_not_raise() -> None:
    """Boundary: nothing survives the clamp, and the caller still gets a valid image."""
    crop = crop_with_padding(gradient_frame(), (500, 500, 600, 600))

    assert crop.shape == (1, 1, 3)


def test_crop_with_padding__returns_an_owned_copy_not_a_view() -> None:
    """A decoder reuses its buffer; a view kept by a track changes underneath it."""
    frame = gradient_frame()

    crop = crop_with_padding(frame, (10, 10, 30, 30))
    crop[0, 0] = 255

    assert frame[10, 10, 0] != 255


# ---------------------------------------------------------------------------
# Resizing and encoding
# ---------------------------------------------------------------------------


@requires_opencv
def test_resize_image__produces_exactly_the_requested_dimensions() -> None:
    """A store whose images are all one size is what makes a review grid render."""
    resized = resize_image(gradient_frame(), 64, 48)

    assert resized.shape == (48, 64, 3)


def test_resize_image__non_positive_dimensions__are_rejected() -> None:
    """A zero-sided thumbnail is a configuration error, caught where it is named."""
    with pytest.raises(VisionError, match="must be positive"):
        resize_image(gradient_frame(), 0, 48)


@requires_opencv
def test_encode_jpeg__produces_bytes_a_decoder_reads_back() -> None:
    """A file nobody can open is not a thumbnail."""
    import cv2

    encoded = encode_jpeg(gradient_frame(32, 32), quality=85)
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert encoded[:2] == b"\xff\xd8"
    assert decoded is not None
    assert decoded.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_relative_path__lays_the_store_out_by_camera_and_date(tmp_path: Path) -> None:
    """How the purge job scans it and how a human browses it."""
    writer = ThumbnailWriter(root=tmp_path)

    path = writer.relative_path("cam_03", MOMENT, "3f2a1c9b-0000-0000-0000-000000000000")

    assert path.parts[:4] == ("cam_03", "2026", "08", "10")
    assert path.name.startswith("142211_500_")
    assert path.suffix == ".jpg"


def test_relative_path__two_sightings_in_the_same_second__do_not_collide(tmp_path: Path) -> None:
    """Two vehicles crossing one camera in the same second is ordinary, not exotic."""
    writer = ThumbnailWriter(root=tmp_path)

    first = writer.relative_path("cam_03", MOMENT, "aaaaaaaa-0000-0000-0000-000000000000")
    second = writer.relative_path("cam_03", MOMENT, "bbbbbbbb-0000-0000-0000-000000000000")

    assert first != second
    assert first.parent == second.parent


def test_relative_path__is_stable_for_the_same_sighting(tmp_path: Path) -> None:
    """Idempotency: rewriting a thumbnail must land on the same path, not a second file."""
    writer = ThumbnailWriter(root=tmp_path)
    arguments = ("cam_03", MOMENT, "aaaaaaaa-0000-0000-0000-000000000000")

    assert writer.relative_path(*arguments) == writer.relative_path(*arguments)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@requires_opencv
def test_write__creates_a_readable_jpeg_at_the_configured_size(tmp_path: Path) -> None:
    """The end-to-end obligation: a file on disk that opens and is the right shape."""
    import cv2

    writer = ThumbnailWriter(root=tmp_path, width=64, height=64, jpeg_quality=80)

    relative = writer.write(
        gradient_frame(),
        (40, 30, 80, 60),
        camera_id="cam_03",
        timestamp_utc=MOMENT,
        sighting_id="aaaaaaaa-0000-0000-0000-000000000000",
    )

    written = tmp_path / relative
    assert written.is_file()

    decoded = cv2.imread(str(written))
    assert decoded is not None
    assert decoded.shape == (64, 64, 3)


@requires_opencv
def test_write__already_cropped__does_not_pad_a_second_time(tmp_path: Path) -> None:
    """The track kept the crop for exactly this; cropping it again pads it twice."""
    writer = ThumbnailWriter(root=tmp_path, width=32, height=32, padding_fraction=0.5)
    crop = gradient_frame(20, 20)

    relative = writer.write(
        crop,
        (0, 0, 20, 20),
        camera_id="cam_03",
        timestamp_utc=MOMENT,
        sighting_id="aaaaaaaa-0000-0000-0000-000000000000",
        already_cropped=True,
    )

    assert (tmp_path / relative).is_file()


@requires_opencv
def test_write__creates_missing_directories(tmp_path: Path) -> None:
    """A fresh deployment has no store yet, and that must not be an error."""
    writer = ThumbnailWriter(root=tmp_path / "nowhere" / "yet", width=16, height=16)

    relative = writer.write(
        gradient_frame(),
        (10, 10, 40, 40),
        camera_id="cam_01",
        timestamp_utc=MOMENT,
        sighting_id="aaaaaaaa-0000-0000-0000-000000000000",
    )

    assert (writer.root / relative).is_file()


@requires_opencv
def test_write__a_zero_area_box__still_produces_a_file(tmp_path: Path) -> None:
    """The degenerate crop path has to survive all the way to disk, not just the crop."""
    writer = ThumbnailWriter(root=tmp_path, width=16, height=16)

    relative = writer.write(
        gradient_frame(),
        (50, 50, 50, 50),
        camera_id="cam_01",
        timestamp_utc=MOMENT,
        sighting_id="aaaaaaaa-0000-0000-0000-000000000000",
    )

    assert (tmp_path / relative).is_file()


def test_from_settings__reads_the_store_and_the_thumbnail_shape_from_config(
    build_settings: Callable[..., Settings],
) -> None:
    """No hardcoded paths or sizes: the store location is a deployment decision."""
    settings = build_settings()
    writer = ThumbnailWriter.from_settings(settings)

    assert writer.root == settings.storage.thumbnail_dir
    assert writer.width == settings.detection.thumbnail_width
    assert writer.jpeg_quality == settings.detection.thumbnail_jpeg_quality
