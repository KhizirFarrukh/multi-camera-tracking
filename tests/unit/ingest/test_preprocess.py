"""Unit tests for ROI masking, preprocessing, and coordinate recovery.

The coordinate round-trip is the reason this module exists. A box reported in
resized-and-rotated space is a perfectly plausible box in the wrong place, and
nothing downstream can tell -- so a known point is pushed through every
transform, individually and composed, and mapped back.

The rotation inverse was wrong when first written, off by the frame width at one
corner and exact at another. That is what these tests are for.
"""

from __future__ import annotations

import numpy as np
import pytest

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest import CoordinateMapping, Preprocessor, RegionOfInterest, polygon_mask
from multicam_tracker.ingest.frame import Frame

pytestmark = pytest.mark.unit

WIDTH, HEIGHT = 160, 120
KNOWN_POINT = (100, 30)
REGION = [(60, 10), (150, 10), (150, 80), (60, 80)]


def _marked_image() -> np.ndarray:
    """Return a black image with one white pixel at the known point.

    Returns:
        A BGR image.
    """
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[KNOWN_POINT[1], KNOWN_POINT[0]] = 255
    return image


def _block_image(x0: int = 98, y0: int = 28, size: int = 6) -> np.ndarray:
    """Return an image with a small white block.

    A block rather than a pixel, because nearest-neighbour downsampling can drop
    a single pixel entirely -- which is a property of the resize, not of the
    mapping under test.

    Args:
        x0: Block left edge.
        y0: Block top edge.
        size: Block side length.

    Returns:
        A BGR image.
    """
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[y0 : y0 + size, x0 : x0 + size] = 255
    return image


def _found_point(image: np.ndarray) -> tuple[float, float]:
    """Return the position of the marked pixel in an image.

    Args:
        image: The processed image.

    Returns:
        ``(x, y)``.

    Raises:
        AssertionError: If the mark is not present, which would make the test
            vacuous rather than passing.
    """
    ys, xs = np.where(image[:, :, 0] > 200)
    assert len(xs) > 0, "the marked point was lost in preprocessing"
    return (float(xs[0]), float(ys[0]))


# ---------------------------------------------------------------------------
# The coordinate round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "preprocessor"),
    [
        ("identity", Preprocessor()),
        ("rotate 90", Preprocessor(rotation_degrees=90)),
        ("rotate 180", Preprocessor(rotation_degrees=180)),
        ("rotate 270", Preprocessor(rotation_degrees=270)),
        ("crop", Preprocessor(roi=RegionOfInterest(REGION, crop_to_bounds=True))),
        (
            "rotate and crop",
            Preprocessor(roi=RegionOfInterest(REGION, crop_to_bounds=True), rotation_degrees=90),
        ),
    ],
)
def test_a_known_point__maps_back_exactly_through_every_lossless_transform(
    name: str, preprocessor: Preprocessor
) -> None:
    """Exact, because none of these transforms discards a pixel.

    Anything less than exact here is an arithmetic error, and an arithmetic
    error in this direction puts every detection in the wrong place.
    """
    processed, mapping = preprocessor.apply(_marked_image())

    x, y = _found_point(processed)

    assert mapping.to_source(x, y) == pytest.approx(KNOWN_POINT), name


@pytest.mark.parametrize(
    ("name", "preprocessor"),
    [
        ("half resize", Preprocessor(target_width=80, target_height=60)),
        ("rotate and resize", Preprocessor(rotation_degrees=90, target_width=60, target_height=80)),
        (
            "crop, rotate, resize",
            Preprocessor(
                roi=RegionOfInterest(REGION, crop_to_bounds=True),
                rotation_degrees=90,
                target_width=45,
                target_height=35,
            ),
        ),
    ],
)
def test_a_known_block__maps_back_within_the_resize_granularity(
    name: str, preprocessor: Preprocessor
) -> None:
    """Under a downsample the mapping cannot beat the pixels the resize threw away.

    So the assertion is the honest one: the recovered centre is within one
    processed pixel of the truth, which is the best any inverse can do.
    """
    processed, mapping = preprocessor.apply(_block_image())

    ys, xs = np.where(processed[:, :, 0] > 200)
    assert len(xs) > 0, name
    centre_x = (float(xs.min()) + float(xs.max())) / 2
    centre_y = (float(ys.min()) + float(ys.max())) / 2

    back_x, back_y = mapping.to_source(centre_x, centre_y)
    granularity = max(1 / mapping.scale_x, 1 / mapping.scale_y)

    assert abs(back_x - 100.5) <= granularity, name
    assert abs(back_y - 30.5) <= granularity, name


def test_a_bounding_box__comes_back_with_its_corners_in_the_right_order() -> None:
    """A rotation swaps which corner is top-left.

    Returning them unsorted produces a negative-width box, which the Sighting
    model rejects one layer too late to explain why.
    """
    mapping = CoordinateMapping(rotation_degrees=90, rotated_size=(HEIGHT, WIDTH))

    x1, y1, x2, y2 = mapping.bbox_to_source([10, 20, 40, 60])

    assert x1 < x2
    assert y1 < y2


def test_a_bounding_box_without_four_coordinates__is_rejected() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match=r"\[x1, y1, x2, y2\]"):
        CoordinateMapping().bbox_to_source([10, 20, 30])


def test_the_identity_mapping__returns_coordinates_unchanged() -> None:
    """The common case must cost nothing and change nothing."""
    mapping = CoordinateMapping(rotated_size=(WIDTH, HEIGHT))

    assert mapping.to_source(42.0, 17.0) == (42.0, 17.0)
    assert mapping.bbox_to_source([1, 2, 3, 4]) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Region of interest
# ---------------------------------------------------------------------------


def test_an_roi_mask__zeroes_exactly_the_region_outside_the_polygon() -> None:
    """Inside survives, outside is filled -- the sidewalk stops producing detections."""
    image = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
    preprocessor = Preprocessor(roi=RegionOfInterest(REGION))

    processed, _mapping = preprocessor.apply(image)

    assert processed[45, 100, 0] == 200
    assert processed[5, 5, 0] == 0


def test_a_polygon_extending_beyond_the_frame__is_clipped_rather_than_rejected() -> None:
    """An operator tracing a region over the far pavement should not have to be exact."""
    oversized = [(-50, -50), (400, -50), (400, 300), (-50, 300)]

    mask = polygon_mask(oversized, WIDTH, HEIGHT)

    assert mask.shape == (HEIGHT, WIDTH)
    assert mask.all()


def test_a_polygon_with_too_few_vertices__is_rejected() -> None:
    """Two points is a line, and a line encloses nothing."""
    with pytest.raises(IngestError, match="at least three vertices"):
        polygon_mask([(0, 0), (10, 10)], WIDTH, HEIGHT)


def test_a_degenerate_polygon__is_rejected() -> None:
    """It would mask out the whole frame, and a camera that silently sees nothing
    is worse than one that fails to start."""
    with pytest.raises(IngestError, match="degenerate"):
        polygon_mask([(10, 10), (20, 10), (30, 10)], WIDTH, HEIGHT)


def test_cropping_to_the_region__shrinks_the_frame_to_its_bounds() -> None:
    """Cheaper for the detector, and the mapping accounts for it."""
    preprocessor = Preprocessor(roi=RegionOfInterest(REGION, crop_to_bounds=True))

    processed, mapping = preprocessor.apply(_marked_image())

    assert processed.shape[1] == 150 - 60
    assert processed.shape[0] == 80 - 10
    assert mapping.crop_origin == (60, 10)


# ---------------------------------------------------------------------------
# Resize and rotation
# ---------------------------------------------------------------------------


def test_resize__changes_the_processed_dimensions_as_configured() -> None:
    """What the detector asked for is what it gets."""
    processed, mapping = Preprocessor(target_width=80, target_height=60).apply(_marked_image())

    assert processed.shape[:2] == (60, 80)
    assert mapping.scale_x == pytest.approx(0.5)
    assert mapping.scale_y == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("degrees", "expected"), [(90, (HEIGHT, WIDTH)), (180, (WIDTH, HEIGHT)), (270, (HEIGHT, WIDTH))]
)
def test_rotation__transposes_the_frame_for_right_angles(
    degrees: int, expected: tuple[int, int]
) -> None:
    """A camera mounted sideways produces portrait frames, and the sizes must follow."""
    processed, _mapping = Preprocessor(rotation_degrees=degrees).apply(_marked_image())

    assert (processed.shape[1], processed.shape[0]) == expected


def test_an_arbitrary_rotation_angle__is_rejected() -> None:
    """Its inverse is not exact: a box mapped back is larger than the object.

    A camera mounted at 37 degrees is a mounting problem, not a software one.
    """
    with pytest.raises(IngestError, match="right angle"):
        Preprocessor(rotation_degrees=37)


def test_a_half_specified_resize__is_rejected() -> None:
    """One dimension alone leaves the aspect ratio to be guessed at."""
    with pytest.raises(IngestError, match="both a width and a height"):
        Preprocessor(target_width=640)


def test_non_positive_resize_dimensions__are_rejected() -> None:
    """Boundary."""
    with pytest.raises(IngestError, match="must be positive"):
        Preprocessor(target_width=0, target_height=480)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def test_preprocessing_a_frame__leaves_every_timestamp_untouched() -> None:
    """The image changes; the frame's claim about when it was captured does not."""
    from datetime import UTC, datetime

    moment = datetime(2026, 8, 10, 14, 0, tzinfo=UTC)
    frame = Frame(
        image=_marked_image(),
        frame_index=17,
        raw_timestamp=moment,
        timestamp_utc=moment,
        source_id="src",
        camera_id="cam_01",
        is_keyframe=True,
    )

    processed, _mapping = Preprocessor(target_width=80, target_height=60).apply_to_frame(frame)

    assert processed.raw_timestamp == frame.raw_timestamp
    assert processed.timestamp_utc == frame.timestamp_utc
    assert processed.frame_index == 17
    assert processed.is_keyframe
    assert processed.shape == (80, 60)


def test_an_identity_preprocessor__says_so() -> None:
    """Lets a caller skip the work entirely rather than copying every frame."""
    assert Preprocessor().is_identity
    assert not Preprocessor(rotation_degrees=90).is_identity
