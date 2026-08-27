"""Per-camera masking, resizing, rotation -- and getting the coordinates back.

Every camera needs something done to its frames before detection: a mask over
the neighbouring property, a resize because the detector wants 640 pixels, a
rotation because the installer mounted it sideways. Each of those changes where
things are in the image.

**Emitted coordinates are always in original source-frame space.** This is the
rule the whole module exists to keep. A detector run on a half-size rotated
frame reports a box in *that* frame's coordinates; storing it as-is puts the
vehicle in the wrong part of the picture, and nothing downstream can tell --
the box is a perfectly plausible box, in the wrong place. Thumbnails get cropped
from the wrong region, review shows the wrong car, and every one of those
failures looks like a detector problem rather than an arithmetic one.

So a :class:`Preprocessor` returns both the processed image and the
:class:`CoordinateMapping` that undoes it, and the mapping is tested through
resize, rotation, and crop -- individually and composed -- with a known point.

Rotation is limited to right angles on purpose. An arbitrary angle needs an
interpolating warp (OpenCV), and more importantly its inverse is not exact:
mapping a box back through a 37-degree rotation gives an axis-aligned box that
is bigger than the object. Right angles are exact in both directions, and a
camera mounted at 37 degrees is a mounting problem rather than a software one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from multicam_tracker.exceptions import IngestError
from multicam_tracker.ingest.frame import Frame

__all__ = [
    "CoordinateMapping",
    "Preprocessor",
    "RegionOfInterest",
    "polygon_mask",
]

_POLYGON_MIN_VERTICES = 3
_BBOX_LENGTH = 4
_VALID_ROTATIONS = (0, 90, 180, 270)


def polygon_mask(polygon: list[tuple[int, int]], width: int, height: int) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of the pixels inside a polygon.

    Even-odd ray casting, evaluated per row: it needs no OpenCV, handles
    concave shapes, and a mask is computed once per camera rather than per
    frame.

    Args:
        polygon: Vertices as ``(x, y)`` in source-frame coordinates. Vertices
            outside the frame are clipped by the rasterisation rather than
            rejected -- an operator drawing a region over the far pavement
            should not have to trace the frame edge exactly.
        width: Frame width.
        height: Frame height.

    Returns:
        A ``(height, width)`` mask, ``True`` inside the polygon.

    Raises:
        IngestError: If the polygon has fewer than three vertices or encloses no
            area. A degenerate region would mask out the entire frame, and a
            camera that silently sees nothing is worse than one that fails to
            start.
    """
    if len(polygon) < _POLYGON_MIN_VERTICES:
        raise IngestError(
            "A region of interest needs at least three vertices",
            {"vertices": len(polygon)},
        )

    points = np.array(polygon, dtype=np.float64)
    area = 0.5 * abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], 1))
            - np.dot(points[:, 1], np.roll(points[:, 0], 1))
        )
    )
    if area <= 0.0:
        raise IngestError(
            "A region of interest must enclose some area; this polygon is degenerate "
            "and would mask out the whole frame",
            {"polygon": polygon},
        )

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    mask = np.zeros((height, width), dtype=bool)

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    for index in range(len(points)):
        x1, y1 = x_coords[index], y_coords[index]
        x2, y2 = x_coords[index - 1], y_coords[index - 1]
        # Standard even-odd test: count edge crossings to the left of each pixel.
        straddles = (y_grid < y1) != (y_grid < y2)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = x1 + (y_grid - y1) / np.where(y2 == y1, np.nan, y2 - y1) * (x2 - x1)
        mask ^= straddles & (x_grid < np.nan_to_num(crossing_x, nan=-1.0))

    return mask


@dataclass(frozen=True)
class CoordinateMapping:
    """How to get from processed-frame coordinates back to the original.

    The inverse of whatever the preprocessor did, applied in reverse order:
    undo the resize, then the rotation, then the crop.

    Args:
        crop_origin: ``(x, y)`` of the crop's top-left in the *rotated* frame.
        rotation_degrees: Rotation that was applied, anticlockwise.
        scale_x: Horizontal resize factor that was applied.
        scale_y: Vertical resize factor that was applied.
        rotated_size: ``(width, height)`` after rotation, before cropping.
    """

    crop_origin: tuple[int, int] = (0, 0)
    rotation_degrees: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotated_size: tuple[int, int] = (0, 0)

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map one processed-frame point back to source-frame coordinates.

        Args:
            x: Horizontal position in the processed frame.
            y: Vertical position in the processed frame.

        Returns:
            The point in original source-frame coordinates.
        """
        # 1. undo the resize
        unscaled_x = x / self.scale_x
        unscaled_y = y / self.scale_y

        # 2. undo the crop, returning to the rotated frame
        rotated_x = unscaled_x + self.crop_origin[0]
        rotated_y = unscaled_y + self.crop_origin[1]

        # 3. undo the rotation.
        #
        # np.rot90(k=1) maps source (x, y) to (y, source_width - 1 - x), so the
        # inverse reads the source width off the *rotated height*. Getting the
        # off-by-one wrong here is the classic version of this bug: the box is
        # a pixel out at one corner of the frame and forty out at the other,
        # which reads as a flaky detector rather than as arithmetic.
        rotated_width, rotated_height = self.rotated_size
        if self.rotation_degrees == 90:
            return (rotated_height - 1 - rotated_y, rotated_x)
        if self.rotation_degrees == 180:
            return (rotated_width - 1 - rotated_x, rotated_height - 1 - rotated_y)
        if self.rotation_degrees == 270:
            return (rotated_y, rotated_width - 1 - rotated_x)
        return (rotated_x, rotated_y)

    def bbox_to_source(self, bbox: list[int]) -> list[int]:
        """Map a bounding box back to source-frame coordinates.

        Both corners are mapped and then re-ordered, because a rotation swaps
        which corner is top-left. Returning the corners unsorted would produce a
        box with negative width, which the Sighting model rejects -- loudly, but
        one layer too late to say why.

        Args:
            bbox: ``[x1, y1, x2, y2]`` in the processed frame.

        Returns:
            ``[x1, y1, x2, y2]`` in source-frame coordinates, rounded to whole
            pixels.

        Raises:
            IngestError: If the box does not have four coordinates.
        """
        if len(bbox) != _BBOX_LENGTH:
            raise IngestError("A bounding box must be [x1, y1, x2, y2]", {"bbox": bbox})

        first = self.to_source(bbox[0], bbox[1])
        second = self.to_source(bbox[2], bbox[3])

        return [
            round(min(first[0], second[0])),
            round(min(first[1], second[1])),
            round(max(first[0], second[0])),
            round(max(first[1], second[1])),
        ]


@dataclass(frozen=True)
class RegionOfInterest:
    """A per-camera polygon limiting where detections are looked for.

    Args:
        polygon: Vertices as ``(x, y)`` in source-frame coordinates.
        crop_to_bounds: Crop the frame to the polygon's bounding box as well as
            masking. Cheaper for the detector, and the mapping accounts for it.
    """

    polygon: list[tuple[int, int]]
    crop_to_bounds: bool = False

    def bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return the polygon's bounding box, clipped to the frame.

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            ``(x1, y1, x2, y2)``, clipped so a region drawn past the frame edge
            is trimmed rather than rejected.
        """
        xs = [point[0] for point in self.polygon]
        ys = [point[1] for point in self.polygon]
        return (
            max(0, min(xs)),
            max(0, min(ys)),
            min(width, max(xs)),
            min(height, max(ys)),
        )


@dataclass
class Preprocessor:
    """Applies one camera's frame preparation, and remembers how to undo it.

    Order is fixed and deliberate: mask, then rotate, then crop, then resize.
    The mask is expressed in source coordinates so it is applied before anything
    moves; the resize is last so it operates on the smallest image.

    Args:
        roi: Region of interest, if any.
        rotation_degrees: Right-angle rotation to apply, anticlockwise.
        target_width: Resize target width. Requires a height too.
        target_height: Resize target height.
        mask_fill: Value written outside the region. Black, so a masked area
            reads as "nothing here" rather than as an object.

    Raises:
        IngestError: If the rotation is not a right angle, or only one resize
            dimension is given.
    """

    roi: RegionOfInterest | None = None
    rotation_degrees: int = 0
    target_width: int | None = None
    target_height: int | None = None
    mask_fill: int = 0
    _mask_cache: dict[tuple[int, int], npt.NDArray[np.bool_]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            IngestError: If the rotation is not a right angle, or the resize is
                half-specified.
        """
        if self.rotation_degrees not in _VALID_ROTATIONS:
            raise IngestError(
                "Rotation must be a right angle. An arbitrary angle needs an "
                "interpolating warp, and its inverse is not exact -- a box mapped back "
                "through it is larger than the object it bounds",
                {"rotation_degrees": self.rotation_degrees, "allowed": list(_VALID_ROTATIONS)},
            )
        if (self.target_width is None) != (self.target_height is None):
            raise IngestError(
                "Resize needs both a width and a height; one alone leaves the aspect "
                "ratio to be guessed at",
                {"target_width": self.target_width, "target_height": self.target_height},
            )
        if self.target_width is not None and (self.target_width <= 0 or self.target_height <= 0):  # type: ignore[operator]
            raise IngestError(
                "Resize dimensions must be positive",
                {"target_width": self.target_width, "target_height": self.target_height},
            )

    @property
    def is_identity(self) -> bool:
        """Return whether this preprocessor would change nothing."""
        return self.roi is None and self.rotation_degrees == 0 and self.target_width is None

    def apply(
        self, image: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], CoordinateMapping]:
        """Prepare one image and return the mapping that undoes it.

        Args:
            image: The source-frame image.

        Returns:
            ``(processed_image, mapping)``.
        """
        height, width = image.shape[:2]
        working = image

        if self.roi is not None:
            working = self._masked(working, width, height)

        rotated = self._rotated(working)
        rotated_height, rotated_width = rotated.shape[:2]

        crop_origin = (0, 0)
        if self.roi is not None and self.roi.crop_to_bounds:
            rotated, crop_origin = self._cropped(rotated, width, height)

        scale_x = scale_y = 1.0
        if self.target_width is not None and self.target_height is not None:
            cropped_height, cropped_width = rotated.shape[:2]
            rotated = self._resized(rotated, self.target_width, self.target_height)
            scale_x = self.target_width / cropped_width
            scale_y = self.target_height / cropped_height

        return rotated, CoordinateMapping(
            crop_origin=crop_origin,
            rotation_degrees=self.rotation_degrees,
            scale_x=scale_x,
            scale_y=scale_y,
            rotated_size=(rotated_width, rotated_height),
        )

    def apply_to_frame(self, frame: Frame) -> tuple[Frame, CoordinateMapping]:
        """Prepare one frame, keeping every piece of its metadata.

        Args:
            frame: The source frame.

        Returns:
            ``(processed_frame, mapping)``. The timestamps are untouched: they
            are the frame's claim about when it was captured, and no amount of
            image processing changes that.
        """
        processed, mapping = self.apply(frame.image)
        return frame.replacing_image(processed), mapping

    # -- steps -------------------------------------------------------------

    def _mask_for(self, width: int, height: int) -> npt.NDArray[np.bool_]:
        """Return the cached polygon mask for one frame size.

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            The boolean mask.
        """
        key = (width, height)
        if key not in self._mask_cache:
            assert self.roi is not None
            self._mask_cache[key] = polygon_mask(self.roi.polygon, width, height)
        return self._mask_cache[key]

    def _masked(
        self, image: npt.NDArray[np.uint8], width: int, height: int
    ) -> npt.NDArray[np.uint8]:
        """Zero everything outside the region of interest.

        Args:
            image: The source image.
            width: Frame width.
            height: Frame height.

        Returns:
            A new image with the outside filled.
        """
        mask = self._mask_for(width, height)
        masked = image.copy()
        masked[~mask] = self.mask_fill
        return masked

    def _rotated(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Apply the configured right-angle rotation.

        Args:
            image: The image to rotate.

        Returns:
            The rotated image, or the input when no rotation is configured.
        """
        if self.rotation_degrees == 0:
            return image
        return np.rot90(image, k=self.rotation_degrees // 90).copy()

    def _cropped(
        self, image: npt.NDArray[np.uint8], source_width: int, source_height: int
    ) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
        """Crop to the region's bounding box in the rotated frame.

        Args:
            image: The rotated image.
            source_width: Width before rotation.
            source_height: Height before rotation.

        Returns:
            ``(cropped_image, crop_origin)`` where the origin is in rotated-frame
            coordinates.
        """
        assert self.roi is not None
        x1, y1, x2, y2 = self.roi.bounds(source_width, source_height)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        rotated_corners = [
            self._rotate_point(x, y, source_width, source_height) for x, y in corners
        ]

        left = int(max(0, min(point[0] for point in rotated_corners)))
        top = int(max(0, min(point[1] for point in rotated_corners)))
        right = int(min(image.shape[1], max(point[0] for point in rotated_corners)))
        bottom = int(min(image.shape[0], max(point[1] for point in rotated_corners)))

        if right <= left or bottom <= top:
            # The region fell entirely outside the frame after rotation. Better
            # to hand back the whole frame than an empty array a detector would
            # crash on.
            return image, (0, 0)

        return image[top:bottom, left:right].copy(), (left, top)

    def _rotate_point(self, x: int, y: int, width: int, height: int) -> tuple[float, float]:
        """Map a source-frame point into rotated-frame coordinates.

        Args:
            x: Horizontal position in the source frame.
            y: Vertical position in the source frame.
            width: Source frame width.
            height: Source frame height.

        Returns:
            The point after rotation.
        """
        if self.rotation_degrees == 90:
            return (float(y), float(width - 1 - x))
        if self.rotation_degrees == 180:
            return (float(width - 1 - x), float(height - 1 - y))
        if self.rotation_degrees == 270:
            return (float(height - 1 - y), float(x))
        return (float(x), float(y))

    @staticmethod
    def _resized(image: npt.NDArray[np.uint8], width: int, height: int) -> npt.NDArray[np.uint8]:
        """Resize by nearest-neighbour sampling.

        Nearest-neighbour rather than an interpolating resize because this must
        work without OpenCV, and the mapping back to source coordinates is a
        pure scale either way. A camera whose detector wants better resampling
        can have OpenCV do it; the coordinate arithmetic does not change.

        Args:
            image: The image to resize.
            width: Target width.
            height: Target height.

        Returns:
            The resized image.
        """
        source_height, source_width = image.shape[:2]
        rows = (np.arange(height) * source_height // height).clip(0, source_height - 1)
        columns = (np.arange(width) * source_width // width).clip(0, source_width - 1)
        resized: npt.NDArray[np.uint8] = image[rows][:, columns].copy()
        return resized
