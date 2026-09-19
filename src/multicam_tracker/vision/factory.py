"""Building the configured detector, and nothing else knowing which one it is.

The stage 11 exit criterion is that the detector implementation is swappable
*purely through configuration*. This is where that is true or not: one function,
one setting, and no caller anywhere that names a concrete class.

That matters twice over. In CI, ``MCT_DETECTION__DETECTOR_BACKEND=fixture``
lets every downstream stage run with no weights, no GPU, and no network. In
production, replacing the model is a config change and a weights file rather
than a code change -- which is what makes it possible to replace it at all.
"""

from __future__ import annotations

from multicam_tracker.config import Settings
from multicam_tracker.exceptions import ConfigurationError
from multicam_tracker.logging_config import get_logger
from multicam_tracker.vision.detector_protocol import Detector
from multicam_tracker.vision.fake_detector import FakeDetector
from multicam_tracker.vision.fixture_detector import FixtureDetector
from multicam_tracker.vision.yolo_detector import YoloDetector

__all__ = ["create_detector"]

logger = get_logger(__name__)


def create_detector(settings: Settings) -> Detector:
    """Return the detector this deployment is configured to use.

    Args:
        settings: Loaded settings. The backend comes from
            ``detection.detector_backend``; each backend reads its own
            parameters from the sections that describe it.

    Returns:
        A detector. Nothing is loaded or read from disk beyond what the chosen
        backend needs to exist -- in particular the YOLO backend does not touch
        its weights until it is first asked to detect something.

    Raises:
        ConfigurationError: If the fixture backend is selected without a
            recording to replay. Falling back to an empty fixture would make
            every clip look like an empty road, which no test would catch.
    """
    backend = settings.detection.detector_backend

    if backend == "yolo":
        detector: Detector = YoloDetector.from_settings(settings)
    elif backend == "fake":
        # Scripted with nothing: a fake that invents detections from
        # configuration would be placeholder logic, which the contract forbids.
        # Tests construct FakeDetector directly with the script they mean.
        detector = FakeDetector()
    elif backend == "fixture":
        path = settings.detection.fixture_detections_path
        if path is None:
            raise ConfigurationError(
                "Missing required setting: the fixture detector needs a recording to replay",
                {
                    "field": "detection.fixture_detections_path",
                    "env_var": "MCT_DETECTION__FIXTURE_DETECTIONS_PATH",
                },
            )
        detector = FixtureDetector.from_file(path)
    else:  # pragma: no cover - the Literal type makes this unreachable
        raise ConfigurationError(
            "Unknown detector backend",
            {"field": "detection.detector_backend", "value": backend},
        )

    logger.info(
        "detector_created",
        backend=backend,
        model_id=detector.model_id,
    )
    return detector
