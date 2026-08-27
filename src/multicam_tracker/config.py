"""Application configuration.

All configuration enters the process through :class:`Settings`. Nothing in
``src/`` hardcodes a host, path, credential, or threshold; values come from
environment variables (prefixed ``MCT_``, nested with ``__``), a local ``.env``
file, or ``config/thresholds.yaml``.

Precedence, highest first:

1. explicit keyword arguments to ``Settings(...)`` -- used by tests
2. environment variables -- ``MCT_DATABASE__PORT=5544``
3. the ``.env`` file
4. ``config/thresholds.yaml`` -- supplies the ``thresholds`` section
5. field defaults declared below

Two deliberate choices are worth knowing about before you read the code.

**Thresholds have no defaults in Python.** Every field of
:class:`ThresholdSettings` is required, so the values can only come from
``thresholds.yaml``. That is what makes "no magic numbers hardcoded in logic"
(global contract, ``confidence_model.thresholds_config``) enforceable rather
than aspirational: a threshold added to the model but forgotten in the YAML
fails loudly at startup instead of quietly defaulting.

**No secret has a default value.** ``database.password`` and ``api.secret_key``
default to empty. Non-secret connection defaults (localhost, 5432) exist so a
fresh clone starts, but nothing that could be a credential is baked in.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from multicam_tracker.exceptions import ConfigurationError

__all__ = [
    "ApiSettings",
    "DatabaseSettings",
    "IngestSettings",
    "PathingSettings",
    "RetentionSettings",
    "Settings",
    "StorageSettings",
    "ThresholdSettings",
    "ThresholdsYamlSource",
    "TimeSyncSettings",
    "TopologySettings",
    "VisionSettings",
    "default_thresholds_file",
    "get_settings",
    "reset_settings_cache",
]

ENV_PREFIX = "MCT_"
"""Prefix every environment variable must carry to be seen by :class:`Settings`."""

ENV_NESTED_DELIMITER = "__"
"""Separator between a section name and a field name, e.g. ``MCT_DATABASE__PORT``."""

THRESHOLDS_FILE_ENV_VAR = f"{ENV_PREFIX}THRESHOLDS_FILE"

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent


def default_thresholds_file() -> Path:
    """Locate ``config/thresholds.yaml`` for a source checkout.

    Resolves relative to this file first, which is correct for an editable
    install, and falls back to the current working directory so the package
    still finds a config tree when installed as a wheel elsewhere. Set
    ``MCT_THRESHOLDS_FILE`` to bypass the search entirely.

    Returns:
        Path to the thresholds file. The path is not guaranteed to exist; the
        settings source reports a clear error if it does not.
    """
    checkout_candidate = _REPO_ROOT / "config" / "thresholds.yaml"
    if checkout_candidate.is_file():
        return checkout_candidate
    return Path.cwd() / "config" / "thresholds.yaml"


class _StrictSection(BaseModel):
    """Base for settings sections: unknown keys are an error, not a shrug.

    A typo in ``MCT_DATABSE__PORT`` that is silently ignored produces a service
    connecting to the wrong database with no signal at all. Forbidding extras
    turns that into a startup failure.
    """

    model_config = ConfigDict(extra="forbid")


class DatabaseSettings(_StrictSection):
    """PostgreSQL connection settings."""

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "postgres"
    password: SecretStr = SecretStr("")
    dbname: str = "multicam"
    pool_size: int = Field(default=5, ge=1)
    pool_max_overflow: int = Field(default=10, ge=0)
    connect_timeout_sec: int = Field(default=10, ge=1)

    @property
    def dsn(self) -> str:
        """Return a SQLAlchemy/psycopg connection URL including the password.

        Returns:
            A ``postgresql+psycopg://`` URL. User and password are
            percent-encoded so values containing ``@`` or ``:`` connect
            correctly.
        """
        user = quote(self.user, safe="")
        password = quote(self.password.get_secret_value(), safe="")
        credentials = f"{user}:{password}" if password else user
        return f"postgresql+psycopg://{credentials}@{self.host}:{self.port}/{self.dbname}"

    @property
    def safe_dsn(self) -> str:
        """Return the connection URL with the password masked, for logging.

        Returns:
            The same URL as :attr:`dsn` with the password replaced by ``***``.
        """
        user = quote(self.user, safe="")
        credentials = f"{user}:***" if self.password.get_secret_value() else user
        return f"postgresql+psycopg://{credentials}@{self.host}:{self.port}/{self.dbname}"


class StorageSettings(_StrictSection):
    """Filesystem locations for generated artefacts."""

    thumbnail_dir: Path = Path("storage/thumbnails")
    media_dir: Path = Path("storage/media")

    @model_validator(mode="after")
    def _expand_user_paths(self) -> StorageSettings:
        """Expand a leading ``~`` in configured directories.

        Returns:
            The instance with user-relative paths resolved to absolute ones.
        """
        self.thumbnail_dir = self.thumbnail_dir.expanduser()
        self.media_dir = self.media_dir.expanduser()
        return self


class VisionSettings(_StrictSection):
    """Model locations and inference parameters for the CV stages (11-13)."""

    vehicle_detector_path: Path = Path("weights/vehicle_detector.pt")
    plate_detector_path: Path = Path("weights/plate_detector.pt")
    reid_model_path: Path = Path("weights/reid_osnet.pt")
    ocr_model_dir: Path | None = None
    embedding_dim: int = Field(default=512, ge=1)
    device: Literal["cpu", "cuda"] = "cpu"
    frame_sample_rate_fps: float = Field(default=3.0, gt=0.0, le=120.0)
    detection_min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)


class TopologySettings(_StrictSection):
    """Operational knobs for the camera graph (stage 04).

    The *speed model* used to derive travel times lives in the ``defaults``
    block of ``topology.yaml``, not here: it describes a particular road
    network, so it belongs with the topology it describes. What lives here is
    everything that governs how the graph is *queried*, which is a property of
    this deployment rather than of the roads.
    """

    topology_file: Path = Path("config/topology.yaml")
    implausible_speed_kph: float = Field(
        default=200.0,
        gt=0.0,
        description="Implied speed above which a link is warned about, not rejected",
    )
    min_redetection_gap_sec: float = Field(
        default=60.0,
        ge=0.0,
        description="Below this gap, two sightings on one camera are one pass, not two transits",
    )
    plausibility_decay_half_life_sec: float = Field(
        default=120.0,
        gt=0.0,
        description="Seconds outside a travel window at which the plausibility score halves",
    )
    unlinked_plausibility_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Score for a camera pair the topology does not connect. Non-zero on purpose: "
            "an undeclared route is unexplained, not impossible, and a hard zero would "
            "discard a real detour instead of merely penalising it."
        ),
    )
    max_visited_nodes: int = Field(
        default=10_000,
        ge=1,
        description="Expansion cap for multi-hop reachability; results are flagged truncated",
    )


class ThresholdSettings(_StrictSection):
    """Decision thresholds, loaded from ``config/thresholds.yaml``.

    Every field is required on purpose. See the module docstring.
    """

    plate_auto_accept_min_confidence: float = Field(ge=0.0, le=1.0)
    plate_fuzzy_max_edit_distance: int = Field(ge=0)
    plate_fuzzy_max_weighted_distance: float = Field(ge=0.0)
    plate_max_length_delta: int = Field(ge=0)
    plate_confusion_substitution_cost: float = Field(ge=0.0, le=1.0)
    plate_exact_method_weight: float = Field(ge=0.0, le=1.0)
    plate_fuzzy_method_weight: float = Field(ge=0.0, le=1.0)
    plate_distance_penalty_per_unit: float = Field(ge=0.0)
    plate_review_min_confidence: float = Field(ge=0.0, le=1.0)
    embedding_auto_accept_min_similarity: float = Field(ge=-1.0, le=1.0)
    embedding_review_min_similarity: float = Field(ge=-1.0, le=1.0)
    embedding_margin_min: float = Field(ge=0.0, le=2.0)
    embedding_only_score_ceiling: float = Field(ge=0.0, le=1.0)
    embedding_agreement_boost: float = Field(ge=0.0, le=1.0)
    embedding_disagreement_similarity: float = Field(ge=-1.0, le=1.0)
    embedding_max_references: int = Field(ge=1)
    hop_implausible_penalty: float = Field(ge=0.0, le=1.0)
    path_node_inclusion_bonus: float = Field(ge=0.0, le=1.0)
    path_gap_edge_penalty: float = Field(ge=0.0, le=1.0)
    path_ambiguity_margin_min: float = Field(ge=0.0, le=1.0)
    path_weakest_link_tolerance: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _review_band_is_ordered(self) -> ThresholdSettings:
        """Reject an inverted review band.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the auto-accept similarity is below the review
                similarity, which would leave no band in which a match is sent
                for human review.
        """
        if self.embedding_auto_accept_min_similarity < self.embedding_review_min_similarity:
            msg = (
                "embedding_auto_accept_min_similarity must be >= "
                "embedding_review_min_similarity; otherwise no review band exists"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _visual_evidence_cannot_outrank_plate_evidence(self) -> ThresholdSettings:
        """Reject a visual score ceiling that could match a plate match.

        The rule that appearance never outranks a plate is enforced structurally
        by this one inequality. Leaving it as a comment beside the YAML would
        let a plausible-looking edit invert the evidence hierarchy silently, and
        the symptom -- a visual guess presented with the authority of a plate
        read -- would surface as an operator acting on the wrong vehicle.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the ceiling reaches the weakest score a retained
                plate match can carry.
        """
        weakest_plate_score = self.plate_review_min_confidence * self.plate_exact_method_weight
        if self.embedding_only_score_ceiling >= weakest_plate_score:
            msg = (
                f"embedding_only_score_ceiling ({self.embedding_only_score_ceiling}) must be "
                f"< plate_review_min_confidence x plate_exact_method_weight "
                f"({weakest_plate_score}); otherwise a visual-only match can outrank a "
                f"plate match"
            )
            raise ValueError(msg)
        return self


class IngestSettings(_StrictSection):
    """Frame sampling, motion gating, and live buffering (stage 10).

    The defaults encode the stage's two standing trades: sample at a rate a
    detector can keep up with rather than at whatever the camera produces, and
    on a live source prefer the newest frame to a complete record of an
    increasingly stale past.
    """

    target_fps: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Detection cadence. Five a second sees every vehicle that passes at "
            "urban speeds while costing a sixth of a 30 fps camera's frames."
        ),
    )
    motion_sensitivity: float = Field(
        default=2.0,
        ge=0.0,
        description=(
            "Mean greyscale difference above which a frame counts as changed. "
            "Tuned above sensor noise and below a vehicle entering frame."
        ),
    )
    motion_force_interval_sec: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Longest a frame may go unsampled however static the scene. This is what "
            "keeps a parked vehicle in the record: a car that stops being detected "
            "looks exactly like a car that drove away."
        ),
    )
    live_buffer_frames: int = Field(
        default=8,
        ge=1,
        description=(
            "Frames held between a live decoder and its consumer. Small on purpose: "
            "a deep buffer trades the latency a live view exists for against a "
            "completeness nobody watching can use."
        ),
    )
    live_read_timeout_sec: float = Field(
        default=10.0,
        gt=0.0,
        description="How long a stream may go quiet before it is treated as dead",
    )
    max_queued_frames: int = Field(
        default=64,
        ge=1,
        description=(
            "Frames the multi-source reader may hold across all cameras. Bounds "
            "memory: the queue holds decoded images, and an unbounded one is a leak "
            "with a slow consumer attached."
        ),
    )


class TimeSyncSettings(_StrictSection):
    """Clock offset, drift, and temporal integrity policy (stage 09).

    The premise of the whole system is that timestamps from independent cameras
    are comparable. These values decide when that premise stops being taken on
    trust.
    """

    verification_staleness_hours: float = Field(
        default=168.0,
        gt=0.0,
        description=(
            "How long a camera's clock verification stays good for. A week by default: "
            "long enough not to nag, short enough that a camera drifting since its last "
            "check is caught before a month of routes are built on it."
        ),
    )
    drift_alert_ms: float = Field(
        default=2000.0,
        gt=0.0,
        description=(
            "Constant offset magnitude that raises an alert. Two seconds is well below "
            "the smallest plausible travel time in a typical topology, so an alert fires "
            "long before hop ordering itself becomes unreliable."
        ),
    )
    drift_rate_alert_ms_per_hour: float = Field(
        default=500.0,
        gt=0.0,
        description="Fitted drift rate that raises an alert, independent of current magnitude",
    )
    min_reference_passes: int = Field(
        default=3,
        ge=2,
        description=(
            "Fewest reference passes per camera before an offset estimate is emitted. "
            "Below this the system is underdetermined and the estimator refuses rather "
            "than returning an arbitrary solution."
        ),
    )
    watermark_lateness_sec: float = Field(
        default=120.0,
        ge=0.0,
        description="How far behind the watermark a live record may arrive and still be accepted",
    )
    block_on_drift_alert: bool = Field(
        default=False,
        description=(
            "Whether an active drift alert refuses reconstruction outright rather than "
            "attaching a caveat. Off by default: a flagged route an operator can weigh "
            "is more useful than no route, and the caveat travels with the trajectory."
        ),
    )


class PathingSettings(_StrictSection):
    """Operational knobs for trajectory reconstruction (stage 08).

    The *decision* values -- what makes a path preferable, when a result is too
    close to call -- live in ``thresholds.yaml`` with every other threshold.
    What lives here is how the search is executed: how far it looks, how much of
    it it will do, and how long it waits for a late-arriving sighting. Those are
    properties of a deployment rather than of the evidence.
    """

    max_skip_hops: int = Field(
        default=3,
        ge=1,
        description="Hop budget when testing whether an unlinked pair is reachable indirectly",
    )
    reorder_buffer_sec: float = Field(
        default=120.0,
        ge=0.0,
        description=(
            "How far back a late-arriving sighting may land and still trigger a suffix "
            "recomputation. Beyond this it is rejected rather than silently dropped: a "
            "stream delivering minutes out of order has a problem the operator should see."
        ),
    )
    k_best_default: int = Field(
        default=3, ge=1, description="Alternative paths enumerated when none is requested"
    )
    stop_gap_multiplier: float = Field(
        default=3.0,
        gt=1.0,
        description=(
            "Elapsed time beyond this multiple of the plausible maximum is reported as a "
            "stop or a departure from the network, not merely a slow transit"
        ),
    )


class RetentionSettings(_StrictSection):
    """Data retention policy (global contract, operational constraints).

    Enforced by the purge job in stage 18; the schema support lands in stage 03.
    """

    sighting_ttl_days: int = Field(default=90, ge=1)
    thumbnail_ttl_days: int = Field(default=30, ge=1)
    purge_enabled: bool = True

    @model_validator(mode="after")
    def _thumbnails_do_not_outlive_sightings(self) -> RetentionSettings:
        """Reject a thumbnail TTL longer than the sighting TTL.

        Returns:
            The validated instance.

        Raises:
            ValueError: If thumbnails would survive the sighting rows that
                reference them, leaving orphaned imagery behind after a purge.
        """
        if self.thumbnail_ttl_days > self.sighting_ttl_days:
            msg = (
                "thumbnail_ttl_days must be <= sighting_ttl_days; otherwise the "
                "purge job leaves orphaned thumbnails with no sighting row"
            )
            raise ValueError(msg)
        return self


class ApiSettings(_StrictSection):
    """FastAPI server and authentication settings (stage 16)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    auth_enabled: bool = False
    secret_key: SecretStr = SecretStr("")
    access_token_ttl_minutes: int = Field(default=60, ge=1)

    @property
    def has_secret_key(self) -> bool:
        """Return whether a non-empty secret key is configured."""
        return bool(self.secret_key.get_secret_value())


class ThresholdsYamlSource(PydanticBaseSettingsSource):
    """Settings source that supplies the ``thresholds`` section from YAML.

    Args:
        settings_cls: The settings class being populated.
        path: Path to the YAML file.
    """

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Return no per-field value; this source only populates ``thresholds``.

        Args:
            field: Field metadata, unused.
            field_name: Field name, echoed back unchanged.

        Returns:
            A ``(value, key, is_complex)`` triple indicating no value was found.
        """
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Read the YAML file and return it as the ``thresholds`` section.

        Returns:
            ``{"thresholds": {...}}`` mapping suitable for merging into the
            settings values.

        Raises:
            ConfigurationError: If the file is missing, unreadable, not valid
                YAML, or does not parse to a mapping.
        """
        if not self._path.is_file():
            raise ConfigurationError(
                "Thresholds file not found; set MCT_THRESHOLDS_FILE or create the file",
                {"field": "thresholds_file", "path": str(self._path)},
            )
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(
                "Thresholds file could not be read",
                {"field": "thresholds_file", "path": str(self._path), "reason": str(exc)},
            ) from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigurationError(
                "Thresholds file must contain a YAML mapping",
                {
                    "field": "thresholds_file",
                    "path": str(self._path),
                    "parsed_type": type(raw).__name__,
                },
            )
        return {"thresholds": raw}

    def __repr__(self) -> str:
        """Return an unambiguous representation including the source path."""
        return f"{type(self).__name__}(path={str(self._path)!r})"


class Settings(BaseSettings):
    """Root settings object.

    Instantiate through :func:`get_settings` in application code so the instance
    is built once and shared.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    thresholds_file: Path = Field(default_factory=default_thresholds_file)

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    topology: TopologySettings = Field(default_factory=TopologySettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    pathing: PathingSettings = Field(default_factory=PathingSettings)
    timesync: TimeSyncSettings = Field(default_factory=TimeSyncSettings)
    thresholds: ThresholdSettings
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML thresholds source below the environment sources.

        Ordering puts YAML above field defaults but below environment variables,
        so an operator can override a single threshold with
        ``MCT_THRESHOLDS__PLATE_FUZZY_MAX_EDIT_DISTANCE`` without editing the
        checked-in file.

        Args:
            settings_cls: The settings class being populated.
            init_settings: Values passed directly to the constructor.
            env_settings: Values from environment variables.
            dotenv_settings: Values from the ``.env`` file.
            file_secret_settings: Values from a secrets directory.

        Returns:
            The ordered tuple of settings sources, highest precedence first.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            ThresholdsYamlSource(settings_cls, cls._resolve_thresholds_path(init_settings)),
            file_secret_settings,
        )

    @classmethod
    def _resolve_thresholds_path(cls, init_settings: PydanticBaseSettingsSource) -> Path:
        """Resolve which YAML file the thresholds source should read.

        Args:
            init_settings: The init source, inspected for an explicit
                ``thresholds_file`` keyword so tests can point at a fixture.

        Returns:
            The resolved path, honouring constructor argument, then
            ``MCT_THRESHOLDS_FILE``, then the checkout default.
        """
        init_kwargs: dict[str, Any] = getattr(init_settings, "init_kwargs", {}) or {}
        explicit = init_kwargs.get("thresholds_file")
        if explicit is not None:
            return Path(explicit)

        from_env = os.environ.get(THRESHOLDS_FILE_ENV_VAR)
        if from_env:
            return Path(from_env)

        return default_thresholds_file()

    @model_validator(mode="after")
    def _check_deployment_invariants(self) -> Settings:
        """Enforce settings combinations that are unsafe rather than merely odd.

        Returns:
            The validated instance.

        Raises:
            ConfigurationError: If authentication is enabled without a secret
                key, or if a production environment runs with debug output or
                authentication disabled.
        """
        if self.api.auth_enabled and not self.api.has_secret_key:
            raise ConfigurationError(
                "Missing required setting: a secret key is required when API auth is enabled",
                {
                    "field": "api.secret_key",
                    "env_var": f"{ENV_PREFIX}API{ENV_NESTED_DELIMITER}SECRET_KEY",
                },
            )

        if self.environment == "production":
            if not self.api.auth_enabled:
                raise ConfigurationError(
                    "Missing required setting: API authentication cannot be disabled in production",
                    {"field": "api.auth_enabled", "environment": self.environment},
                )
            if self.debug:
                raise ConfigurationError(
                    "Invalid setting: debug output cannot be enabled in production",
                    {"field": "debug", "environment": self.environment},
                )
        return self

    @property
    def log_as_json(self) -> bool:
        """Return whether logs should render as JSON.

        Returns:
            ``False`` in debug mode, where the human-readable console renderer is
            more useful; ``True`` otherwise.
        """
        return not self.debug


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance, building it on first call.

    Returns:
        The cached :class:`Settings` instance. Repeat calls return the same
        object.

    Raises:
        ConfigurationError: If settings fail to load or validate. Pydantic's
            :class:`~pydantic.ValidationError` is translated here so callers
            only ever have to catch the project's own hierarchy, and the
            offending field paths are carried in the error context.
    """
    try:
        return Settings()
    except PydanticValidationError as exc:
        fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ConfigurationError(
            "Invalid configuration",
            {"fields": fields, "errors": details},
        ) from exc


def reset_settings_cache() -> None:
    """Clear the :func:`get_settings` cache.

    Tests that manipulate the environment call this so the next
    :func:`get_settings` rebuilds from the changed environment. Application code
    has no reason to call it.
    """
    get_settings.cache_clear()
