"""Unit tests for :mod:`multicam_tracker.config`.

Configuration is the one module every later stage depends on, and a silent
config bug is the kind that surfaces three stages downstream as an unexplainable
threshold behaviour. These tests pin down precedence, type strictness, and the
failure messages.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import SecretStr
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

from multicam_tracker.config import (
    DatabaseSettings,
    RetentionSettings,
    Settings,
    ThresholdSettings,
    ThresholdsYamlSource,
    get_settings,
    reset_settings_cache,
)
from multicam_tracker.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Loading and precedence
# ---------------------------------------------------------------------------


def test_settings__no_environment_variables__loads_defaults(
    build_settings: Callable[..., Settings],
) -> None:
    """A fresh clone with no environment configuration must start."""
    settings = build_settings()

    assert settings.environment == "local"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.database.host == "localhost"
    assert settings.database.port == 5432
    assert settings.api.auth_enabled is False


def test_settings__mct_prefixed_env_var__overrides_default(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[..., Settings]
) -> None:
    """A top-level setting is overridable through its MCT_ environment variable."""
    monkeypatch.setenv("MCT_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MCT_ENVIRONMENT", "staging")

    settings = build_settings()

    assert settings.log_level == "WARNING"
    assert settings.environment == "staging"


def test_settings__nested_delimiter_env_var__populates_nested_section(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[..., Settings]
) -> None:
    """MCT_DATABASE__PORT reaches Settings.database.port, not a top-level field."""
    monkeypatch.setenv("MCT_DATABASE__PORT", "5544")
    monkeypatch.setenv("MCT_DATABASE__DBNAME", "multicam_test")

    settings = build_settings()

    assert settings.database.port == 5544
    assert settings.database.dbname == "multicam_test"
    # Untouched siblings keep their defaults rather than being reset.
    assert settings.database.host == "localhost"


def test_settings__committed_yaml__thresholds_are_loaded(
    build_settings: Callable[..., Settings],
) -> None:
    """The committed thresholds file supplies exactly the contract's defaults."""
    thresholds = build_settings().thresholds

    assert thresholds.plate_auto_accept_min_confidence == pytest.approx(0.85)
    assert thresholds.plate_fuzzy_max_edit_distance == 2
    # 0.95 rather than the contract prior of 0.92: the stage 07 sweep found 0.92
    # auto-accepts 21 hard-negative decoys and 0.95 auto-accepts none, at no cost
    # to recall. See docs/REID_MATCHING.md.
    assert thresholds.embedding_auto_accept_min_similarity == pytest.approx(0.95)
    assert thresholds.embedding_review_min_similarity == pytest.approx(0.75)
    assert thresholds.embedding_margin_min == pytest.approx(0.04)
    assert thresholds.embedding_only_score_ceiling == pytest.approx(0.45)
    assert thresholds.hop_implausible_penalty == pytest.approx(0.5)


def test_settings__custom_yaml_path__reads_that_file(
    custom_thresholds_file: Path,
) -> None:
    """Pointing thresholds_file at another YAML loads that file's values."""
    settings = Settings(_env_file=None, thresholds_file=custom_thresholds_file)

    assert settings.thresholds.plate_fuzzy_max_edit_distance == 1
    assert settings.thresholds.plate_auto_accept_min_confidence == pytest.approx(0.70)


def test_settings__env_override_of_a_threshold__beats_the_yaml_file(
    monkeypatch: pytest.MonkeyPatch, build_settings: Callable[..., Settings]
) -> None:
    """Environment variables sit above the YAML source in the precedence order."""
    monkeypatch.setenv("MCT_THRESHOLDS__PLATE_FUZZY_MAX_EDIT_DISTANCE", "1")

    thresholds = build_settings().thresholds

    assert thresholds.plate_fuzzy_max_edit_distance == 1
    # Values the environment did not mention still come from the YAML file.
    assert thresholds.plate_auto_accept_min_confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_settings__missing_secret_key_with_auth_enabled__raises_with_field_name(
    build_settings: Callable[..., Settings],
) -> None:
    """Enabling auth without a secret names the offending field in the error."""
    with pytest.raises(ConfigurationError) as excinfo:
        build_settings(api={"auth_enabled": True})

    assert excinfo.value.context["field"] == "api.secret_key"
    assert "api.secret_key" in str(excinfo.value)


def test_settings__invalid_port_type__raises_validation_error_without_coercing(
    build_settings: Callable[..., Settings],
) -> None:
    """A non-numeric port fails loudly instead of silently becoming a default."""
    with pytest.raises(PydanticValidationError) as excinfo:
        build_settings(database={"port": "abc"})

    assert any(error["loc"] == ("database", "port") for error in excinfo.value.errors())


def test_settings__port_out_of_range__is_rejected(
    build_settings: Callable[..., Settings],
) -> None:
    """Port bounds are enforced at the settings boundary, not at connect time."""
    with pytest.raises(PydanticValidationError):
        build_settings(database={"port": 70000})


def test_settings__missing_thresholds_file__raises_configuration_error(
    tmp_path: Path,
) -> None:
    """A missing thresholds file is a startup failure with an actionable message."""
    missing = tmp_path / "nope" / "thresholds.yaml"

    with pytest.raises(ConfigurationError) as excinfo:
        Settings(_env_file=None, thresholds_file=missing)

    # Assert on the structured context, not on str(): the rendered form uses
    # repr(), which escapes the backslashes in a Windows path.
    assert excinfo.value.context["field"] == "thresholds_file"
    assert excinfo.value.context["path"] == str(missing)
    assert "MCT_THRESHOLDS_FILE" in excinfo.value.message


def test_settings__thresholds_file_is_not_a_mapping__raises_configuration_error(
    tmp_path: Path,
) -> None:
    """A YAML list where a mapping is expected fails with the parsed type named."""
    path = tmp_path / "thresholds.yaml"
    path.write_text("- 0.85\n- 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        Settings(_env_file=None, thresholds_file=path)

    assert excinfo.value.context["parsed_type"] == "list"


def test_settings__malformed_yaml__raises_configuration_error(tmp_path: Path) -> None:
    """A YAML syntax error is reported as a configuration failure, not a traceback."""
    path = tmp_path / "thresholds.yaml"
    path.write_text("plate_auto_accept_min_confidence: [0.85\n", encoding="utf-8")

    with pytest.raises(ConfigurationError) as excinfo:
        Settings(_env_file=None, thresholds_file=path)

    assert excinfo.value.message == "Thresholds file could not be read"
    assert "reason" in excinfo.value.context


def test_settings__empty_thresholds_file__reports_missing_thresholds(tmp_path: Path) -> None:
    """Boundary: an empty file parses to nothing and every threshold is missing."""
    path = tmp_path / "thresholds.yaml"
    path.write_text("", encoding="utf-8")

    with pytest.raises(PydanticValidationError) as excinfo:
        Settings(_env_file=None, thresholds_file=path)

    # Every threshold, not a sample: the point of the required-field design is
    # that a value omitted from the YAML cannot silently fall back to a default,
    # so the error has to name all of them.
    missing = {error["loc"][-1] for error in excinfo.value.errors()}
    assert missing == {
        "plate_auto_accept_min_confidence",
        "plate_fuzzy_max_edit_distance",
        "plate_fuzzy_max_weighted_distance",
        "plate_max_length_delta",
        "plate_confusion_substitution_cost",
        "plate_exact_method_weight",
        "plate_fuzzy_method_weight",
        "plate_distance_penalty_per_unit",
        "plate_review_min_confidence",
        "embedding_auto_accept_min_similarity",
        "embedding_review_min_similarity",
        "embedding_margin_min",
        "embedding_only_score_ceiling",
        "embedding_agreement_boost",
        "embedding_disagreement_similarity",
        "embedding_max_references",
        "hop_implausible_penalty",
        "path_node_inclusion_bonus",
        "path_gap_edge_penalty",
        "path_ambiguity_margin_min",
        "path_weakest_link_tolerance",
        "detection_min_bbox_area_px",
        "detection_max_bbox_area_fraction",
        "detection_min_aspect_ratio",
        "detection_max_aspect_ratio",
        "detection_roi_min_overlap",
        "track_association_min_iou",
        "track_max_age_frames",
        "track_min_hits_to_confirm",
        "best_frame_confidence_weight",
        "best_frame_sharpness_weight",
        "best_frame_area_weight",
        "best_frame_centrality_weight",
        "best_frame_edge_penalty",
    }


def test_settings__thresholds_file_env_var__selects_the_file(
    monkeypatch: pytest.MonkeyPatch, custom_thresholds_file: Path
) -> None:
    """MCT_THRESHOLDS_FILE redirects the YAML source without a constructor argument."""
    monkeypatch.setenv("MCT_THRESHOLDS_FILE", str(custom_thresholds_file))

    settings = Settings(_env_file=None)

    assert settings.thresholds_file == custom_thresholds_file
    assert settings.thresholds.plate_fuzzy_max_edit_distance == 1


def test_thresholds_yaml_source__get_field_value__reports_no_value(
    thresholds_file: Path,
) -> None:
    """The source populates a whole section; it answers no per-field lookups."""
    source = ThresholdsYamlSource(Settings, thresholds_file)

    value, key, is_complex = source.get_field_value(FieldInfo(), "anything")

    assert (value, key, is_complex) == (None, "anything", False)


def test_thresholds_yaml_source__repr__names_the_path(thresholds_file: Path) -> None:
    """Failure output should say which file the source was reading."""
    source = ThresholdsYamlSource(Settings, thresholds_file)

    assert repr(source) == f"ThresholdsYamlSource(path={str(thresholds_file)!r})"


def test_settings__unknown_key_in_thresholds_yaml__is_rejected(tmp_path: Path) -> None:
    """A typo in the thresholds file is an error, not a silently ignored key."""
    path = tmp_path / "thresholds.yaml"
    path.write_text(
        "plate_auto_accept_min_confidence: 0.85\n"
        "plate_fuzzy_max_edit_distance: 2\n"
        "embedding_auto_accept_min_similarity: 0.92\n"
        "embedding_review_min_similarity: 0.75\n"
        "hop_implausible_penalty: 0.5\n"
        "plate_auto_acept_min_confidence: 0.99\n",
        encoding="utf-8",
    )

    with pytest.raises(PydanticValidationError) as excinfo:
        Settings(_env_file=None, thresholds_file=path)

    assert any(error["type"] == "extra_forbidden" for error in excinfo.value.errors())


def test_settings__incomplete_thresholds_yaml__reports_the_missing_threshold(
    tmp_path: Path,
) -> None:
    """Thresholds have no Python defaults, so an omission cannot pass unnoticed."""
    path = tmp_path / "thresholds.yaml"
    path.write_text("plate_auto_accept_min_confidence: 0.85\n", encoding="utf-8")

    with pytest.raises(PydanticValidationError) as excinfo:
        Settings(_env_file=None, thresholds_file=path)

    missing = {error["loc"][-1] for error in excinfo.value.errors()}
    assert "plate_fuzzy_max_edit_distance" in missing


# ---------------------------------------------------------------------------
# Cross-field invariants
# ---------------------------------------------------------------------------


def threshold_kwargs(**overrides: object) -> dict[str, object]:
    """Return a complete, valid set of threshold values with overrides applied.

    Every ThresholdSettings field is required, so constructing one directly
    means naming all of them. Enumerating that list inside each test made every
    new threshold a change to every cross-field test, which is churn that hides
    the one value the test actually cares about.

    Args:
        **overrides: Values to replace.

    Returns:
        Keyword arguments for ThresholdSettings.
    """
    values: dict[str, object] = {
        "plate_auto_accept_min_confidence": 0.85,
        "plate_fuzzy_max_edit_distance": 2,
        "plate_fuzzy_max_weighted_distance": 0.9,
        "plate_max_length_delta": 2,
        "plate_confusion_substitution_cost": 0.5,
        "plate_exact_method_weight": 1.0,
        "plate_fuzzy_method_weight": 0.9,
        "plate_distance_penalty_per_unit": 0.12,
        "plate_review_min_confidence": 0.5,
        "embedding_auto_accept_min_similarity": 0.95,
        "embedding_review_min_similarity": 0.75,
        "embedding_margin_min": 0.04,
        "embedding_only_score_ceiling": 0.45,
        "embedding_agreement_boost": 0.05,
        "embedding_disagreement_similarity": 0.4,
        "embedding_max_references": 8,
        "hop_implausible_penalty": 0.5,
        "path_node_inclusion_bonus": 0.0,
        "path_gap_edge_penalty": 0.05,
        "path_ambiguity_margin_min": 0.05,
        "path_weakest_link_tolerance": 0.05,
        "detection_min_bbox_area_px": 400,
        "detection_max_bbox_area_fraction": 0.5,
        "detection_min_aspect_ratio": 0.25,
        "detection_max_aspect_ratio": 5.0,
        "detection_roi_min_overlap": 0.5,
        "track_association_min_iou": 0.3,
        "track_max_age_frames": 5,
        "track_min_hits_to_confirm": 3,
        "best_frame_confidence_weight": 0.35,
        "best_frame_sharpness_weight": 0.30,
        "best_frame_area_weight": 0.20,
        "best_frame_centrality_weight": 0.15,
        "best_frame_edge_penalty": 0.5,
    }
    values.update(overrides)
    return values


def test_threshold_settings__committed_defaults__construct_cleanly() -> None:
    """The helper itself must describe a valid configuration, or it proves nothing."""
    assert ThresholdSettings(**threshold_kwargs()).plate_review_min_confidence == pytest.approx(0.5)


def test_threshold_settings__inverted_review_band__is_rejected() -> None:
    """An auto-accept below the review floor would leave no review band at all."""
    with pytest.raises(PydanticValidationError, match="review band"):
        ThresholdSettings(
            **threshold_kwargs(
                embedding_auto_accept_min_similarity=0.60,
                embedding_review_min_similarity=0.75,
            )
        )


def test_threshold_settings__visual_ceiling_reaching_plate_scores__is_rejected() -> None:
    """The evidence hierarchy is one inequality, so it is checked at startup.

    A ceiling at or above the weakest retained plate score would let a visual
    guess be presented with the authority of a plate read.
    """
    with pytest.raises(PydanticValidationError, match="outrank a plate match"):
        ThresholdSettings(**threshold_kwargs(embedding_only_score_ceiling=0.5))


def test_threshold_settings__inverted_aspect_window__is_rejected() -> None:
    """A minimum above the maximum rejects every detection, silently seeing nothing."""
    with pytest.raises(PydanticValidationError, match="inverted window"):
        ThresholdSettings(
            **threshold_kwargs(
                detection_min_aspect_ratio=5.0,
                detection_max_aspect_ratio=0.25,
            )
        )


def test_threshold_settings__aspect_bounds_exactly_equal__are_accepted() -> None:
    """Boundary: a degenerate but non-inverted window is odd, not invalid."""
    settings = ThresholdSettings(
        **threshold_kwargs(detection_min_aspect_ratio=2.0, detection_max_aspect_ratio=2.0)
    )

    assert settings.detection_min_aspect_ratio == pytest.approx(2.0)


def test_threshold_settings__best_frame_weights_not_summing_to_one__are_rejected() -> None:
    """Weights summing past one put the composite score outside [0, 1]."""
    with pytest.raises(PydanticValidationError, match=r"must sum to 1.0"):
        ThresholdSettings(**threshold_kwargs(best_frame_confidence_weight=0.5))


def test_threshold_settings__committed_values__satisfy_the_evidence_hierarchy(
    build_settings: Callable[..., Settings],
) -> None:
    """The shipped configuration must itself obey the rule it enforces."""
    thresholds = build_settings().thresholds

    assert thresholds.embedding_only_score_ceiling < (
        thresholds.plate_review_min_confidence * thresholds.plate_exact_method_weight
    )


def test_retention_settings__thumbnails_outliving_sightings__is_rejected() -> None:
    """A longer thumbnail TTL would leave orphaned imagery after a purge."""
    with pytest.raises(PydanticValidationError, match="orphaned thumbnails"):
        RetentionSettings(sighting_ttl_days=30, thumbnail_ttl_days=90)


def test_settings__production_without_auth__is_rejected(
    build_settings: Callable[..., Settings],
) -> None:
    """Production may not run with authentication disabled."""
    with pytest.raises(ConfigurationError) as excinfo:
        build_settings(environment="production", api={"auth_enabled": False})

    assert excinfo.value.context["field"] == "api.auth_enabled"


def test_settings__production_with_debug__is_rejected(
    build_settings: Callable[..., Settings],
) -> None:
    """Production may not run with debug output enabled."""
    with pytest.raises(ConfigurationError) as excinfo:
        build_settings(
            environment="production",
            debug=True,
            api={"auth_enabled": True, "secret_key": "s3cret"},
        )

    assert excinfo.value.context["field"] == "debug"


def test_settings__production_fully_configured__is_accepted(
    build_settings: Callable[..., Settings],
) -> None:
    """The valid production combination is not blocked by the guardrails."""
    settings = build_settings(
        environment="production",
        api={"auth_enabled": True, "secret_key": "s3cret"},
    )

    assert settings.api.has_secret_key is True
    assert settings.log_as_json is True


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------


def test_database_settings__dsn__includes_credentials_and_encodes_them() -> None:
    """Special characters in a password must not break the connection URL."""
    database = DatabaseSettings(
        host="db.internal", port=6543, user="track@svc", password=SecretStr("p@ss:w/rd")
    )

    assert database.dsn == (
        "postgresql+psycopg://track%40svc:p%40ss%3Aw%2Frd@db.internal:6543/multicam"
    )


def test_database_settings__safe_dsn__masks_the_password() -> None:
    """The loggable DSN never contains the secret."""
    database = DatabaseSettings(password=SecretStr("hunter2"))

    assert "hunter2" not in database.safe_dsn
    assert database.safe_dsn.startswith("postgresql+psycopg://postgres:***@")


def test_database_settings__no_password__omits_the_credential_separator() -> None:
    """An empty password produces a URL with no dangling colon."""
    database = DatabaseSettings(password=SecretStr(""))

    assert database.dsn == "postgresql+psycopg://postgres@localhost:5432/multicam"


def test_settings__debug_enabled__selects_console_rendering(
    build_settings: Callable[..., Settings],
) -> None:
    """Debug mode implies human-readable logs; everything else implies JSON."""
    assert build_settings(debug=True).log_as_json is False
    assert build_settings(debug=False).log_as_json is True


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_get_settings__called_twice__returns_the_same_instance() -> None:
    """Settings are built once per process; callers share one object."""
    first = get_settings()
    second = get_settings()

    assert first is second


def test_get_settings__after_cache_reset__rebuilds_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resetting the cache is what lets a test change the environment."""
    first = get_settings()
    assert first.log_level == "INFO"

    monkeypatch.setenv("MCT_LOG_LEVEL", "ERROR")
    reset_settings_cache()
    second = get_settings()

    assert second is not first
    assert second.log_level == "ERROR"


def test_get_settings__invalid_environment_value__raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic's error is translated so callers only catch the project hierarchy."""
    monkeypatch.setenv("MCT_DATABASE__PORT", "abc")
    reset_settings_cache()

    with pytest.raises(ConfigurationError) as excinfo:
        get_settings()

    assert "database.port" in excinfo.value.context["fields"]
