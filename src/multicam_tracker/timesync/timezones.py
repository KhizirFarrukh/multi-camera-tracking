"""Local time, and why the system refuses to guess at it.

Two hours a year are genuinely ambiguous in every zone that observes daylight
saving. When the clocks go back, 01:30 happens twice; when they go forward,
02:30 does not happen at all. A library asked to interpret those instants will
usually pick one silently -- ``fold=0``, or the post-transition offset -- and be
wrong half the time.

Half the time is far too often here. A one-hour error in a camera's timestamps
is not a blurred reading; it puts a vehicle at a camera an hour before or after
it was really there, and every travel-time window it is then compared against
gives a confident wrong answer. So both cases raise, naming the instant and the
camera, and an operator resolves them by stating the UTC offset explicitly.

**Leap seconds.** UTC leap seconds are not represented. Python's ``datetime``
has no 23:59:60, the POSIX clock does not tick it, and every timestamp this
system ingests comes from a device that has already smeared or repeated the
second. Within one second the system's conclusions do not change: the smallest
travel-time window in a realistic topology is tens of seconds. The policy is
therefore explicit rather than accidental -- leap seconds are absorbed by
whatever the source did with them, and never corrected for here.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from multicam_tracker.exceptions import ValidationError

__all__ = ["resolve_local_time", "zone_for"]


def zone_for(timezone_name: str) -> ZoneInfo:
    """Return the IANA zone for a name.

    Args:
        timezone_name: An IANA identifier such as ``Europe/London``.

    Returns:
        The zone.

    Raises:
        ValidationError: If the name is unknown. An unknown zone is a
            configuration error, and defaulting to UTC would silently shift
            every timestamp from that camera.
    """
    try:
        return ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValidationError(
            "Unknown timezone; expected an IANA identifier such as 'Europe/London'",
            {"timezone": timezone_name, "error": str(exc)},
        ) from exc


def _is_ambiguous(local: datetime, zone: ZoneInfo) -> bool:
    """Return whether a local time occurs twice in a zone.

    Args:
        local: A naive local datetime.
        zone: The zone to interpret it in.

    Returns:
        ``True`` when the two folds have different UTC offsets, which is exactly
        the repeated hour at a fall-back transition.
    """
    first = local.replace(tzinfo=zone, fold=0)
    second = local.replace(tzinfo=zone, fold=1)
    return first.utcoffset() != second.utcoffset()


def _is_non_existent(local: datetime, zone: ZoneInfo) -> bool:
    """Return whether a local time never occurs in a zone.

    Args:
        local: A naive local datetime.
        zone: The zone to interpret it in.

    Returns:
        ``True`` when converting to UTC and back does not round-trip, which is
        the signature of the skipped hour at a spring-forward transition.
    """
    attached = local.replace(tzinfo=zone)
    round_tripped = attached.astimezone(ZoneInfo("UTC")).astimezone(zone)
    return round_tripped.replace(tzinfo=None) != local


def resolve_local_time(
    local: datetime,
    timezone_name: str,
    *,
    camera_id: str | None = None,
    utc_offset: timedelta | None = None,
) -> datetime:
    """Convert a camera's local timestamp to UTC, refusing to guess.

    Args:
        local: The naive local timestamp as the camera reported it. An already
            aware datetime is converted directly, since it carries no ambiguity.
        timezone_name: IANA zone the camera reports its time in.
        camera_id: Named in errors, so an operator knows which device to fix.
        utc_offset: The explicit offset to use, which resolves an otherwise
            ambiguous instant. This is how an operator answers the question the
            error asks: "which of the two 01:30s was it?".

    Returns:
        The instant in UTC.

    Raises:
        ValidationError: If the local time is ambiguous or non-existent in the
            zone and no explicit offset was supplied, or if the supplied offset
            is not one the zone actually uses at that instant.
    """
    zone = zone_for(timezone_name)

    if local.tzinfo is not None:
        return local.astimezone(ZoneInfo("UTC"))

    if utc_offset is not None:
        return _resolve_with_offset(local, zone, utc_offset, timezone_name, camera_id)

    context = {
        "camera_id": camera_id,
        "local_time": local.isoformat(),
        "timezone": timezone_name,
    }

    if _is_non_existent(local, zone):
        raise ValidationError(
            f"Local time {local.isoformat()} does not exist in {timezone_name}: the clocks "
            f"jumped forward across it. The camera's clock is wrong, or its configured "
            f"timezone is. Supply an explicit utc_offset to state what was meant",
            context,
        )

    if _is_ambiguous(local, zone):
        first = local.replace(tzinfo=zone, fold=0)
        second = local.replace(tzinfo=zone, fold=1)
        raise ValidationError(
            f"Local time {local.isoformat()} occurs twice in {timezone_name}: the clocks "
            f"went back across it, so it could be {first.isoformat()} or "
            f"{second.isoformat()}. Supply an explicit utc_offset to say which",
            {
                **context,
                "candidates": [first.isoformat(), second.isoformat()],
            },
        )

    return local.replace(tzinfo=zone).astimezone(ZoneInfo("UTC"))


def _resolve_with_offset(
    local: datetime,
    zone: ZoneInfo,
    utc_offset: timedelta,
    timezone_name: str,
    camera_id: str | None,
) -> datetime:
    """Interpret a local time using an operator-supplied UTC offset.

    Args:
        local: The naive local timestamp.
        zone: The camera's zone.
        utc_offset: The offset the operator states applied.
        timezone_name: Zone name, for error context.
        camera_id: Camera, for error context.

    Returns:
        The instant in UTC.

    Raises:
        ValidationError: If the zone uses neither fold's offset at that instant.
            An arbitrary offset would be a second guess dressed as a decision.
    """
    candidates = {local.replace(tzinfo=zone, fold=fold).utcoffset() for fold in (0, 1)}
    if utc_offset not in candidates:
        raise ValidationError(
            f"UTC offset {utc_offset} is not one {timezone_name} uses at "
            f"{local.isoformat()}; it uses "
            f"{', '.join(str(candidate) for candidate in sorted(candidates, key=str))}",
            {
                "camera_id": camera_id,
                "local_time": local.isoformat(),
                "timezone": timezone_name,
                "supplied_offset": str(utc_offset),
            },
        )
    return (local - utc_offset).replace(tzinfo=ZoneInfo("UTC"))
