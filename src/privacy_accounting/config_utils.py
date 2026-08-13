"""Shared configuration parsing for privacy-accounting mechanisms."""

from __future__ import annotations

from typing import Any, Mapping


def extract_mapping_value(
    mapping: Mapping[str, Any],
    *,
    canonical_name: str,
    aliases: tuple[str, ...],
) -> Any:
    """Extract one value while supporting legacy configuration aliases.

    If multiple accepted keys are present, they must contain equal values.
    """
    present = [
        key
        for key in (canonical_name, *aliases)
        if key in mapping
    ]

    if not present:
        accepted = ", ".join((canonical_name, *aliases))
        raise KeyError(
            f"Missing configuration value for {canonical_name!r}. "
            f"Accepted keys: {accepted}."
        )

    value = mapping[present[0]]
    for key in present[1:]:
        other_value = mapping[key]
        if other_value != value:
            raise ValueError(
                f"Conflicting values were supplied for {canonical_name!r}: "
                f"{present[0]}={value!r}, {key}={other_value!r}."
            )

    return value
