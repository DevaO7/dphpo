from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TwoStageSettings:
    """Configuration values that define the two-stage HPO mechanism."""

    num_survivors: int
    stage_2_expected_trials: float


def get_two_stage_settings(config) -> TwoStageSettings:
    """Read and validate the distinct Stage-1 and Stage-2 counts."""
    try:
        raw_num_survivors = config.two_stage.num_survivors
        raw_stage_2_expected_trials = (
            config.two_stage.stage_2_expected_trials
        )
    except (AttributeError, KeyError) as error:
        raise ValueError(
            "The experiment configuration must define "
            "two_stage.num_survivors and "
            "two_stage.stage_2_expected_trials."
        ) from error

    if isinstance(raw_num_survivors, bool):
        raise ValueError(
            "two_stage.num_survivors must be a positive integer; "
            f"got {raw_num_survivors!r}."
        )
    try:
        num_survivors = int(raw_num_survivors)
        num_survivors_as_float = float(raw_num_survivors)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "two_stage.num_survivors must be a positive integer; "
            f"got {raw_num_survivors!r}."
        ) from error
    if (
        not math.isfinite(num_survivors_as_float)
        or num_survivors_as_float != num_survivors
        or num_survivors < 1
    ):
        raise ValueError(
            "two_stage.num_survivors must be a positive integer; "
            f"got {raw_num_survivors!r}."
        )

    if isinstance(raw_stage_2_expected_trials, bool):
        raise ValueError(
            "two_stage.stage_2_expected_trials must be finite and "
            "greater than 1; got "
            f"{raw_stage_2_expected_trials!r}."
        )
    try:
        stage_2_expected_trials = float(
            raw_stage_2_expected_trials
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "two_stage.stage_2_expected_trials must be finite and "
            "greater than 1; got "
            f"{raw_stage_2_expected_trials!r}."
        ) from error
    if (
        not math.isfinite(stage_2_expected_trials)
        or stage_2_expected_trials <= 1
    ):
        raise ValueError(
            "two_stage.stage_2_expected_trials must be finite and "
            "greater than 1; got "
            f"{raw_stage_2_expected_trials!r}."
        )

    return TwoStageSettings(
        num_survivors=num_survivors,
        stage_2_expected_trials=stage_2_expected_trials,
    )
