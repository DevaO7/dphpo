import numpy as np


DP_NOISE_STREAM = 1


def derive_seed(
    base_seed: int,
    *coordinates: int,
    stream: int,
) -> int:
    """Derive a deterministic 64-bit seed for one independent RNG stream."""
    components = (base_seed, stream, *coordinates)
    if any(
        isinstance(component, bool)
        or not isinstance(component, int)
        or component < 0
        for component in components
    ):
        raise ValueError(
            "Seed components must be non-negative integers; "
            f"got {components!r}."
        )

    seed_sequence = np.random.SeedSequence(components)
    return int(
        seed_sequence.generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
