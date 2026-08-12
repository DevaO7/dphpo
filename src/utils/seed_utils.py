import numpy as np
import random
import torch


DP_NOISE_STREAM = 1


def set_global_seed(seed: int = 42) -> None:
    """Seed process-wide RNGs and request deterministic cuDNN behavior."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "The global seed must be a non-negative integer; "
            f"got {seed!r}."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These flags improve cuDNN reproducibility. They do not make every
    # PyTorch operation deterministic; individual algorithms and data-loader
    # workers may still require their own deterministic configuration.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
