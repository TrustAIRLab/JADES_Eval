"""Persistence-domain validation; never clips scores or changes the formula."""
import math


class NonFiniteOutputError(ValueError):
    code = "non_finite_output"


def require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteOutputError("Non-finite numerical output cannot be persisted")
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item)
