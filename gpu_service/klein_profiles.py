"""Shared, measured sampling profiles for the app and direct service clients."""


def steps_for(kind: str, quality: str) -> int:
    return 4 if kind == "lower" or quality == "detail" else 3
