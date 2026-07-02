from collections.abc import Mapping

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _scalar(raw: str) -> str:
    """Normalize a raw env value: drop a trailing inline comment, then whitespace.

    ``docker compose``'s ``env_file`` keeps everything after ``=`` as the value, so a
    line like ``MAX_CONTAINERS=10  # note`` arrives as ``"10  # note"``. Numeric/bool
    parsers would otherwise raise and silently fall back to their default. A ``#`` is
    only treated as a comment when preceded by whitespace (or at the start), so values
    that legitimately contain ``#`` (e.g. ``a#b``) are left intact.
    """
    for i, ch in enumerate(raw):
        if ch == "#" and (i == 0 or raw[i - 1] in " \t"):
            return raw[:i].strip()
    return raw.strip()


def env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(_scalar(raw))
    except ValueError:
        return default


def env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return float(_scalar(raw))
    except ValueError:
        return default


def env_str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    value = _scalar(raw) if raw is not None else ""
    return value if value else default


def env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Parse a boolean env var. True: 1/true/yes/on; False: 0/false/no/off (any case)."""
    value = env.get(name)
    if value is None or not _scalar(value):
        return default
    normalized = _scalar(value).lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return default


def env_required(env: Mapping[str, str], name: str) -> str:
    """Return ``env[name]`` (stripped); raise if it is unset or blank."""
    value = env.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def env_int_strict(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def env_float_strict(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def env_bool_strict(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {value!r}")
