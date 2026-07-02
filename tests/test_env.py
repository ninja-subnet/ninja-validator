"""Env parsing helpers — notably tolerance of inline comments.

`docker compose`'s `env_file` keeps everything after `=` as the value, so a line like
`MAX_CONTAINERS=10  # note` arrives as the string `"10  # note"`. The numeric/bool/str
parsers must ignore a whitespace-preceded `#` comment rather than silently falling back
to their default (the bug that pinned MAX_CONTAINERS to 4 regardless of .env).
"""

from __future__ import annotations

from tau.utils.env import env_bool, env_float, env_int, env_str


def test_env_int_tolerates_inline_comment() -> None:
    assert env_int({"X": "10                 # sandboxes per loop tick"}, "X", 4) == 10
    assert env_int({"X": "  7  "}, "X", 4) == 7
    assert env_int({}, "X", 4) == 4  # missing -> default
    assert env_int({"X": "notanint"}, "X", 4) == 4  # unparseable -> default


def test_env_float_tolerates_inline_comment() -> None:
    assert env_float({"X": "2.5  # max cost usd"}, "X", 1.0) == 2.5
    assert env_float({}, "X", 1.0) == 1.0


def test_env_bool_tolerates_inline_comment() -> None:
    assert env_bool({"X": "1  # enabled"}, "X", False) is True
    assert env_bool({"X": "false # off"}, "X", True) is False


def test_env_str_strips_comment_but_keeps_embedded_hash() -> None:
    # A whitespace-preceded '#' is a comment...
    assert env_str({"M": "deepseek/v4  # model"}, "M", "d") == "deepseek/v4"
    # ...but a '#' inside the value (no preceding space) is preserved (e.g. passwords).
    assert env_str({"P": "p#ss"}, "P", "d") == "p#ss"
    assert env_str({}, "M", "default") == "default"
