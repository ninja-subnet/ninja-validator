"""Entrypoint for the DB-backed public dashboard API."""

from __future__ import annotations

from .public import serve


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
