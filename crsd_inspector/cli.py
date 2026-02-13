"""CRSD Inspector CLI wrapper over renderflow."""

from __future__ import annotations

import sys
from typing import Sequence

DEFAULT_PROVIDER = "crsd-inspector"


def _inject_provider(args: list[str]) -> list[str]:
    if not args:
        return ["list-workflows", "--provider", DEFAULT_PROVIDER]

    if args[0] == "list":
        return ["list-workflows", "--provider", DEFAULT_PROVIDER]

    known_provider_scoped = {"run", "list-workflows", "show-params", "execute"}
    if args[0] in known_provider_scoped and "--provider" not in args and "--target-package" not in args:
        return [args[0], "--provider", DEFAULT_PROVIDER] + args[1:]

    return args


def main(argv: Sequence[str] | None = None):
    args = list(argv) if argv is not None else sys.argv[1:]
    forwarded = _inject_provider(args)

    from renderflow.cli import main as renderflow_main

    return renderflow_main(forwarded)


if __name__ == "__main__":
    main()
