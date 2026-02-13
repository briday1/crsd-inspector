"""CRSD Inspector CLI powered by renderflow provider-scoped CLI."""

from __future__ import annotations

import sys
from typing import Sequence

from renderflow.cli import provider_main


DEFAULT_PROVIDER = "crsd-inspector"


def main(argv: Sequence[str] | None = None):
    return provider_main(
        provider_name=DEFAULT_PROVIDER,
        argv=argv if argv is not None else sys.argv[1:],
        prog_name="crsd-inspector",
        description="CRSD Inspector CLI",
    )


if __name__ == "__main__":
    main()
