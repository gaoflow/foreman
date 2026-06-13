"""Cross-platform launcher for ``lint-imports``.

The justfile recipe ``just import-linter`` needs ``PYTHONPATH`` set to
``packages/foreman`` so import-linter / grimp can resolve the
``tests`` root package referenced in
``[tool.importlinter].root_packages`` in workspace-root
``pyproject.toml``. The shell-prefix form
``PYTHONPATH=packages/foreman uv run lint-imports`` works on POSIX
but breaks on Windows cmd.exe (the recipe's runner per
``set windows-shell := ["cmd.exe", "/c"]`` in the justfile). This
helper sets the env var via :mod:`os` and execs ``lint-imports``
on either platform.

Added during the foreman#307 LabelManager work, 2026-06-13.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = "packages/foreman"
    return subprocess.call(["lint-imports", *sys.argv[1:]], env=env)


if __name__ == "__main__":
    sys.exit(main())
