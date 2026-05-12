"""Allow running doctor as ``python -m project_code_intelligence.doctor``."""

from __future__ import annotations

import sys

from project_code_intelligence.doctor.cli import main

raise SystemExit(main(sys.argv[1:]))
