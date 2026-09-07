"""Run the ``carbon_intensity`` in-package checks under the single test command.

Those modules live next to the code they check and stay runnable on their own
(``python -m carbon_intensity.check_baselines``) for focused debugging. This
wrapper is what makes ``unittest discover -s tests`` the one command that runs
everything, instead of six extra invocations nobody remembers.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from carbon_intensity import (
    check_baselines,
    check_forecasting,
    check_history,
    check_scheduling_impact,
    check_snapshots,
    check_walkforward,
)


MODULES = (
    check_history,
    check_baselines,
    check_forecasting,
    check_snapshots,
    check_walkforward,
    check_scheduling_impact,
)


class InPackageChecks(unittest.TestCase):
    """Each module asserts its own invariants and prints a summary line."""

    def test_in_package_checks_pass(self) -> None:
        for module in MODULES:
            with self.subTest(module=module.__name__):
                # The checks report by printing; keep the test output readable.
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
