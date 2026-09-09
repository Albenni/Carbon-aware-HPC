"""Run the script-style ``carbon_intensity`` checks under the single test command.

Each ``check_carbon_intensity_*`` module asserts its own invariants in a
``main()`` and stays runnable on its own (``python
tests/check_carbon_intensity_history.py``) for focused debugging. They carry no
``TestCase``, so this wrapper is what makes ``unittest discover -s tests`` the
one command that runs everything, instead of seven extra invocations nobody
remembers.
"""

from __future__ import annotations

import contextlib
import io
import unittest

import check_carbon_intensity_baselines
import check_carbon_intensity_boosted
import check_carbon_intensity_forecasting
import check_carbon_intensity_history
import check_carbon_intensity_scheduling_impact
import check_carbon_intensity_snapshots
import check_carbon_intensity_walkforward


# Ordered the way the pipeline is built: data, then baselines, then models.
MODULES = (
    check_carbon_intensity_history,
    check_carbon_intensity_baselines,
    check_carbon_intensity_forecasting,
    check_carbon_intensity_snapshots,
    check_carbon_intensity_walkforward,
    check_carbon_intensity_boosted,
    check_carbon_intensity_scheduling_impact,
)


class ScriptStyleChecks(unittest.TestCase):
    """Each module asserts its own invariants and prints a summary line."""

    def test_script_style_checks_pass(self) -> None:
        for module in MODULES:
            with self.subTest(module=module.__name__):
                # The checks report by printing; keep the test output readable.
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
