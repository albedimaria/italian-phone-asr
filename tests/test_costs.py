"""COSTS.md: the declared total equals the sum of the rows and stays under the cap."""

import re

from iphasr.paths import REPO

CAP = 25.0


def test_costs_sum_and_cap():
    text = (REPO / "COSTS.md").read_text(encoding="utf-8")
    rows = [l for l in text.splitlines() if re.match(r"\|\s*\d{4}-\d{2}-\d{2}", l)]
    assert rows, "COSTS.md has no rows"
    total = sum(float(l.split("|")[6].strip()) for l in rows)
    declared = float(re.search(r"Total out-of-pocket: \$([\d.]+)", text).group(1))
    assert abs(total - declared) < 0.005, f"rows sum to {total:.2f}, declared {declared:.2f}"
    assert declared <= CAP
