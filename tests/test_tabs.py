from __future__ import annotations

import ast
import re
from pathlib import Path

from ml_workbench.tabs import TAB_ORDER, TABS

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DOC = REPO_ROOT / "docs" / "contracts.md"


def _documented_tab_order() -> tuple[str, ...]:
    text = CONTRACTS_DOC.read_text()
    match = re.search(r"TAB_ORDER = (\[[^\]]+\])", text, re.S)
    assert match is not None, "TAB_ORDER literal not found in docs/contracts.md"
    literal = ast.literal_eval(match.group(1))
    return tuple(literal)


def test_tab_order_matches_contract() -> None:
    assert TAB_ORDER == _documented_tab_order()


def test_tab_metadata_is_sequential() -> None:
    assert len(TABS) == 10
    assert [tab.number for tab in TABS] == list(range(1, 11))
    assert [tab.design_section for tab in TABS] == [
        "6.1",
        "6.2",
        "6.3",
        "6.4",
        "6.5",
        "6.6",
        "6.7",
        "6.8",
        "6.9",
        "6.10",
    ]


def test_tab_phases_match_plan() -> None:
    phases = {tab.id: tab.implemented_phase for tab in TABS}
    assert phases["data"] == 2
    assert phases["cleaning"] == 2
    assert phases["preprocessing"] == 3
    assert phases["eda"] == 3
    assert phases["modelling"] == 4
    assert phases["training"] == 4
    assert phases["prediction"] == 4
    assert phases["error_analysis"] == 5
    assert phases["explainability"] == 5
    assert phases["outcome"] == 5
