from __future__ import annotations

import ast
import re
from pathlib import Path

from ml_workbench.rules import IMPLEMENTED_RULES

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_DOC = REPO_ROOT / "docs" / "rules.md"

SECTION_RE = re.compile(r"^## \d+\. (.+)$", re.M)
ROW_RE = re.compile(r"^\|\s*([A-Z]+-[0-9]+[a-c]?)(?:\s*⚠)?\s*\|(.*)\|\s*$", re.M)
TEST_NAME_RE = re.compile(r"`(test_\w+)`")


def _documented_rules() -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for match in ROW_RE.finditer(RULES_DOC.read_text()):
        name_match = TEST_NAME_RE.search(match.group(2))
        if name_match:
            rules.append((match.group(1), name_match.group(1)))
    return rules


def _defined_tests() -> dict[str, ast.FunctionDef]:
    defined: dict[str, ast.FunctionDef] = {}
    for path in (REPO_ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                defined[node.name] = node
    return defined


def _is_skipped(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        rendered = ast.unparse(decorator)
        if "pytest.mark.skip" in rendered:
            return True
    return False


def test_docs_rules_parse() -> None:
    rules = _documented_rules()
    assert len(rules) >= 120
    ids = [rule_id for rule_id, _name in rules]
    names = [name for _rule_id, name in rules]
    assert len(ids) == len(set(ids)), "duplicate rule ids in docs/rules.md"
    assert len(names) == len(set(names)), "duplicate test names in docs/rules.md"


def test_every_rule_has_a_test_function() -> None:
    defined = _defined_tests()
    missing = [name for _rule_id, name in _documented_rules() if name not in defined]
    assert not missing, f"rules without a test function: {sorted(missing)}"


def test_implemented_rules_are_documented() -> None:
    documented = {rule_id for rule_id, _name in _documented_rules()}
    undocumenteded = IMPLEMENTED_RULES - documented
    assert not undocumenteded, (
        f"engine implements rules missing from docs: {sorted(undocumenteded)}"
    )


def test_implemented_rules_are_not_skipped() -> None:
    name_by_id = dict(_documented_rules())
    defined = _defined_tests()
    for rule_id in sorted(IMPLEMENTED_RULES):
        test_name = name_by_id[rule_id]
        node = defined[test_name]
        assert not _is_skipped(node), f"{rule_id} is implemented but {test_name} is skipped"
