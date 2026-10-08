"""Phase 8 association rules: pure mining/filter/recommendation logic (METRIC-08,
PRED-07). Each row of the dataset is one basket; every non-null cell value is an item."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from itertools import combinations
from typing import Any

import pandas as pd

MAX_ITEMSET_LEN: int = 3  # KISS: cap itemset size so mining stays fast on tabular data
MAX_FREQUENT_ITEMSETS: int = 2_000


def baskets_from_frame(frame: pd.DataFrame) -> list[frozenset[str]]:
    """One basket per row: stringified non-null cell values become items."""
    baskets: list[frozenset[str]] = []
    for row in frame.itertuples(index=False, name=None):
        items = {str(value) for value in row if not pd.isna(value)}
        if items:
            baskets.append(frozenset(items))
    return baskets


def item_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    """EDA-08: per-item basket counts and frequencies."""
    counts: Counter[str] = Counter()
    baskets = baskets_from_frame(frame)
    for basket in baskets:
        counts.update(basket)
    total = len(baskets) or 1
    rows = [
        {"item": item, "count": count, "frequency": round(count / total, 4)}
        for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    return pd.DataFrame(rows, columns=["item", "count", "frequency"])


def basket_size_distribution(frame: pd.DataFrame) -> pd.Series:
    """EDA-08: how many items each basket holds (size → basket count)."""
    sizes = Counter(len(basket) for basket in baskets_from_frame(frame))
    series = pd.Series(sizes, name="baskets")
    series.index.name = "size"
    return series.sort_index()


def _supports(
    itemsets: Iterable[frozenset[str]], baskets: list[frozenset[str]]
) -> dict[frozenset[str], int]:
    counts: dict[frozenset[str], int] = {itemset: 0 for itemset in itemsets}
    for basket in baskets:
        for itemset in counts:
            if itemset <= basket:
                counts[itemset] += 1
    return counts


def frequent_itemsets(
    baskets: list[frozenset[str]],
    min_support: float,
    *,
    max_len: int = MAX_ITEMSET_LEN,
) -> dict[frozenset[str], float]:
    """Apriori level-wise mining of frequent itemsets (support ≥ min_support)."""
    n_baskets = len(baskets) or 1
    frequent: dict[frozenset[str], float] = {}
    singles = sorted({item for basket in baskets for item in basket})
    counts = _supports((frozenset({item}) for item in singles), baskets)
    current = {
        itemset: support
        for itemset, support in ((k, v / n_baskets) for k, v in counts.items())
        if support >= min_support
    }
    frequent.update(current)
    for _length in range(2, min(max_len, len(singles)) + 1):
        prev = list(current)
        candidates = (
            {frozenset(pair) for pair in combinations(sorted(set().union(*prev)), 2)}
            if prev
            else set()
        )
        # Apriori property: every subset of a frequent itemset must be frequent.
        candidates = {
            candidate
            for candidate in candidates
            if all(
                frozenset(subset) in current
                for subset in combinations(candidate, len(candidate) - 1)
            )
        }
        if not candidates or len(frequent) + len(candidates) > MAX_FREQUENT_ITEMSETS:
            break
        counts = _supports(candidates, baskets)
        current = {
            itemset: support
            for itemset, support in ((k, v / n_baskets) for k, v in counts.items())
            if support >= min_support
        }
        frequent.update(current)
    return frequent


def mine_rules(
    baskets: list[frozenset[str]],
    *,
    min_support: float = 0.1,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
    max_len: int = MAX_ITEMSET_LEN,
) -> list[dict[str, Any]]:
    """METRIC-08: association rules from frequent itemsets, filtered by
    confidence and lift, sorted by lift (then confidence) descending."""
    if not baskets:
        return []
    frequent = frequent_itemsets(baskets, min_support, max_len=max_len)
    rules: list[dict[str, Any]] = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for size in range(1, len(items)):
            for ante in combinations(items, size):
                antecedent = frozenset(ante)
                consequent = itemset - antecedent
                ante_support = frequent.get(antecedent)
                if not ante_support:
                    continue
                confidence = support / ante_support
                conc_support = frequent.get(consequent)
                if not conc_support:
                    continue
                lift = confidence / conc_support
                if confidence < min_confidence or lift < min_lift:
                    continue
                rules.append(
                    {
                        "antecedent": tuple(sorted(antecedent)),
                        "consequent": tuple(sorted(consequent)),
                        "support": round(support, 6),
                        "confidence": round(confidence, 6),
                        "lift": round(lift, 6),
                    }
                )
    rules.sort(key=lambda rule: (-rule["lift"], -rule["confidence"], -rule["support"]))
    return rules


def filter_rules(
    rules: Iterable[dict[str, Any]],
    *,
    min_support: float = 0.0,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
) -> list[dict[str, Any]]:
    """METRIC-08: keep only rules meeting all three minimums."""
    return [
        rule
        for rule in rules
        if rule["support"] >= min_support
        and rule["confidence"] >= min_confidence
        and rule["lift"] >= min_lift
    ]


def recommend_items(
    basket: Iterable[str],
    rules: Iterable[dict[str, Any]],
    *,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """PRED-07: items to add to a basket, from rules whose antecedent is already
    satisfied, scored by lift × confidence (best score per candidate item)."""
    held = {str(item) for item in basket}
    best: dict[str, dict[str, Any]] = {}
    for rule in rules:
        antecedent = set(rule["antecedent"])
        if not antecedent or not antecedent <= held:
            continue
        for item in rule["consequent"]:
            if item in held:
                continue
            score = float(rule["lift"]) * float(rule["confidence"])
            candidate = {
                "item": item,
                "score": round(score, 6),
                "lift": rule["lift"],
                "confidence": rule["confidence"],
                "support": rule["support"],
                "rule": f"{' + '.join(sorted(antecedent))} => {item}",
            }
            if item not in best or score > best[item]["score"]:
                best[item] = candidate
    ranked = sorted(best.values(), key=lambda row: -row["score"])
    return ranked[:top_n]


def association_rule_stats(rules: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Leaderboard scoreboard for association runs: mean support/confidence/lift."""
    rows = list(rules)
    if not rows:
        return {"support": 0.0, "confidence": 0.0, "lift": 0.0}
    n = len(rows)

    def _mean(key: str) -> float:
        return round(math.fsum(float(rule[key]) for rule in rows) / n, 6)

    return {"support": _mean("support"), "confidence": _mean("confidence"), "lift": _mean("lift")}
