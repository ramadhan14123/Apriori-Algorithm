from __future__ import annotations
import csv
import os
from typing import Dict, List, Tuple


def write_frequent_itemsets_csv(
    out_path: str,
    frequents_by_k: Dict[int, Dict[frozenset, float]],
    support_counts: Dict[frozenset, int],
    num_tx: int,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "itemset", "support_count", "support"])
        for k in sorted(frequents_by_k.keys()):
            for s, sup in frequents_by_k[k].items():
                w.writerow([k, ";".join(sorted(list(s))), support_counts.get(s, 0), round(sup, 6)])


def write_rules_csv(out_path: str, rules: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction"])
        for r in rules:
            w.writerow([
                ";".join(r["antecedent"]),  # type: ignore
                ";".join(r["consequent"]),  # type: ignore
                r["support"],
                r["confidence"],
                r["lift"],
                r["leverage"],
                r["conviction"],
            ])


def write_all_combos_csv(out_path: str, combos: List[Tuple[Tuple[str, ...], int, float]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "itemset", "support_count", "support"])
        for items, cnt, sup in combos:
            w.writerow([len(items), ";".join(items), cnt, round(sup, 6)])


def write_rule_accuracy_csv(out_path: str, rules_eval: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "test_support",
            "test_confidence",
        ])
        for r in rules_eval:
            w.writerow([
                ";".join(r["antecedent"]),  # type: ignore
                ";".join(r["consequent"]),  # type: ignore
                r["support"],
                r["confidence"],
                r["lift"],
                r["leverage"],
                r["conviction"],
                r.get("test_support", ""),
                r.get("test_confidence", ""),
            ])
