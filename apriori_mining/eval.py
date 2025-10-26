from __future__ import annotations
from typing import Dict, List, Set


def evaluate_rules_on_test(
    rules: List[Dict[str, object]],
    test_transactions: List[Set[str]],
) -> List[Dict[str, object]]:
    num_tx = len(test_transactions)
    if num_tx == 0:
        return []

    def count_support(tx_list: List[Set[str]], itemset: Set[str]) -> int:
        c = 0
        for tx in tx_list:
            if itemset.issubset(tx):
                c += 1
        return c

    evaluated: List[Dict[str, object]] = []
    for r in rules:
        ante = set(r["antecedent"])  # type: ignore
        cons = set(r["consequent"])  # type: ignore
        both = ante | cons
        ante_cnt = count_support(test_transactions, ante)
        both_cnt = count_support(test_transactions, both)
        test_support = both_cnt / num_tx
        test_conf = (both_cnt / ante_cnt) if ante_cnt > 0 else 0.0
        r_eval = dict(r)
        r_eval["test_support"] = round(test_support, 6)
        r_eval["test_confidence"] = round(test_conf, 6)
        evaluated.append(r_eval)

    evaluated.sort(key=lambda r: (-r["test_confidence"], -r["test_support"], -r["confidence"]))
    return evaluated
