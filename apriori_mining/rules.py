from __future__ import annotations
from itertools import combinations
from typing import Dict, List


def generate_rules(
    frequents_by_k: Dict[int, Dict[frozenset, float]],
    support_counts: Dict[frozenset, int],
    min_confidence: float,
    num_tx: int,
    min_lift: float = 1.0,
    output_all: bool = False,
) -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    for k, d in frequents_by_k.items():
        if k < 2:
            continue
        for itemset, supp in d.items():
            items = list(itemset)
            for r in range(1, len(items)):
                for antecedent in combinations(items, r):
                    antecedent_fs = frozenset(antecedent)
                    consequent_fs = itemset - antecedent_fs
                    if not consequent_fs:
                        continue
                    supp_itemset = support_counts.get(itemset, 0) / num_tx
                    supp_ante = support_counts.get(antecedent_fs, 0) / num_tx
                    supp_cons = support_counts.get(consequent_fs, 0) / num_tx
                    if supp_ante <= 0:
                        continue
                    confidence = supp_itemset / supp_ante
                    # By default we filter rules by min_confidence and min_lift.
                    # If `output_all` is True we skip filtering so all generated rules are returned
                    # (the original filtering logic is preserved below as commented lines).
                    if not output_all:
                        if confidence + 1e-12 < min_confidence:
                            continue
                    # else: (no filtering by confidence)

                    lift = confidence / supp_cons if supp_cons > 0 else float("inf")
                    if not output_all:
                        if lift + 1e-12 < min_lift:
                            continue
                    # else: (no filtering by lift)
                    leverage = supp_itemset - (supp_ante * supp_cons)
                    conviction = (1 - supp_cons) / (1 - confidence) if (1 - confidence) > 0 else float("inf")
                    rules.append(
                        {
                            "antecedent": tuple(sorted(antecedent_fs)),
                            "consequent": tuple(sorted(consequent_fs)),
                            "support": round(supp_itemset, 6),
                            "confidence": round(confidence, 6),
                            "lift": round(lift, 6),
                            "leverage": round(leverage, 6),
                            "conviction": round(conviction, 6),
                        }
                    )
    rules.sort(key=lambda r: (-r["confidence"], -r["lift"], -r["support"], r["antecedent"], r["consequent"]))
    return rules
