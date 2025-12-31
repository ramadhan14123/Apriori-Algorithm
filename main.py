import argparse
import os
import random
import time
from typing import Dict, List

from apriori_mining import (
    read_transactions,
    build_fast_context,
    apriori,
    all_combinations_support,
    generate_rules,
    evaluate_rules_on_test,
    write_frequent_itemsets_csv,
    write_rules_csv,
    write_all_combos_csv,
    write_rule_accuracy_csv,
)
from apriori_mining.config import AprioriConfig, load_config_json

def main():
	parser = argparse.ArgumentParser(description="Apriori Data Mining (pure Python)")
	parser.add_argument(
		"--data",
		type=str,
		default=None,
		help="Path to CSV dataset (one-hot 0/1).",
	)
	parser.add_argument("--min_support", type=float, default=None, help="Minimum support (0-1) for frequent itemsets.")
	parser.add_argument("--min_confidence", type=float, default=None, help="Minimum confidence (0-1) for rules.")
	parser.add_argument("--output_all_rules", action="store_true", help="If set, include all generated rules (no filtering by min_confidence/min_lift). The original filtering code will be kept as comments.")
	parser.add_argument("--min_lift", type=float, default=None, help="Minimum lift for rules (default: 1.0).")
	parser.add_argument(
		"--all_combos_max_k",
		type=int,
		default=None,
		help="Limit exhaustive combinations to max itemset size k (default: all).",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed for train/test split.")
	parser.add_argument("--test_size", type=float, default=None, help="Test fraction for rule accuracy evaluation.")
	parser.add_argument("--no_eval", action="store_true", help="Skip train/test rule accuracy evaluation.")
	parser.add_argument("--no_progress", action="store_true", help="Disable progress bars/logs.")
	parser.add_argument("--config", type=str, default=None, help="Path to JSON config file to set defaults.")
	args = parser.parse_args()

	default_data = os.path.join(os.path.dirname(__file__), "Datasets", "20_datasets.csv")
	cfg: AprioriConfig = load_config_json(args.config) if args.config else AprioriConfig()
	data_path = args.data or cfg.data or default_data
	min_support = args.min_support if args.min_support is not None else (cfg.min_support if cfg.min_support is not None else 0.05)
	min_confidence = args.min_confidence if args.min_confidence is not None else (cfg.min_confidence if cfg.min_confidence is not None else 0.5)
	min_lift = args.min_lift if args.min_lift is not None else (cfg.min_lift if cfg.min_lift is not None else 1.0)
	max_k = args.all_combos_max_k if args.all_combos_max_k is not None else cfg.all_combos_max_k
	seed = args.seed if args.seed is not None else (cfg.seed if cfg.seed is not None else 42)
	test_size = args.test_size if args.test_size is not None else (cfg.test_size if cfg.test_size is not None else 0.2)
	no_eval = True if args.no_eval else (cfg.no_eval if cfg.no_eval is not None else False)
	progress = False if args.no_progress else (cfg.progress if cfg.progress is not None else True)

	t0 = time.time()
	print("[1/5] Loading dataset and building context...")
	transactions, item_names, ctx = read_transactions(data_path, progress=bool(progress))
	num_tx = len(transactions)
	print(f"Loaded {num_tx} transactions, {len(item_names)} items.")
	print("Items:", ", ".join(item_names))

	# Apriori frequent itemsets
	print("[2/5] Mining frequent itemsets (Apriori)...")
	t1 = time.time()
	frequents_by_k, support_counts = apriori(transactions, min_support=min_support, ctx=ctx, progress=bool(progress))
	t2 = time.time()
	print(f"Frequent itemsets computed in {t2 - t1:.2f}s. Levels: {list(frequents_by_k.keys())}")

	# Association rules
	print("[3/5] Generating association rules...")
	rules = generate_rules(
		frequents_by_k, support_counts, min_confidence, num_tx, min_lift=min_lift, output_all=args.output_all_rules
	)
	if args.output_all_rules:
		print(f"Generated {len(rules)} rules (no confidence/lift filtering applied)")
	else:
		print(f"Generated {len(rules)} rules with confidence >= {min_confidence} and lift >= {min_lift}")

	# Exhaustive combinations
	print("[4/5] Computing support for all combinations...")
	t3 = time.time()
	combos = all_combinations_support(transactions, item_names, max_k=max_k, ctx=ctx)
	t4 = time.time()
	print(f"All combinations (non-zero support) computed in {t4 - t3:.2f}s. Total combos: {len(combos)}")

	# Optional train/test evaluation
	rules_eval: List[Dict[str, object]] = []
	if not no_eval and num_tx > 1 and len(rules) > 0:
		print("[5/5] Evaluating rules on test split...")
		random.seed(seed)
		idx = list(range(num_tx))
		random.shuffle(idx)
		split = int((1 - test_size) * num_tx)
		train_idx, test_idx = idx[:split], idx[split:]
		train_tx = [transactions[i] for i in train_idx]
		test_tx = [transactions[i] for i in test_idx]
		# Refit on train to avoid test leakage when counting support
		ctx_train = build_fast_context(train_tx, item_names, progress=bool(progress))
		frequents_train, counts_train = apriori(train_tx, min_support=min_support, ctx=ctx_train, progress=bool(progress))
		rules_train = generate_rules(
			frequents_train, counts_train, min_confidence, len(train_tx), min_lift=min_lift, output_all=args.output_all_rules
		)
		rules_eval = evaluate_rules_on_test(rules_train, test_tx)
		print(f"Evaluated {len(rules_eval)} rules on test set ({len(test_tx)} tx).")

	# Outputs
	out_dir = os.path.join(os.path.dirname(__file__), "Outputs")
	write_frequent_itemsets_csv(os.path.join(out_dir, "frequent_itemsets.csv"), frequents_by_k, support_counts, num_tx)
	write_rules_csv(os.path.join(out_dir, "association_rules.csv"), rules)
	write_all_combos_csv(os.path.join(out_dir, "all_combinations.csv"), combos)
	if rules_eval:
		write_rule_accuracy_csv(os.path.join(out_dir, "rule_accuracy_test.csv"), rules_eval)

	# Print quick top summaries
	def fmt_itemset(s: frozenset) -> str:
		return "{" + ", ".join(sorted(list(s))) + "}"

	print("\nTop frequent itemsets (by support):")
	for k in sorted(frequents_by_k.keys()):
		items_sorted = sorted(frequents_by_k[k].items(), key=lambda x: -x[1])[:5]
		for s, sup in items_sorted:
			print(f"  k={k} {fmt_itemset(s)} -> support={sup:.3f}, count={support_counts.get(s,0)}")

	print("\nTop rules (by confidence):")
	for r in rules[:10]:
		print(
			f"  {','.join(r['antecedent'])} => {','.join(r['consequent'])} | "
			f"supp={r['support']:.3f}, conf={r['confidence']:.3f}, lift={r['lift']:.3f}"
		)

	if rules_eval:
		print("\nTop rules on test (by test_confidence):")
		for r in rules_eval[:10]:
			print(
				f"  {','.join(r['antecedent'])} => {','.join(r['consequent'])} | "
				f"test_conf={r['test_confidence']:.3f}, test_supp={r['test_support']:.3f}"
			)

	t5 = time.time()
	print(f"\nDone in {t5 - t0:.2f}s. Outputs saved to: {out_dir}")


if __name__ == "__main__":
	main()

