from __future__ import annotations

import json

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="diplom-eval-stopping",
        description="Evaluate probabilistic oracle distributions and stopping strategies.",
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    p.add_argument(
        "--distribution-models",
        default="finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid",
        help="Comma-separated distribution models.",
    )
    p.add_argument(
        "--strategies",
        default="cumulative_probability,future_improvement,hazard,quantile,budget",
        help="Comma-separated stopping strategies.",
    )
    p.add_argument(
        "--threshold-grid",
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated thresholds for non-budget strategies.",
    )
    p.add_argument(
        "--budget-grid",
        default="2,4,6,8",
        help="Comma-separated target expected step budgets for budget strategy.",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Override N_sup at evaluation.")
    p.add_argument("--out", default=None, help="Optional output JSON path.")
    p.add_argument(
        "--honest-split-ratio",
        type=float,
        default=0.0,
        help="Fraction of validation used to select threshold/strategy before scoring the remainder (0 disables, e.g. 0.5).",
    )
    p.add_argument(
        "--selection-metric",
        default="token_acc",
        help="Metric for selecting threshold/strategy on the holdout portion of validation.",
    )
    p.add_argument(
        "--selection-mode",
        default="max",
        choices=["min", "max"],
        help="Optimization mode for --selection-metric.",
    )
    p.add_argument(
        "--answer-policies",
        default="last,argmax_interval",
        help="Comma-separated answer policies for metric calculation (e.g. last,argmax_interval).",
    )
    p.add_argument(
        "--progress-bar",
        type=bool,
        default=None,
        help="Enable/disable progress bar. Defaults to train.progress_bar from YAML.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.runner.eval_stopping import eval_stopping_from_yaml

    res = eval_stopping_from_yaml(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        distribution_models=args.distribution_models,
        strategies=args.strategies,
        threshold_grid=args.threshold_grid,
        budget_grid=args.budget_grid,
        max_steps=args.max_steps,
        out_path=args.out,
        honest_split_ratio=args.honest_split_ratio,
        selection_metric=args.selection_metric,
        selection_mode=args.selection_mode,
        answer_policies=args.answer_policies,
        progress_bar=args.progress_bar,
    )
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
