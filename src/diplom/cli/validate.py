from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="diplom-validate",
        description="Validate a checkpoint or run evaluation from a YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    p.add_argument("--checkpoint", default=None, help="Optional path to model checkpoint.")
    p.add_argument(
        "--oracle-policy",
        default="none",
        choices=["none", "greedy", "sampling"],
        help="Inference policy for TRMOracle recursion budget selection.",
    )
    p.add_argument(
        "--oracle-max-steps",
        type=int,
        default=None,
        help="Maximum reasoning steps during oracle inference (can exceed N_sup).",
    )
    p.add_argument(
        "--oracle-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for --oracle-policy sampling.",
    )
    p.add_argument(
        "--progress-bar",
        type=bool,
        default=None,
        help="Enable/disable validation progress bar. Defaults to train.progress_bar from YAML.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.runner.validate import validate_from_yaml  # lazy import

    validate_from_yaml(
        args.config,
        checkpoint_path=args.checkpoint,
        oracle_policy=args.oracle_policy,
        oracle_max_steps=args.oracle_max_steps,
        oracle_temperature=args.oracle_temperature,
        progress_bar=args.progress_bar,
    )


if __name__ == "__main__":
    main()

