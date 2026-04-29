from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="diplom-train", description="Train an experiment from a YAML config.")
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to load before training.",
    )
    p.add_argument(
        "--oracle-only",
        action="store_true",
        help="Freeze backbone and train only oracle head parameters.",
    )
    p.add_argument(
        "--live-plots",
        action="store_true",
        help="Update plots during training (writes run_dir/plots.png periodically).",
    )
    p.add_argument(
        "--live-plot-every",
        type=int,
        default=None,
        help="Refresh interval in global steps for live plots (overrides config when set).",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.runner.train import train_from_yaml  # imported lazily for fast --help

    train_from_yaml(
        args.config,
        init_checkpoint=args.init_checkpoint,
        oracle_only=args.oracle_only,
        live_plots_override=True if args.live_plots else None,
        live_plot_every_override=args.live_plot_every,
    )


if __name__ == "__main__":
    main()

