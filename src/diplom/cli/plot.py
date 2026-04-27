from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="diplom-plot", description="Plot metrics from a run directory.")
    p.add_argument("--run-dir", required=True, help="Path to a run directory (runs/<id>).")
    p.add_argument("--out", default=None, help="Optional output path for the figure (PNG).")
    p.add_argument(
        "--no-ema-band",
        action="store_true",
        help="Disable EMA trend line and rolling-std band on noisy series.",
    )
    p.add_argument("--ema-alpha", type=float, default=0.06, help="EMA smoothing factor (higher = faster tracking).")
    p.add_argument(
        "--val-ema-alpha",
        type=float,
        default=0.35,
        help="EMA alpha for validation series on the bottom panel only (sparse points; higher = less history).",
    )
    p.add_argument("--std-window", type=int, default=25, help="Rolling window size for local std band.")
    p.add_argument("--std-sigma", type=float, default=1.0, help="Band half-width in multiples of rolling std.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.viz.plot_run import plot_run  # lazy import

    plot_run(
        args.run_dir,
        out_path=args.out,
        ema_band=not args.no_ema_band,
        ema_alpha=args.ema_alpha,
        val_ema_alpha=args.val_ema_alpha,
        std_window=args.std_window,
        std_sigma=args.std_sigma,
    )


if __name__ == "__main__":
    main()

