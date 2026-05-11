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
        help="EMA alpha for validation series (sparse points; higher = less history).",
    )
    p.add_argument("--std-window", type=int, default=25, help="Rolling window size for local std band.")
    p.add_argument("--std-sigma", type=float, default=1.0, help="Band half-width in multiples of rolling std.")
    p.add_argument(
        "--out-loss",
        default=None,
        help="Save only the main-loss training panel (train main + total loss).",
    )
    p.add_argument(
        "--out-val-acc",
        default=None,
        help="Save val/train token accuracy on a single y-axis (no dual axis).",
    )
    p.add_argument(
        "--out-val-loss",
        default=None,
        help="Save validation loss-like metrics on a single y-axis.",
    )
    p.add_argument(
        "--plot-exact-acc",
        action="store_true",
        help="Do not filter out keys containing exact_acc (default: exclude for clarity on ARC).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG dpi when using --out-loss / --out-val-acc / --out-val-loss.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.viz.plot_run import plot_run, plot_training_panels  # lazy import

    training_outs = (args.out_loss, args.out_val_acc, args.out_val_loss)
    if any(training_outs):
        exclude: tuple[str, ...] = () if args.plot_exact_acc else ("exact_acc",)
        plot_training_panels(
            args.run_dir,
            out_loss=args.out_loss,
            out_val_acc=args.out_val_acc,
            out_val_loss=args.out_val_loss,
            ema_band=not args.no_ema_band,
            ema_alpha=args.ema_alpha,
            val_ema_alpha=args.val_ema_alpha,
            std_window=args.std_window,
            std_sigma=args.std_sigma,
            dpi=args.dpi,
            exclude_metric_keys=exclude,
        )
        return

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
