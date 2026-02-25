from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="diplom-plot", description="Plot metrics from a run directory.")
    p.add_argument("--run-dir", required=True, help="Path to a run directory (runs/<id>).")
    p.add_argument("--out", default=None, help="Optional output path for the figure (PNG).")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.viz.plot_run import plot_run  # lazy import

    plot_run(args.run_dir, out_path=args.out)


if __name__ == "__main__":
    main()

