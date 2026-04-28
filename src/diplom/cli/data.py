from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="diplom-data", description="Dataset download/prepare utilities.")
    sub = p.add_subcommands()

    sudoku = ArgumentParser(description="Sudoku datasets (e.g., Sudoku-Extreme).")
    sudoku.add_argument(
        "--dry-run",
        type=bool,
        default=True,
        help="If true, print actions only. Use --dry-run=false to execute.",
    )
    sudoku.add_argument("--download-metadata", action="store_true", default=False)
    sudoku.add_argument("--materialize", action="store_true", default=False)
    sudoku.add_argument("--out-dir", default="data/sudoku")
    sudoku.add_argument("--subsample", type=int, default=None)
    sudoku.add_argument("--augment", type=int, default=None)
    sub.add_subcommand("sudoku", sudoku)

    text = ArgumentParser(description="Text datasets (HuggingFace datasets).")
    text.add_argument("--name", required=True)
    text.add_argument(
        "--dataset-config",
        default=None,
        help="Optional HF dataset config/subset (e.g. wikitext-103-raw-v1).",
    )
    text.add_argument("--split", default="train")
    text.add_argument(
        "--dry-run",
        type=bool,
        default=True,
        help="If true, print actions only. Use --dry-run=false to execute.",
    )
    text.add_argument("--materialize", action="store_true", default=False)
    text.add_argument("--out-dir", default="data/text")
    sub.add_subcommand("text", text)

    arc = ArgumentParser(description="ARC-AGI datasets (HuggingFace datasets -> jsonl for arc_agi task).")
    arc.add_argument("--name", default="lordspline/arc-agi", help="HF dataset name.")
    arc.add_argument(
        "--dataset-config",
        default=None,
        help="Optional HF dataset config/subset.",
    )
    arc.add_argument("--split-train", default="training", help="HF split to export as train jsonl.")
    arc.add_argument("--split-val", default="evaluation", help="HF split to export as val jsonl.")
    arc.add_argument("--train-filename", default="train.jsonl", help="Output train jsonl file name.")
    arc.add_argument("--val-filename", default="val.jsonl", help="Output val jsonl file name.")
    arc.add_argument(
        "--dry-run",
        type=bool,
        default=True,
        help="If true, print actions only. Use --dry-run=false to execute.",
    )
    arc.add_argument("--materialize", action="store_true", default=False)
    arc.add_argument("--out-dir", default="data/arc_agi")
    sub.add_subcommand("arc-agi", arc)

    ts_stocks = ArgumentParser(description="Stock price time series (yfinance).")
    ts_stocks.add_argument("--tickers", nargs="+", required=True)
    ts_stocks.add_argument("--period", default="5y")
    ts_stocks.add_argument("--interval", default="1d")
    ts_stocks.add_argument(
        "--dry-run",
        type=bool,
        default=True,
        help="If true, print actions only. Use --dry-run=false to execute.",
    )
    ts_stocks.add_argument("--materialize", action="store_true", default=False)
    ts_stocks.add_argument("--out-dir", default="data/timeseries/stocks")
    sub.add_subcommand("timeseries-stocks", ts_stocks)

    ts_public = ArgumentParser(description="Public time series (HuggingFace datasets).")
    ts_public.add_argument("--name", required=True)
    ts_public.add_argument("--split", default="train")
    ts_public.add_argument(
        "--dry-run",
        type=bool,
        default=True,
        help="If true, print actions only. Use --dry-run=false to execute.",
    )
    ts_public.add_argument("--materialize", action="store_true", default=False)
    ts_public.add_argument("--out-dir", default="data/timeseries/public")
    sub.add_subcommand("timeseries-public", ts_public)

    ts_synth = ArgumentParser(description="Synthetic time series for smoke tests.")
    ts_synth.add_argument("--length", type=int, default=4096)
    ts_synth.add_argument("--out-dir", default="data/timeseries/synth")
    sub.add_subcommand("timeseries-synth", ts_synth)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from diplom.data.dispatch import dispatch_data_command  # lazy import

    dispatch_data_command(args)


if __name__ == "__main__":
    main()

