from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(prog="diplom-train", description="Train an experiment from a YAML config.")
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.runner.train import train_from_yaml  # imported lazily for fast --help

    train_from_yaml(args.config)


if __name__ == "__main__":
    main()

