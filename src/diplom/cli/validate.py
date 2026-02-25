from __future__ import annotations

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="diplom-validate",
        description="Validate a checkpoint or run evaluation from a YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to experiment YAML config.")
    p.add_argument("--checkpoint", default=None, help="Optional path to model checkpoint.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from diplom.runner.validate import validate_from_yaml  # lazy import

    validate_from_yaml(args.config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

