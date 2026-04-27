from __future__ import annotations

from diplom.cli.eval_stopping import build_parser


def test_eval_stopping_parser_defaults():
    p = build_parser()
    args = p.parse_args(["--config", "configs/text_wikitext103_trm_oracle.yaml"])
    assert args.config.endswith(".yaml")
    assert "finite_discrete" in args.distribution_models
    assert "cumulative_probability" in args.strategies
