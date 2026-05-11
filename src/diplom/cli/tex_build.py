from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from jsonargparse import ArgumentParser


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="diplom-tex-build",
        description="Build LaTeX document (prefer latexmk, fallback to xelatex passes).",
    )
    p.add_argument("--workdir", default="tex", help="Directory containing the LaTeX project.")
    p.add_argument("--main", default="thesis.tex", help="Main TeX file name.")
    p.add_argument("--out-dir", default="build", help="Output directory for generated files.")
    p.add_argument(
        "--engine",
        default="xelatex",
        choices=["xelatex", "lualatex", "pdflatex"],
        help="TeX engine for latexmk/fallback runs.",
    )
    p.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Fallback engine passes when latexmk is unavailable.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Clean auxiliary build artifacts via latexmk -c before build.",
    )
    return p


def _run(cmd: list[str], cwd: Path, *, extra_env: dict[str, str] | None = None) -> None:
    # Some systems ship a broken default `paper` config; xdvipdfmx then fails with
    # "Unrecognized paper format: # Simply write the paper name".
    env = {**os.environ, "PAPER": "a4", "PAPERSIZE": "a4"}
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
    args = build_parser().parse_args()
    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        raise SystemExit(f"Workdir does not exist: {workdir}")
    main_tex = workdir / args.main
    if not main_tex.exists():
        raise SystemExit(f"Main TeX file not found: {main_tex}")
    out_dir = workdir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = str(args.engine)

    if shutil.which("latexmk"):
        if args.clean:
            _run(["latexmk", "-c", f"-outdir={args.out_dir}", args.main], cwd=workdir)
        _run(["latexmk", f"-{engine}", "-interaction=nonstopmode", f"-outdir={args.out_dir}", args.main], cwd=workdir)
        return

    # Fallback when latexmk is not installed.
    engine_bin = shutil.which(engine)
    if not engine_bin:
        raise SystemExit(f"Neither latexmk nor {engine} found in PATH.")
    passes = max(int(args.passes), 1)
    stem = main_tex.stem
    out_rel = args.out_dir

    def run_engine() -> None:
        _run(
            [
                engine_bin,
                "-interaction=nonstopmode",
                f"-output-directory={out_rel}",
                args.main,
            ],
            cwd=workdir,
        )

    run_engine()
    bcf = out_dir / f"{stem}.bcf"
    biber_bin = shutil.which("biber")
    bibtex_bin = shutil.which("bibtex")
    if bcf.exists() and biber_bin:
        _run([biber_bin, stem], cwd=str(out_dir))
    elif (out_dir / f"{stem}.aux").exists() and bibtex_bin:
        _run(
            [bibtex_bin, stem],
            cwd=out_dir,
            extra_env={"BIBINPUTS": str(workdir) + os.pathsep + os.environ.get("BIBINPUTS", "")},
        )
    for _ in range(passes):
        run_engine()


if __name__ == "__main__":
    main()
