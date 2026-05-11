"""Network architecture figures for thesis (matplotlib, inch coordinates)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Match thesis Cyrillic/math readability
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    }
)

# Transformer-style pastel palette (soft, print-friendly)
C_INPUT = "#c6e6ff"
C_ENC = "#ffd6a8"
C_LOOP_BORDER = "#6b6b6b"
C_LOOP_FACE = "#e9eaeb"
C_STATE = "#f7f7f8"
C_CORE = "#ffab9e"
C_NORM = "#e6def8"
C_LM = "#9fd89f"
C_TIME = "#ffe799"
C_STOP = "#dddddd"


def _rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    fontsize: float = 9,
    linewidth: float = 1.0,
    edgecolor: str = "0.22",
    text_color: str = "0.05",
    rounding: float = 0.08,
    zorder: int = 4,
    fontweight: str | None = None,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    kw: dict = {
        "ha": "center",
        "va": "center",
        "fontsize": fontsize,
        "color": text_color,
        "zorder": zorder + 1,
        "linespacing": 1.28,
    }
    if fontweight:
        kw["fontweight"] = fontweight
    ax.text(x + w / 2, y + h / 2, text, **kw)
    return patch


def _arrow(
    ax,
    xy_from: tuple[float, float],
    xy_to: tuple[float, float],
    *,
    lw: float = 1.2,
    color: str = "0.28",
    style: str = "-",
    zorder: float = 3,
    shrink_a: float = 4,
    shrink_b: float = 4,
    rad: float = 0,
    mutation_scale: float = 12,
) -> FancyArrowPatch:
    ap = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(ap)
    return ap


def _curved_vert_arrow(
    ax,
    xa: float,
    y_top: float,
    y_bot: float,
    *,
    xrad: float = 0,
    lw: float = 1.2,
    color: str = "0.32",
) -> None:
    """Arrow along x=xa from y_top (high) downward to y_bot (low)."""
    _arrow(ax, (xa, y_top), (xa, y_bot), lw=lw, color=color, shrink_a=2, shrink_b=3, rad=xrad)


def _polyline_dashed_arrow(
    ax,
    pts: list[tuple[float, float]],
    *,
    lw: float = 1.05,
    color: str = "0.45",
) -> None:
    """Dashed orthogonal-ish route through pts; arrowhead only on final segment."""
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        xa, ya = pts[i]
        xb, yb = pts[i + 1]
        linestyle = (0, (6, 4))
        if i == len(pts) - 2:
            _arrow(ax, (xa, ya), (xb, yb), lw=lw, color=color, style=linestyle, shrink_b=5)
        else:
            ln = Line2D(
                [xa, xb],
                [ya, yb],
                linestyle=(0, (6, 4)),
                linewidth=lw,
                color=color,
                solid_capstyle="round",
                zorder=3,
            )
            ax.add_line(ln)


def draw_trm_loop_heads_stop(ax, *, x0: float, y0: float, ww: float, fontsize: float) -> tuple[float, float]:
    """Draw TRM block; forward pass bottom→top inside figure coords. Returns (cx, ymax) for ylim."""
    cx = x0 + ww / 2

    loop_x = x0 + 0.32
    loop_w = ww - 0.64
    loop_y = y0 + 0.18
    loop_h = 4.95

    loop_bg = FancyBboxPatch(
        (loop_x, loop_y),
        loop_w,
        loop_h,
        boxstyle="round,pad=0.025,rounding_size=0.16",
        linewidth=1.15,
        edgecolor=C_LOOP_BORDER,
        facecolor=C_LOOP_FACE,
        zorder=1,
    )
    ax.add_patch(loop_bg)

    subtitle = (
        r"внутренние шаги $k$: до принятия решения по $\widehat{p}_{t,k}$\,/\,$\varphi_k$ "
        r"или при $k\,{=}\,K$"
    )
    ax.text(
        loop_x + loop_w / 2,
        loop_y + loop_h - 0.2,
        subtitle,
        ha="center",
        va="top",
        fontsize=fontsize - 1.35,
        color="0.18",
        zorder=2,
    )

    ix1 = loop_x + 0.36
    ix2 = loop_x + loop_w - 0.36
    inner_w = ix2 - ix1

    state_w = inner_w * 0.76
    state_x = ix1 + (inner_w - state_w) / 2
    state_y = loop_y + 0.78
    state_h = 0.62
    _rounded_box(
        ax,
        state_x,
        state_y,
        state_w,
        state_h,
        "скрытое состояние  " + r"$Z_{t,k}$",
        facecolor=C_STATE,
        fontsize=fontsize - 0.35,
        edgecolor="0.35",
    )

    core_w = inner_w * 0.85
    core_x = ix1 + (inner_w - core_w) / 2
    core_y = state_y + state_h + 0.34
    core_h = 1.28
    _rounded_box(
        ax,
        core_x,
        core_y,
        core_w,
        core_h,
        "ядро рекурсии  " + r"$\mathcal{U}_{\theta}$"
        "\n"
        r"$Z \leftarrow Z + \mathcal{U}_{\theta}(Z, C_t, Q_{t,k})$",
        facecolor=C_CORE,
        fontsize=fontsize - 0.1,
        edgecolor="#8a3828",
        linewidth=1.08,
    )

    norm_w = inner_w * 0.55
    norm_x = ix1 + (inner_w - norm_w) / 2
    norm_y = core_y + core_h + 0.3
    norm_h = 0.54
    _rounded_box(
        ax,
        norm_x,
        norm_y,
        norm_w,
        norm_h,
        "нормализация\n(layer norm или аналог; по конфигурации опц.)",
        facecolor=C_NORM,
        fontsize=fontsize - 1.45,
        edgecolor="#4a4180",
    )

    inner_cx = (ix1 + ix2) / 2
    _arrow(ax, (inner_cx, state_y + state_h), (inner_cx, core_y), lw=1.15)
    _arrow(ax, (inner_cx, core_y + core_h), (inner_cx, norm_y), lw=1.15)

    left_x = loop_x + 0.26
    y_step_top = norm_y + norm_h * 0.58
    y_step_bot = state_y + state_h * 0.18
    _curved_vert_arrow(ax, left_x, y_step_top, y_step_bot, xrad=-0.11, lw=1.35, color="#2a5870")
    ax.text(
        left_x - 0.52,
        (y_step_top + y_step_bot) / 2,
        "новый индекс  " + r"$k \leftarrow k{+}1$",
        ha="right",
        va="center",
        fontsize=fontsize - 2.0,
        color="#2a5870",
        rotation=90,
    )

    bus_y_lo = norm_y + norm_h
    bus_y_mid = loop_y + loop_h + 0.28
    _arrow(ax, (inner_cx, bus_y_lo), (inner_cx, bus_y_mid), lw=1.2)

    lm_w = 3.08
    tm_w = 3.08
    gh = 1.03
    gap = 0.62
    span = lm_w + gap + tm_w
    lm_x = cx - span / 2
    tm_x = lm_x + lm_w + gap
    heads_bottom = bus_y_mid
    lm_c = lm_x + lm_w / 2
    tm_c = tm_x + tm_w / 2

    _arrow(ax, (inner_cx, bus_y_mid), (lm_c, heads_bottom), lw=1.08, shrink_b=8)
    _arrow(ax, (inner_cx, bus_y_mid), (tm_c, heads_bottom), lw=1.08, shrink_b=8)

    _rounded_box(
        ax,
        lm_x,
        heads_bottom,
        lm_w,
        gh,
        "LM-голова  " + r"$g_{\eta}$" "\n" r"$\mathrm{softmax} \Rightarrow Q_{t,k}$",
        facecolor=C_LM,
        fontsize=fontsize - 0.08,
        edgecolor="#356035",
        linewidth=1.05,
    )
    _rounded_box(
        ax,
        tm_x,
        heads_bottom,
        tm_w,
        gh,
        r"$R_{\psi}(\mathbf{h})$" "\n" r"$\mathrm{softmax} \Rightarrow \widehat{p}_{t,k}$",
        facecolor=C_TIME,
        fontsize=fontsize - 0.25,
        edgecolor="#917000",
        linewidth=1.05,
    )

    # Q вход в ядро: обходится слева без пересечения головок
    y_core_in = core_y + core_h * 0.42
    _polyline_dashed_arrow(
        ax,
        [
            (lm_c, heads_bottom + 0.12),
            (lm_x - 0.92, heads_bottom + 0.62),
            (lm_x - 0.92, y_core_in),
            (core_x - 0.12, y_core_in),
        ],
        lw=1.12,
        color="0.38",
    )
    ax.text(
        lm_x - 1.06,
        (heads_bottom + 0.62 + y_core_in) / 2,
        r"$Q_{t,k}$",
        fontsize=fontsize - 1.15,
        color="0.4",
        ha="right",
        va="center",
    )

    stop_w = 4.05
    stop_h = 0.64
    stop_y = heads_bottom + gh + 0.45
    stop_x = cx - stop_w / 2
    _arrow(ax, (tm_c, heads_bottom + gh), (cx, stop_y), lw=1.12)

    _rounded_box(
        ax,
        stop_x,
        stop_y,
        stop_w,
        stop_h,
        r"решение о следующем шаге / остановке (по $\widehat{p}_{t,k}$)",
        facecolor=C_STOP,
        fontsize=fontsize - 1.05,
        edgecolor="0.4",
    )

    ymax_reached = stop_y + stop_h + 0.25
    return cx, ymax_reached


def draw_pure_trm_network(*, outfile_base: Path, dpi: int, formats: list[str]) -> None:
    W, H = 7.45, 10.85
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fs = 9.45
    cx = W / 2

    in_w, in_h = 4.05, 0.58
    in_x = cx - in_w / 2
    in_y = 0.45
    _rounded_box(
        ax,
        in_x,
        in_y,
        in_w,
        in_h,
        r"Входные токены / префикс $X_{1:t}$",
        facecolor=C_INPUT,
        fontsize=fs - 0.1,
        edgecolor="#35668a",
    )

    enc_w, enc_h = 4.5, 0.78
    enc_x = cx - enc_w / 2
    enc_y = in_y + in_h + 0.55
    _rounded_box(
        ax,
        enc_x,
        enc_y,
        enc_w,
        enc_h,
        "кодировщик префикса\n" + r"$E_{\phi^{(\mathrm{trm})}}$"
        "\n"
        r"$\rightarrow Z_{t,0}$",
        facecolor=C_ENC,
        fontsize=fs - 0.15,
        edgecolor="#8a5820",
    )

    _arrow(ax, (cx, in_y + in_h), (cx, enc_y), lw=1.2)

    stack_y0 = enc_y + enc_h + 0.55
    ww = W - 0.85
    x_frame = (W - ww) / 2
    _arrow(ax, (cx, enc_y + enc_h), (cx, stack_y0 + 0.12), lw=1.2)

    _, ymax = draw_trm_loop_heads_stop(ax, x0=x_frame, y0=stack_y0, ww=ww, fontsize=fs)

    margin = 0.25
    ax.set_ylim(0, max(H, ymax + 0.45))
    ax.set_xlim(-margin, W + margin)
    plt.tight_layout(pad=0.08)
    _save_figure(fig, outfile_base, dpi=dpi, formats=formats)
    plt.close(fig)


def draw_sllm_trm_network(*, outfile_base: Path, dpi: int, formats: list[str]) -> None:
    W, H = 7.95, 13.2
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    fs = 9.35
    cx = W / 2 + 0.15

    in_w, in_h = 2.15, 0.58
    in_x = cx - 3.15
    in_y = 0.45
    _rounded_box(ax, in_x, in_y, in_w, in_h, r"$X_{1:t}$", facecolor=C_INPUT, fontsize=fs)

    sllm_x = in_x + in_w + 0.48
    sllm_y = in_y - 0.06
    sllm_w = 3.92
    sllm_h = 2.75

    shell = FancyBboxPatch(
        (sllm_x, sllm_y),
        sllm_w,
        sllm_h,
        boxstyle="round,pad=0.028,rounding_size=0.14",
        linewidth=1.18,
        edgecolor="#474747",
        facecolor=C_LOOP_FACE,
        zorder=1,
    )
    ax.add_patch(shell)
    ax.text(
        sllm_x + sllm_w / 2,
        sllm_y + sllm_h - 0.2,
        "sLLM  " + r"$\phi^{(s)}$",
        ha="center",
        va="top",
        fontsize=fs + 0.15,
        fontweight="bold",
        color="0.1",
        zorder=3,
    )

    emb_x = sllm_x + 0.28
    emb_w = sllm_w - 0.56
    emb_y = sllm_y + 0.48
    emb_h = 0.58
    _rounded_box(
        ax,
        emb_x,
        emb_y,
        emb_w,
        emb_h,
        "token + позиционное\n Embedding",
        facecolor="#b8cbf5",
        fontsize=fs - 1.15,
        zorder=2,
        edgecolor="#3d5080",
    )

    blk_y = emb_y + emb_h + 0.32
    blk_h = sllm_y + sllm_h - blk_y - 0.35
    _rounded_box(
        ax,
        emb_x,
        blk_y,
        emb_w,
        blk_h,
        "слои внимания и ПСП\n(стек блоков трансформера)",
        facecolor="#b8cbf5",
        fontsize=fs - 1.1,
        edgecolor="#3d5080",
        zorder=2,
    )
    _arrow(ax, (emb_x + emb_w / 2, emb_y + emb_h), (emb_x + emb_w / 2, blk_y), lw=1.1)

    lora_w, lora_h = 1.28, blk_h + emb_h + 0.45
    lora_x = sllm_x - lora_w - 0.35
    lora_y = emb_y - 0.06
    lorabox = FancyBboxPatch(
        (lora_x, lora_y),
        lora_w,
        lora_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.05,
        edgecolor="#5a5a5a",
        facecolor="#f2f2f2",
        linestyle=(0, (4.5, 3.5)),
        zorder=2,
    )
    ax.add_patch(lorabox)
    ax.text(
        lora_x + lora_w / 2,
        lora_y + lora_h / 2 + 0.15,
        "(опц.) LoRA",
        ha="center",
        va="center",
        fontsize=fs - 1.05,
        color="0.12",
        zorder=4,
        fontweight="bold",
    )
    ax.text(
        lora_x + lora_w / 2,
        lora_y + lora_h / 2 - 0.28,
        r"$W\,{+}\,\alpha\,B^{\top}A$" "\n" "малого ранга " + r"$r$",
        ha="center",
        va="center",
        fontsize=fs - 1.65,
        color="0.15",
        zorder=4,
        linespacing=1.2,
    )
    yt1 = emb_y + emb_h * 0.5
    yt2 = blk_y + blk_h * 0.55
    _arrow(ax, (lora_x + lora_w, yt1), (sllm_x + 0.06, yt1), lw=1.0, style=(0, (4.5, 3.5)))
    _arrow(ax, (lora_x + lora_w, yt2), (sllm_x + 0.06, yt2), lw=1.0, style=(0, (4.5, 3.5)))

    _arrow(ax, (in_x + in_w, in_y + in_h * 0.52), (sllm_x, in_y + in_h * 0.52), lw=1.15)

    h_w, h_h = 2.05, 0.55
    h_x = sllm_x + sllm_w + 0.45
    h_y = sllm_y + sllm_h / 2 - h_h / 2
    _rounded_box(
        ax,
        h_x,
        h_y,
        h_w,
        h_h,
        r"$h_t\in\mathbb{R}^{d_h}$",
        facecolor="#eeeeee",
        fontsize=fs - 0.2,
        edgecolor="0.45",
    )

    ax.text(
        h_x + h_w / 2,
        h_y + h_h + 0.25,
        r"последний токен / pool",
        ha="center",
        va="bottom",
        fontsize=fs - 2.05,
        color="0.4",
        style="italic",
    )

    _arrow(ax, (sllm_x + sllm_w, blk_y + blk_h * 0.72), (h_x, h_y + h_h * 0.65), lw=1.12)

    proj_w = 4.95
    proj_h = 0.92
    proj_x = cx - proj_w / 2
    proj_y = sllm_y + sllm_h + 0.65
    _rounded_box(
        ax,
        proj_x,
        proj_y,
        proj_w,
        proj_h,
        r"проектор:\ $P_{\psi}: \mathbb{R}^{d_h}\!\to\!\mathbb{R}^{d_z}$"
        "\n"
        r"$Z_{t,0}$",
        facecolor=C_ENC,
        fontsize=fs - 0.55,
        edgecolor="#805020",
        linewidth=1.08,
    )

    lc = min(h_x + h_w / 2, proj_x + proj_w - 0.65)
    _arrow(ax, (h_x + h_w / 2, h_y + h_h), (lc, proj_y + 0.02), lw=1.08, rad=0.06)

    stack_y0 = proj_y + proj_h + 0.58
    ww = W - 1.05
    xf = (W - ww) / 2
    pm = proj_x + proj_w / 2
    _arrow(ax, (pm, proj_y + proj_h), (pm, stack_y0 + 0.12), lw=1.15)

    _, ymax = draw_trm_loop_heads_stop(ax, x0=xf, y0=stack_y0, ww=ww, fontsize=fs)

    margin = 0.25
    ax.set_ylim(0, max(H, ymax + 0.45))
    ax.set_xlim(-margin, W + margin)
    plt.tight_layout(pad=0.09)
    _save_figure(fig, outfile_base, dpi=dpi, formats=formats)
    plt.close(fig)


def _save_figure(fig, outfile_base: Path, dpi: int, formats: list[str]) -> None:
    fmt_set = {f.lower().strip() for f in formats}
    if "pdf" in fmt_set:
        fig.savefig(
            outfile_base.with_suffix(".pdf"),
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.06,
            transparent=False,
        )
    if "png" in fmt_set:
        fig.savefig(
            outfile_base.with_suffix(".png"),
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.06,
            transparent=False,
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Render TRM / sLLM+TRM architecture diagrams (matplotlib).")
    p.add_argument("--out-dir", type=Path, default=Path("tex/img"))
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--formats", type=str, default="pdf,png")
    ns = p.parse_args(argv)
    out_dir: Path = ns.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fmts = [s.strip() for s in ns.formats.split(",") if s.strip()]

    draw_pure_trm_network(outfile_base=out_dir / "arch_trm_network", dpi=ns.dpi, formats=fmts)
    draw_sllm_trm_network(outfile_base=out_dir / "arch_sllm_trm_network", dpi=ns.dpi, formats=fmts)


if __name__ == "__main__":
    main()
