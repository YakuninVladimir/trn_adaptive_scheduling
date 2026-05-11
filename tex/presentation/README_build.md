# Сборка презентации (`tex/presentation`)

Исходники: [`presentation.tex`](presentation.tex); числовые блоки генерируются в [`presentation_data.tex`](presentation_data.tex) из CSV таблиц диссера (WikiText и ARC совпадают с `tex/tables/`).

## 1. Синхронизация данных

Из корня репозитория:

```bash
uv run python scripts/render_presentation_data.py
```

Читает `tex/tables/wikitext_lm_base.csv`, `arc_stopping_base.csv`, `wikitext_lm_ft_compare.csv`, `arc_stopping_finetune_compare.csv`.

Опциональные параметры см. `--help`; выходной файл задаётся флагом `--out`.

## 2. Сборка PDF

Требуется полноценный TeX Live (пакеты `latexmk`, `fontspec`, `babel` с русским, `pgfplots`; шрифты DejaVu или замена в преамбуле) и предпочтительно движок `xelatex`.

```bash
uv run diplom-tex-build \
  --workdir tex/presentation \
  --main presentation.tex \
  --out-dir build
```

Результат: `tex/presentation/build/presentation.pdf`.

Альтернатива:

```bash
cd tex/presentation && latexmk -xelatex -interaction=nonstopmode -outdir=build presentation.tex
```

Замечание: среды с урезанным TeX (без `babel-russian`, без метрик шрифтов) дадут ошибки `fontspec` / `babel`; на полной установке сборка должна совпадать с исходным бандлом.

## 3. Текст доклада

Файл [`speech.md`](speech.md) — не используется при сборке PDF, хранится рядом для удобства.
