# Вероятностное моделирование времени остановки

- **Главный файл:** [thesis.tex](thesis.tex) (подключает [preamble.tex](preamble.tex), [references.bib](references.bib) и фрагменты в [chapters/](chapters/)).
- **Сборка из корня репозитория:** см. [COMMANDS.md](../COMMANDS.md) (`diplom-tex-build`). Итоговый PDF: `tex/build/thesis.pdf`.
- **Шрифты (XeLaTeX):** в преамбуле заданы файлы DejaVu из дерева TeX Live (`.../texmf-dist/fonts/truetype/public/dejavu/`). На Overleaf обычно достаточно заменить блок `\setmainfont{...}` на системные имена `DejaVu Serif` и т.д.
- **Литература:** `biblatex` с `backend=bibtex` (достаточно `bibtex` из дистрибутива; `biber` не обязателен). Для ручной отладки: `PAPER=a4 xelatex -output-directory=build thesis.tex`, затем из `build/` вызвать `BIBINPUTS=..: bibtex thesis`.
- **Образец оформления ВКР:** при необходимости сверки с шаблоном МГУ можно клонировать репозиторий `yes-science` (каталог `msu-thesis/`), например в `~/documents/repos/yes-science`, и сравнить титульные макросы с [BYUPhys.cls](BYUPhys.cls).

Содержание: введение (цель, задачи, новизна, обзор литературы), постановка, ARC-AGI, рекурсивная модель и SLM, семейства распределений (включая `power`), правила остановки, обучение и метрики, экспериментальный план, результаты ARC-AGI с таблицами и гистограммой \(\tau^\star\), обсуждение, заключение, приложения.
