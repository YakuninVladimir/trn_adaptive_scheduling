# Сборка презентации

Требуется TeX Live с `xelatex` и `latexmk`.

```bash
latexmk -xelatex -interaction=nonstopmode presentation.tex
```

Результат сборки: `presentation.pdf`.

Файл самодостаточный: графики построены средствами `pgfplots`, внешние изображения не нужны.
