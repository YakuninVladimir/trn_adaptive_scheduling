## diplom

Исследовательский каркас для моделей **HRM/TRM** и планировщиков глубины (scheduler) в стиле **CGAR**.

### Быстрый старт (pyenv + poetry)

```bash
cd /home/vovochka/diplom

# 1) Python (pyenv)
pyenv install -s 3.11.8
pyenv local 3.11.8

# 2) Poetry: привязать env к pyenv python
poetry env use "$(pyenv which python)"

# 3) Poetry deps
poetry install

# 4) Проверка CLI
poetry run diplom-data --help
poetry run diplom-train --help
poetry run diplom-validate --help
poetry run diplom-plot --help
```

### Данные (по умолчанию без скачивания)

Все команды `diplom-data` по умолчанию работают в **dry-run** режиме и не тянут большие датасеты.

Примеры:

```bash
# Sudoku-Extreme (HuggingFace) - показать, что будет сделано
poetry run diplom-data sudoku

# Текстовый датасет (HuggingFace)
poetry run diplom-data text --name ag_news

# Временной ряд акций (yfinance)
poetry run diplom-data timeseries-stocks --tickers AAPL MSFT
```

### Обучение (по конфигу)

```bash
poetry run diplom-train --config configs/sudoku_trm_cgar.yaml
```

