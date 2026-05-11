## diplom

Исследовательский каркас для моделей **HRM/TRM** и планировщиков глубины (scheduler) в стиле **CGAR**.

### Быстрый старт (pyenv + uv)

```bash
cd /home/vovochka/diplom

# 1) Python (pyenv)
pyenv install -s 3.11.8
pyenv local 3.11.8

# 2) uv: создать/синхронизировать окружение и lock
uv sync --extra dev

# 3) Проверка CLI
uv run diplom-data --help
uv run diplom-train --help
uv run diplom-validate --help
uv run diplom-plot --help
uv run diplom-eval-stopping --help
```

### Данные (по умолчанию без скачивания)

Все команды `diplom-data` по умолчанию работают в **dry-run** режиме и не тянут большие датасеты.

Примеры:

```bash
# Sudoku-Extreme (HuggingFace) - показать, что будет сделано
uv run diplom-data sudoku

# Текстовый датасет (HuggingFace)
uv run diplom-data text --name ag_news

# WikiText-103 (raw)
uv run diplom-data text --name wikitext --dataset-config wikitext-103-raw-v1 --split train

# ARC-AGI (jsonl for arc_agi task)
uv run diplom-data arc-agi --name lordspline/arc-agi --split-train training --split-val evaluation --dry-run false --materialize

# Временной ряд акций (yfinance)
uv run diplom-data timeseries-stocks --tickers AAPL MSFT
```

### Обучение (по конфигу)

```bash
uv run diplom-train --config configs/sudoku_trm_cgar.yaml

# live-обновление графиков во время обучения
uv run diplom-train --config configs/sudoku_trm_oracle_10min.yaml --live-plots --live-plot-every 20
```

### Быстрый прогон (~10 минут)

```bash
# 1) Маленький датасет
uv run diplom-data sudoku --dry-run false --materialize --subsample 100 --augment 10

# 2) Быстрое обучение TRM+Oracle
uv run diplom-train --config configs/sudoku_trm_oracle_10min.yaml
```

### Валидация / Oracle inference policies

```bash
# Жадная политика (delta=0 тоже доступен: "остановиться сейчас")
uv run diplom-validate --config configs/sudoku_trm_oracle_10min.yaml --oracle-policy greedy --oracle-max-steps 32

# Семплирование из распределения oracle (итеративно: sampled delta -> пройти delta шагов -> заново oracle)
uv run diplom-validate --config configs/sudoku_trm_oracle_10min.yaml --oracle-policy sampling --oracle-max-steps 32 --oracle-temperature 1.0
```

### Новые сценарии экспериментов

```bash
# TRM + oracle на ARC-AGI
uv run diplom-train --config configs/arc_agi_trm_oracle.yaml

# TRM + oracle на WikiText-103
uv run diplom-train --config configs/text_wikitext103_trm_oracle.yaml

# TRM без oracle с фиксированными итерациями
uv run diplom-train --config configs/text_wikitext103_trm_fixed.yaml

# Frozen Falcon-H1-Tiny -> TRM correction module (`task.train_fraction: 0.1`)
uv run diplom-train --config configs/text_qwen_correction_trm.yaml

# LoRA ablation (Falcon; `task.train_fraction: 0.1`)
uv run diplom-train --config configs/text_qwen_lora_ablation.yaml
```

### Валидация распределений и стратегий остановки

```bash
uv run diplom-eval-stopping \
  --config configs/text_wikitext103_trm_oracle.yaml \
  --checkpoint runs/text_wikitext103_trm_oracle/checkpoints/step_1000.pt \
  --distribution-models finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid \
  --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
  --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
  --budget-grid 2,4,6,8
```

### Сборка LaTeX

```bash
uv run diplom-tex-build --workdir tex --main thesis.tex --out-dir build --engine xelatex --passes 2
```

