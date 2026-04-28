# Справочник команд CLI

Все entrypoints объявлены в `pyproject.toml` как `project.scripts`. Запуск из корня репозитория:

```bash
uv run <команда> [аргументы]
```

Справка по любой команде: `uv run <команда> --help`. Для подкоманд `diplom-data`: `uv run diplom-data <subcommand> --help`.

---

## Базовые команды uv

| Команда | Аргументы | Назначение |
|---------|-----------|------------|
| `uv sync --extra dev` | `--extra dev` — установить dev-группу (`pytest`, `ruff`) | Создать/обновить окружение и поставить зависимости проекта. |
| `uv lock` | нет | Пересобрать `uv.lock` по текущему `pyproject.toml`. |
| `uv add mamba-ssm causal-conv1d` | `mamba-ssm`, `causal-conv1d` — пакеты fast-path для Falcon | Установить зависимости для ускоренного выполнения Falcon (без fallback на naive путь). |
| `uv run pytest` | путь/модуль опционально | Запуск тестов в окружении проекта. |
| `uv run ruff check src tests` | пути проверки | Запуск линтера. |

---

## `diplom-data`

Подготовка и скачивание данных. Использует подкоманды.

### Общее

| Подкоманда | Назначение |
|------------|------------|
| `sudoku` | Sudoku-Extreme (Hugging Face) |
| `text` | Текстовые датасеты через Hugging Face `datasets` |
| `arc-agi` | ARC-AGI через Hugging Face `datasets` c экспортом `train.jsonl`/`val.jsonl` |
| `timeseries-stocks` | Цены бумаг через `yfinance` |
| `timeseries-public` | Временные ряды из Hugging Face |
| `timeseries-synth` | Синтетический ряд (синус) для smoke-тестов |

---

### `diplom-data sudoku`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--dry-run` | bool | `true` | Только план действий без записи на диск. Для реального выполнения: `--dry-run=false`. |
| `--download-metadata` | флаг | выкл. | Вывести список файлов в HF-репозиториях и выбранные `train_repo` / `test_repo`. |
| `--materialize` | флаг | выкл. | Скачать CSV и при необходимости собрать подвыборку с аугментацией. |
| `--out-dir` | str | `data/sudoku` | Корневой каталог для выходных файлов. |
| `--subsample` | int, опц. | — | Число записей из `train.csv` для подвыборки (нужно вместе с `--augment`). |
| `--augment` | int, опц. | — | Сколько аугментированных строк на каждую запись подвыборки (нужно вместе с `--subsample`). |

Поведение `dry_run`: если указаны `--download-metadata` или `--materialize`, фактическое значение `dry_run` берётся из переданного `--dry-run` (при `--materialize` обычно передают `--dry-run=false`).

---

### `diplom-data text`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--name` | str | **обязателен** | Имя датасета на Hugging Face (как в коде `materialize_text_dataset`). |
| `--dataset-config` | str, опц. | — | Конфигурация/поднабор HF датасета (например `wikitext-103-raw-v1`). |
| `--split` | str | `train` | Сплит для загрузки. |
| `--dry-run` | bool | `true` | План без записи; для записи: `--dry-run=false`. |
| `--materialize` | флаг | выкл. | Записать данные на диск. |
| `--out-dir` | str | `data/text` | Каталог вывода. |

---

### `diplom-data arc-agi`

Подготовка ARC-AGI в формат, который ожидает `task.name: arc_agi` (`train.jsonl` + `val.jsonl`).

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--name` | str | `lordspline/arc-agi` | Имя датасета в Hugging Face. |
| `--dataset-config` | str, опц. | — | Конфигурация/поднабор HF датасета. |
| `--split-train` | str | `training` | Сплит, экспортируемый в train jsonl. |
| `--split-val` | str | `evaluation` | Сплит, экспортируемый в val jsonl. |
| `--train-filename` | str | `train.jsonl` | Имя train-файла в `out-dir`. |
| `--val-filename` | str | `val.jsonl` | Имя val-файла в `out-dir`. |
| `--dry-run` | bool | `true` | План без записи; для записи: `--dry-run=false`. |
| `--materialize` | флаг | выкл. | Реально скачать и сохранить jsonl. |
| `--out-dir` | str | `data/arc_agi` | Каталог вывода. |

---

### `diplom-data timeseries-stocks`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--tickers` | str… | **обязателен** | Один или несколько тикеров (например `AAPL MSFT`). |
| `--period` | str | `5y` | Период для `yfinance`. |
| `--interval` | str | `1d` | Интервал свечей. |
| `--dry-run` | bool | `true` | План без записи; для записи: `--dry-run=false`. |
| `--materialize` | флаг | выкл. | Скачать и сохранить. |
| `--out-dir` | str | `data/timeseries/stocks` | Каталог вывода. |

---

### `diplom-data timeseries-public`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--name` | str | **обязателен** | Имя датасета на Hugging Face. |
| `--split` | str | `train` | Сплит. |
| `--dry-run` | bool | `true` | План без записи; для записи: `--dry-run=false`. |
| `--materialize` | флаг | выкл. | Записать данные. |
| `--out-dir` | str | `data/timeseries/public` | Каталог вывода. |

---

### `diplom-data timeseries-synth`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--length` | int | `4096` | Длина синтетического ряда. |
| `--out-dir` | str | `data/timeseries/synth` | Каталог вывода. |

Флагов `dry-run` / `materialize` нет: данные сразу материализуются на диск.

---

## `diplom-train`

Обучение по YAML-конфигу эксперимента.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--config` | str | **обязателен** | Путь к файлу конфигурации (например `configs/sudoku_trm_cgar.yaml`). |
| `--live-plots` | флаг | выкл. | Явно включить периодическое обновление `{run_dir}/plots.png` во время обучения (обычно уже включено через `train.live_plots`). |
| `--live-plot-every` | int, опц. | — | Интервал в глобальных шагах между обновлениями графика; если задан, переопределяет `train.live_plot_every` из YAML. |

Остальные гиперпараметры задаются в YAML: секции `task`, `model`, `scheduler`, `train`. Поля секции `train` соответствуют `TrainConfig` в `src/diplom/runner/config.py`:

| Ключ YAML (`train.*`) | Описание |
|----------------------|----------|
| `seed` | Сид воспроизводимости. |
| `device` | `auto`, `cpu` или `cuda`. |
| `epochs` | Число эпох. |
| `batch_size` | Размер батча. |
| `lr` | Базовый learning rate (AdamW). |
| `weight_decay` | Weight decay оптимизатора. |
| `warmup_steps` | Линейный прогрев: множитель LR от `step/warmup_steps` до `1` (0 = без прогрева). |
| `lr_schedule` | После прогрева: `none` — постоянный LR; `cosine` — косинусное снижение к `lr * lr_min_ratio` к шагу `max_steps`. |
| `lr_min_ratio` | Нижняя граница LR относительно базового при `lr_schedule: cosine` (в конце расписания). |
| `max_steps` | Остановка по числу глобальных шагов (опционально). |
| `log_every` | Частота логирования в `metrics.jsonl`. |
| `eval_every` | Частота валидации. |
| `ckpt_every` | Частота сохранения чекпоинтов в `{run_dir}/checkpoints/` (используется, когда `save_best_only: false`). |
| `save_best_only` | Если `true`, сохраняется только `checkpoints/best.pt` при улучшении выбранной метрики. |
| `best_metric` | Метрика для выбора лучшего чекпоинта: `auto`, `val_loss`, `train_loss` или имя метрики. |
| `best_metric_mode` | Режим оптимизации метрики: `auto`, `min`, `max`. |
| `run_dir` | Каталог прогона (метрики, графики, чекпоинты). |
| `beta_halt` | Вес BCE для halting head (если включён). |
| `use_halt_loss` | Включать ли расчёт/логирование halt-loss (BCE). Для текущих oracle sweep поставлено `false`. |
| `progress_bar` | Включить tqdm. |
| `live_plots` | Live-графики без CLI-флага (по умолчанию включены). |
| `live_plot_every` | Интервал шагов для live-графиков (если не переопределён CLI). |
| `dump_oracle_trace` | Сохранять трассы для оффлайн-обучения оракула: `aux_seq` [T,B,D], `per_sample_ce` [T,B], `x_tokens`, `y`, маски/веса, расписание по шагам, опционально логиты и латенты (y,z). Файлы — **шарды** `oracle_trace_shard_00000.pt` (см. `dump_oracle_trace_shard_batches`). |
| `dump_oracle_trace_dir` | Подкаталог или абсолютный путь; по умолчанию `oracle_traces` внутри `run_dir`. |
| `dump_oracle_trace_every` | Писать батч в буфер каждые N глобальных шагов (`1` = каждый батч подряд). |
| `dump_oracle_trace_shard_batches` | Сколько батчей склеить в один `.pt` (меньше файлов и быстрее последовательное чтение). |
| `dump_oracle_trace_max_batches` | Остановить **запись батчей** после стольких записей всего (`null` = без лимита). |
| `dump_oracle_trace_include_logits` | Включать `logits_seq` [T,B,L,V]. |
| `dump_oracle_trace_include_state` | Включать `state_y_seq`, `state_z_seq` [T,B,L,D] на шаг. |
| `dump_oracle_trace_fp16` | Сохранять float-тензоры на CPU в float16 (экономия места). |
| `amp` | Только CUDA: mixed precision (autocast + GradScaler) для основного и oracle-оптимизаторов. |
| `amp_dtype` | `float16` или `bfloat16` (bf16 только при поддержке GPU). |

Дополнительно для `task.name: sudoku` (`SudokuTaskConfig`):

| Ключ YAML (`task.*`) | По умолчанию | Описание |
|---------------------|--------------|----------|
| `loss_on_empty_cells_only` | `true` | Кросс-энтропия только по пустым клеткам пазла (`.` / токен `0`). |
| `hole_difficulty_reweight` | `true` | Веса по эвристике: число допустимых цифр 1..9 в клетке с учётом только подсказок из пазла (без внешних солверов). |
| `difficulty_power` | `1.0` | Степень для веса: вес ∝ `candidate_count ** power`, затем нормировка к среднему 1 по дыркам. |

`token_acc` считается по тем же пустым клеткам; `exact_acc` — полное совпадение сетки 81 ячейка с решением.

---

### Готовые конфиги sweep по oracle-аппроксимациям (WikiText-103)

Все конфиги лежат в `configs/oracle_sweep_wikitext/` и отличаются только `model.oracle_target_mode`/`model.oracle_distribution_model`.
Во всех sweep-конфигах выставлено: `train.epochs: 5`, `train.live_plots: true`, `train.live_plot_every: 10`.

| Конфиг | Что обучается |
|--------|----------------|
| `text_wikitext103_trm_oracle_delta.yaml` | Legacy delta-oracle (`oracle_target_mode: delta`) |
| `text_wikitext103_trm_oracle_finite_discrete.yaml` | Distribution oracle: finite discrete |
| `text_wikitext103_trm_oracle_smoothed_loss.yaml` | Distribution oracle: smoothed target |
| `text_wikitext103_trm_oracle_mixture_geometric.yaml` | Distribution oracle: mixture geometric |
| `text_wikitext103_trm_oracle_mixture_exponential.yaml` | Distribution oracle: mixture exponential |
| `text_wikitext103_trm_oracle_power.yaml` | Distribution oracle: power-law tail |
| `text_wikitext103_trm_oracle_negative_binomial.yaml` | Distribution oracle: negative binomial |
| `text_wikitext103_trm_oracle_lognormal.yaml` | Distribution oracle: discretized lognormal |
| `text_wikitext103_trm_oracle_hybrid.yaml` | Distribution oracle: hybrid finite+tail |

Команды запуска:

```bash
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_delta.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_finite_discrete.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_smoothed_loss.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_mixture_geometric.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_mixture_exponential.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_power.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_negative_binomial.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_lognormal.yaml
uv run diplom-train --config configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_hybrid.yaml
```

---

### Готовые конфиги sweep по oracle-аппроксимациям (ARC-AGI)

Все конфиги лежат в `configs/oracle_sweep_arc_agi/` и отличаются только `model.oracle_target_mode`/`model.oracle_distribution_model`.
Во всех sweep-конфигах выставлено: `train.epochs: 10`, `train.live_plots: true`, `train.live_plot_every: 10`.

| Конфиг | Что обучается |
|--------|----------------|
| `arc_agi_trm_oracle_delta.yaml` | Legacy delta-oracle (`oracle_target_mode: delta`) |
| `arc_agi_trm_oracle_finite_discrete.yaml` | Distribution oracle: finite discrete |
| `arc_agi_trm_oracle_smoothed_loss.yaml` | Distribution oracle: smoothed target |
| `arc_agi_trm_oracle_mixture_geometric.yaml` | Distribution oracle: mixture geometric |
| `arc_agi_trm_oracle_mixture_exponential.yaml` | Distribution oracle: mixture exponential |
| `arc_agi_trm_oracle_power.yaml` | Distribution oracle: power-law tail |
| `arc_agi_trm_oracle_negative_binomial.yaml` | Distribution oracle: negative binomial |
| `arc_agi_trm_oracle_lognormal.yaml` | Distribution oracle: discretized lognormal |
| `arc_agi_trm_oracle_hybrid.yaml` | Distribution oracle: hybrid finite+tail |

Команды запуска:

```bash
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_delta.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_finite_discrete.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_smoothed_loss.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_geometric.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_mixture_exponential.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_power.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_negative_binomial.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_lognormal.yaml
uv run diplom-train --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_hybrid.yaml
```

---

### Скрипт запуска sweep целиком

Файл: `scripts/run_oracle_sweep.sh`

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `group` | `all` | Какие наборы запускать: `wikitext`, `arc`, `all`. |
| `--dry-run` | выкл. | Показать команды, не запускать обучение. |
| `--continue-on-error` | выкл. | Продолжать sweep, даже если один конфиг упал. |

Примеры:

```bash
# Показать весь план запусков
./scripts/run_oracle_sweep.sh all --dry-run

# Запустить только ARC-AGI sweep
./scripts/run_oracle_sweep.sh arc

# Запустить всё и не останавливаться на первой ошибке
./scripts/run_oracle_sweep.sh all --continue-on-error
```

---

## `diplom-validate`

Валидация по конфигу; опциональная загрузка весов.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--config` | str | **обязателен** | Путь к YAML эксперимента. |
| `--checkpoint` | str, опц. | — | Путь к `.pt` чекпоинту (`state_dict` под ключом `model`). |
| `--oracle-policy` | choice | `none` | Одно из: `none`, `greedy`, `sampling` — политика инференса для `TRMOracle` (для обычного `TRM` без эффекта). |
| `--oracle-max-steps` | int, опц. | — | Верхняя граница шагов рассуждения при oracle-политике (иначе берётся `N_sup` из модели). |
| `--oracle-temperature` | float | `1.0` | Температура softmax при `--oracle-policy sampling`. |

---

## `diplom-eval-stopping`

Валидация распределительных oracle-моделей и стратегий остановки на одном validation rollout.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--config` | str | **обязателен** | Путь к YAML эксперимента (обычно с `model.name: trm_oracle`). |
| `--checkpoint` | str, опц. | — | Путь к `.pt` чекпоинту. |
| `--distribution-models` | str | `finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid` | CSV-список распределительных моделей oracle. |
| `--strategies` | str | `cumulative_probability,future_improvement,hazard,quantile,budget` | CSV-список стратегий остановки. |
| `--threshold-grid` | str | `0.5,0.6,0.7,0.8,0.9` | CSV-пороги для стратегий, использующих threshold. |
| `--budget-grid` | str | `2,4,6,8` | CSV-бюджеты (`E[tau] <= C`) для budget-правила. |
| `--max-steps` | int, опц. | — | Переопределить `N_sup` в eval. |
| `--out` | str, опц. | — | Путь к итоговому JSON; если не задано, пишется в `{run_dir}/stopping_eval.json`. |

Выход: агрегированные метрики по всем комбинациям `distribution_model × strategy × threshold/budget`, включая `mean_steps`, `mean_regret`, `nll`, `brier`, `ece` и task-метрики.

---

## `diplom-tex-build`

Сборка LaTeX-проекта диплома (предпочтительно через `latexmk`, с fallback на прямые прогоны `xelatex`/`lualatex`/`pdflatex`).

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--workdir` | str | `tex` | Каталог LaTeX-проекта. |
| `--main` | str | `thesis.tex` | Главный `.tex` файл. |
| `--out-dir` | str | `build` | Подкаталог для артефактов сборки внутри `workdir`. |
| `--engine` | choice | `xelatex` | Движок: `xelatex`, `lualatex`, `pdflatex`. |
| `--passes` | int | `2` | Число fallback-прогонов движка, если `latexmk` не установлен. |
| `--clean` | флаг | выкл. | Перед сборкой выполнить очистку вспомогательных файлов (`latexmk -c`). |

Поведение:
- при наличии `latexmk` используется он (`-xelatex`/`-lualatex`/`-pdf`);
- при отсутствии `latexmk` выполняется заданное число прогонов выбранного движка.

---

## `diplom-plot`

Построение PNG по `metrics.jsonl` из каталога прогона.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--run-dir` | str | **обязателен** | Каталог с `metrics.jsonl` (например `runs/trm_1h`). |
| `--out` | str, опц. | — | Путь к выходному PNG; если не задан — `{run-dir}/plots.png`. |
| `--no-ema-band` | флаг | выкл. | Отключить EMA и полосу ±σ по скользящему std. |
| `--ema-alpha` | float | `0.06` | Коэффициент EMA (больше — быстрее следует за сырым рядом). |
| `--std-window` | int | `25` | Окно для оценки локального std вокруг каждой точки. |
| `--std-sigma` | float | `1.0` | Полуширина заливки: `EMA ± std_sigma * rolling_std`. |

На графиках по умолчанию: полупрозрачный **сырой** ряд, поверх — **EMA**, полупрозрачная зона **±σ** (скользящее стандартное отклонение сырого ряда). В train-панелях отображаются лоссы и `lr`; `used_sup` отдельно не рисуется.

---

## Примеры

```bash
# Данные Sudoku: реальное скачивание и сборка 1k × 1000 аугментаций
uv run diplom-data sudoku --dry-run false --materialize --subsample 1000 --augment 1000

# Обучение TRM с live-графиками
uv run diplom-train --config configs/sudoku_trm_1h.yaml --live-plots --live-plot-every 20

# Валидация TRMOracle с greedy oracle
uv run diplom-validate --config configs/sudoku_trm_oracle_1h.yaml \
  --checkpoint runs/trm_oracle_1h/checkpoints/step_26000.pt \
  --oracle-policy greedy --oracle-max-steps 32

# График после прогона
uv run diplom-plot --run-dir runs/dev_trm_cgar_sudoku

# WikiText-103 materialization
uv run diplom-data text --name wikitext --dataset-config wikitext-103-raw-v1 --split train --dry-run false --materialize

# ARC-AGI materialization (готовит data/arc_agi/train.jsonl и data/arc_agi/val.jsonl)
uv run diplom-data arc-agi --name lordspline/arc-agi --split-train training --split-val evaluation --dry-run false --materialize

# TRM + Oracle на ARC-AGI
uv run diplom-train --config configs/arc_agi_trm_oracle.yaml

# TRM + Oracle на WikiText-103
uv run diplom-train --config configs/text_wikitext103_trm_oracle.yaml

# TRM без oracle с фиксированным числом итераций
uv run diplom-train --config configs/text_wikitext103_trm_fixed.yaml

# Frozen LLM embeddings -> TRM correction module
uv run diplom-train --config configs/text_qwen_correction_trm.yaml

# LoRA ablation
uv run diplom-train --config configs/text_qwen_lora_ablation.yaml

# Сравнение всех стратегий остановки для заданной oracle-модели
uv run diplom-eval-stopping --config configs/text_wikitext103_trm_oracle.yaml \
  --checkpoint runs/text_wikitext103_trm_oracle/checkpoints/step_1000.pt \
  --distribution-models finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid \
  --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
  --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
  --budget-grid 2,4,6,8 \
  --out runs/text_wikitext103_trm_oracle/stopping_eval.json

# Сборка LaTeX-диплома
uv run diplom-tex-build --workdir tex --main thesis.tex --out-dir build --engine xelatex --passes 2
```
