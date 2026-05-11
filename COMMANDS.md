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
| `uv add mamba-ssm causal-conv1d` | `mamba-ssm`, `causal-conv1d` — опциональные fast-path пакеты | Установить зависимости для ускорённого выполнения моделей на Mamba (без fallback на naive путь), если конфиг их использует. |
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
| `--init-checkpoint` | str, опц. | — | Путь к чекпоинту, который нужно загрузить перед началом обучения (`model` из `.pt`). |
| `--oracle-only` | флаг | выкл. | Заморозить backbone и дообучать только параметры oracle-head (актуально для `TRMOracle`). |
| `--live-plots` | флаг | выкл. | Явно включить периодическое обновление `{run_dir}/plots.png` во время обучения (обычно уже включено через `train.live_plots`). |
| `--live-plot-every` | int, опц. | — | Интервал в глобальных шагах между обновлениями графика; если задан, переопределяет `train.live_plot_every` из YAML. |

Для `model.name: trm_oracle` и `oracle_target_mode: distribution` цель для распределения \(\tau^\*\) на train задаётся по траектории **маскированной потокенной accuracy** на каждом supervision-шаге: \(\tau^\*=\mathrm{argmax}_t \mathrm{token\_acc}(t)\) (равенство по accuracy разрешается детерминированно первым максимумом). Поле `oracle_distribution_lambda` в этом режиме **не входит** в определение \(\tau^\*\) при наличии траектории accuracy; оно используется только для legacy-режима воспроизведения из трасс, где передан один только `per_sample_ce` (см. `oracle_loss_from_rollout(..., per_step_acc=None)`).

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

Все конфиги лежат в `configs/oracle_sweep_wikitext/` и отличаются только `model.oracle_target_mode` / `model.oracle_distribution_model`.

Общая настройка sweep (обновлено): **`model.name: frozen_llm_trm_oracle`** — замороженный **Falcon-H1-Tiny** (`tiiuae/Falcon-H1-Tiny-90M-Base`) даёт `last_hidden_state`, проецируемый в `d_model`; дальше обычный TRM + oracle-головы. Токенизатор тот же Falcon, `vocab_size: 32768`, `pos_encoding: none`. Для WikiText в **`task`** задано **`train_fraction: 0.1`** (на обучение берётся доля 10% от train split). Прогоны пишутся в **`runs/oracle_sweep_wikitext_falcon/<вариант>/`** (старые каталоги `runs/oracle_sweep_wikitext/` без `_falcon` относились к прежнему `trm_oracle` + DistilBERT). В sweep выставлено **`train.batch_size: 16`** (при нехватке VRAM уменьши локально).

Во всех sweep-конфигах: `train.epochs: 5`, `train.live_plots: true`, `train.live_plot_every: 10`.

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

**WikiText + TRM Oracle `finite_discrete` + токенизатор Falcon** (`tiiuae/Falcon-H1-Tiny-90M-Base`, `vocab_size: 32768`): модель по-прежнему обучается с нуля как `TRMOracle`, веса чекпоинта Falcon не подмешиваются (в отличие от `trm_correction`). В **`task`** задано **`train_fraction: 0.1`**.

```bash
uv run diplom-train --config configs/text_falcon_trm_oracle_finite_discrete.yaml
```

**Тот же oracle (`finite_discrete`), но вход рекурсии TRM — эмбеды Falcon:** класс `frozen_llm_trm_oracle`, \(x = \mathrm{Linear}(\texttt{last\_hidden\_state})\), backbone по умолчанию заморожен; слой `backbone_in_proj` и TRM+oracle обучаются. В **`task`** задано **`train_fraction: 0.1`**.

```bash
uv run diplom-train --config configs/text_falcon_frozen_llm_trm_oracle_finite_discrete.yaml
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

### Скрипт валидации sweep целиком

Файл: `scripts/validate_oracle_sweep.sh`

Что делает для каждого конфига:
1. запускает `diplom-validate` c `--checkpoint <run_dir>/checkpoints/best.pt --oracle-policy greedy --oracle-max-steps 8 --progress-bar true`;
2. запускает `diplom-eval-stopping` c `--distribution-models <model.oracle_distribution_model>` (по умолчанию только модель из конфига), `--honest-split-ratio <ratio> --selection-metric token_acc --selection-mode max --answer-policies last,argmax_interval --progress-bar true` и сохраняет `stopping_eval.json` в тот же `run_dir`.
3. если структура чекпоинта не совпадает с текущим YAML (legacy `oracle_use_full_y`/`halt_head`), автоматически создаёт временный совместимый конфиг и использует его только для текущего запуска.

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `group` | `all` | Какие наборы валидировать: `wikitext`, `arc`, `all`. |
| `--dry-run` | выкл. | Показать команды, не запускать валидацию. |
| `--continue-on-error` | выкл. | Продолжать sweep-валидацию, даже если один конфиг упал. |
| `--all-distributions` | выкл. | Вместо модели из YAML прогонять полный список распределений (`finite_discrete,...,hybrid`). |
| `--honest-split-ratio=<0..1>` | `0.0` | Holdout-режим: доля валидационной выборки, отведённой под **подбор** threshold/strategy (например `0.5` = 50/50 подбор/итоговая оценка). |
| `--batch-multiplier=<float>` | `2.0` | Множитель `train.batch_size` только на время валидации sweep (через временный совместимый YAML). |
| `--val-reduce-factor=<float>` | `5.0` | Во сколько раз уменьшить объём val-подмножества только на время валидации (`task.val_fraction /= factor`; если `val_fraction` не задан — ставится `1/factor`; `max_val_samples` тоже делится на factor). |
| `--checkpoint-subdir=<path>` | пусто | Читать чекпоинт и писать `stopping_eval.json` внутри подкаталога `train.run_dir/<path>` (например `oracle_finetune`). |

Примеры:

```bash
# Показать план валидации
./scripts/validate_oracle_sweep.sh all --dry-run

# Провалидировать только ARC-AGI
./scripts/validate_oracle_sweep.sh arc

# Провалидировать всё, продолжая при ошибках
./scripts/validate_oracle_sweep.sh all --continue-on-error

# Провалидировать и прогнать все distribution-модели для каждого чекпоинта
./scripts/validate_oracle_sweep.sh all --all-distributions

# Честная оценка: подбор порога на первой части val, отчёт на оставшейся части
./scripts/validate_oracle_sweep.sh arc --continue-on-error --honest-split-ratio=0.5

# Быстрее/дешевле валидация: batch x2 и val в 5 раз меньше (значения по умолчанию)
./scripts/validate_oracle_sweep.sh wikitext --continue-on-error --batch-multiplier=2 --val-reduce-factor=5

# Валидация зафайнтюненных чекпоинтов (run_dir/oracle_finetune/checkpoints/best.pt)
./scripts/validate_oracle_sweep.sh wikitext --continue-on-error --checkpoint-subdir=oracle_finetune

# То же + все distribution-модели в eval-stopping + honest split
./scripts/validate_oracle_sweep.sh wikitext \
  --continue-on-error \
  --checkpoint-subdir=oracle_finetune \
  --all-distributions \
  --honest-split-ratio=0.5

# Диагностика скрипта: проверка bash-синтаксиса и dry-run ARC-валидации
bash -n scripts/validate_oracle_sweep.sh && ./scripts/validate_oracle_sweep.sh arc --dry-run --continue-on-error
```

---

### Скрипт oracle-only дообучения sweep целиком

Файл: `scripts/finetune_oracle_sweep.sh`

Что делает для каждого конфига:

1. берёт чекпоинт для `--init-checkpoint` — либо **общий** (см. ниже), либо по умолчанию `best.pt` из `train.run_dir/checkpoints/` того же варианта sweep;
2. создаёт временный конфиг с новым `train.run_dir = <исходный_run_dir>/oracle_finetune` (чтобы не перезаписать полный прогон);
3. запускает `diplom-train --init-checkpoint <…> --oracle-only`.

**Один общий чекпоинт для всех типов распределений.** У `TRMOracle` / `frozen_llm_trm_oracle` все параметрические головы распределений уже есть в одном `state_dict`; в YAML меняется только `oracle_distribution_model` и соответствующий шаг потерь. Если базовая модель уже обучена в `runs/oracle_sweep_wikitext_falcon/finite_discrete/`, можно дообучить только oracle-слои под остальные конфиги sweep, не трогая TRM и замороженный Falcon:

```bash
./scripts/finetune_oracle_sweep.sh wikitext \
  --init-checkpoint runs/oracle_sweep_wikitext_falcon/finite_discrete/checkpoints/best.pt
```

Эквивалент через переменную окружения:

```bash
ORACLE_FT_INIT_CKPT=runs/oracle_sweep_wikitext_falcon/finite_discrete/checkpoints/best.pt \
  ./scripts/finetune_oracle_sweep.sh wikitext
```

Режим `--oracle-only`: прямой проход TRM выполняется без градиентов; градиенты идут только в `oracle_head` и линейные головы `dist_*` (см. `TRMOracle.oracle_parameters()`). Замораживаются в том числе `backbone_in_proj`, ядро TRM и HF-backbone.

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `group` | `all` | Какие наборы запускать: `wikitext`, `arc`, `all`. |
| `--dry-run` | выкл. | Показать команды, не запускать дообучение. |
| `--continue-on-error` | выкл. | Продолжать sweep, даже если один конфиг упал. |
| `--init-checkpoint PATH` | — | Один и тот же `.pt` для **каждого** конфига группы; перекрывает `ORACLE_FT_INIT_CKPT`. |
| `ORACLE_FT_INIT_CKPT` | пусто | То же, что `--init-checkpoint`, если флаг не передан. |

Примеры:

```bash
# Показать план oracle-only дообучения (общий чекпоинт с finite_discrete)
./scripts/finetune_oracle_sweep.sh wikitext \
  --init-checkpoint runs/oracle_sweep_wikitext_falcon/finite_discrete/checkpoints/best.pt \
  --dry-run

# Дообучить только ARC-AGI oracle-head (свой best.pt на каждый конфиг, как раньше)
./scripts/finetune_oracle_sweep.sh arc

# Дообучить всё, продолжая при ошибках
./scripts/finetune_oracle_sweep.sh all --continue-on-error
```

---

### Валидация oracle-only fine-tune чекпоинтов

После `scripts/finetune_oracle_sweep.sh` чекпоинты лежат в:
`<base_run_dir>/oracle_finetune/checkpoints/best.pt`.

#### Одна модель (пример: ARC finite_discrete)

```bash
uv run diplom-validate \
  --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_finite_discrete.yaml \
  --checkpoint runs/oracle_sweep_arc_agi/finite_discrete/oracle_finetune/checkpoints/best.pt \
  --oracle-policy greedy \
  --oracle-max-steps 8 \
  --progress-bar true

uv run diplom-eval-stopping \
  --config configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_finite_discrete.yaml \
  --checkpoint runs/oracle_sweep_arc_agi/finite_discrete/oracle_finetune/checkpoints/best.pt \
  --distribution-models finite_discrete \
  --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
  --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
  --budget-grid 2,4,6,8 \
  --honest-split-ratio 0.5 \
  --selection-metric token_acc \
  --selection-mode max \
  --answer-policies last,argmax_interval \
  --progress-bar true \
  --out runs/oracle_sweep_arc_agi/finite_discrete/oracle_finetune/stopping_eval.json
```

#### Все ARC fine-tune модели (batch-скрипт в shell)

```bash
for cfg in configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_*.yaml; do
  run_dir="$(uv run python - <<PY
import yaml
cfg = yaml.safe_load(open('${cfg}', 'r', encoding='utf-8'))
print(cfg.get('train', {}).get('run_dir', ''))
PY
)"
  dist="$(uv run python - <<PY
import yaml
cfg = yaml.safe_load(open('${cfg}', 'r', encoding='utf-8'))
m = cfg.get('model', {}) or {}
mode = str(m.get('oracle_target_mode', 'delta')).lower()
d = str(m.get('oracle_distribution_model', 'finite_discrete'))
print('finite_discrete' if mode == 'delta' else d)
PY
)"
  ckpt="${run_dir}/oracle_finetune/checkpoints/best.pt"
  out="${run_dir}/oracle_finetune/stopping_eval.json"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[oracle-ft-validate] missing checkpoint: ${ckpt}" >&2
    continue
  fi
  uv run diplom-validate --config "${cfg}" --checkpoint "${ckpt}" --oracle-policy greedy --oracle-max-steps 8 --progress-bar true
  uv run diplom-eval-stopping \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --distribution-models "${dist}" \
    --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
    --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
    --budget-grid 2,4,6,8 \
    --honest-split-ratio 0.5 \
    --selection-metric token_acc \
    --selection-mode max \
    --answer-policies last,argmax_interval \
    --progress-bar true \
    --out "${out}"
done
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
| `--progress-bar` | bool, опц. | `null` | Включить/выключить progress bar валидации. Если не задано, берётся `train.progress_bar` из YAML. |

Для текстовых задач (`task.name: wikitext_lm` / `text_lm`) в выводе валидации теперь считаются обе метрики:
- `token_acc` — top-1 точность по токенам;
- `top5_acc` — доля токенов, где правильный класс попал в top-5 логитов.

Пример валидации WikiText-103 (фиксированный TRM):

```bash
uv run diplom-validate \
  --config configs/text_wikitext103_trm_fixed.yaml \
  --checkpoint runs/text_wikitext103_trm_fixed/checkpoints/best.pt \
  --oracle-policy none \
  --progress-bar true
```

Пример валидации WikiText-103 (TRMOracle, greedy policy):

```bash
uv run diplom-validate \
  --config configs/text_wikitext103_trm_oracle.yaml \
  --checkpoint runs/text_wikitext103_trm_oracle/checkpoints/best.pt \
  --oracle-policy greedy \
  --oracle-max-steps 8 \
  --oracle-temperature 1.0 \
  --progress-bar true
```

Валидация **всех зафайнтюненных распределений sweep** (чекпоинты в `oracle_finetune/checkpoints/best.pt`, WikiText):

```bash
for cfg in configs/oracle_sweep_wikitext/text_wikitext103_trm_oracle_*.yaml; do
  run_dir="$(uv run python - <<PY
import yaml
cfg = yaml.safe_load(open('${cfg}', 'r', encoding='utf-8'))
print(cfg.get('train', {}).get('run_dir', ''))
PY
)"
  ckpt="${run_dir}/oracle_finetune/checkpoints/best.pt"
  out="${run_dir}/oracle_finetune/stopping_eval.json"
  [[ -f "${ckpt}" ]] || { echo "[oracle-ft-validate] missing checkpoint: ${ckpt}" >&2; continue; }
  uv run diplom-validate \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --oracle-policy greedy \
    --oracle-max-steps 8 \
    --oracle-temperature 1.0 \
    --progress-bar true
  uv run diplom-eval-stopping \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --distribution-models finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid \
    --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
    --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
    --budget-grid 2,4,6,8 \
    --honest-split-ratio 0.5 \
    --selection-metric token_acc \
    --selection-mode max \
    --answer-policies last,argmax_interval \
    --progress-bar true \
    --out "${out}"
done
```

Сводка по зафайнтюненным WikiText-run'ам:

```bash
uv run python scripts/summarize_stopping_eval.py \
  --group wikitext \
  --metric token_acc_best_of_policies \
  --mode max \
  --run-subdir oracle_finetune \
  --eval-filename stopping_eval.json \
  --out-csv runs/oracle_sweep_wikitext_falcon/oracle_finetune/stopping_summary.csv \
  --out-md runs/oracle_sweep_wikitext_falcon/oracle_finetune/stopping_summary.md
```

Обновить CSV для TeX (глава NLP):

```bash
# Рекомендуемый путь: сводка → wikitext_lm_base.csv + wikitext_lm_ft_compare.csv
# (колонка top5best, сравнение до/после oracle_finetune; для «до'' без stopping_eval.json в корне прогона — из предыдущего CSV)
uv run python scripts/export_wikitext_lm_tex_tables.py \
  --finetune-summary runs/oracle_sweep_wikitext_falcon/oracle_finetune/stopping_summary.csv \
  --tables-dir tex/tables \
  --legacy-ft-csv tex/tables/wikitext_lm_ft_compare.csv
```

Альтернатива вручную через `pandas` (только базовая таблица, без FT-сравнения):

```bash
uv run python - <<'PY'
import pandas as pd
src = pd.read_csv("runs/oracle_sweep_wikitext_falcon/oracle_finetune/stopping_summary.csv")
tok = src["best_token_acc_best_of_policies"].fillna(src["best_token_acc"])
top5 = src["best_top5_acc_best_of_policies"].fillna(src.get("best_top5_acc", pd.Series([pd.NA]*len(src))))
out = pd.DataFrame({
    "family": src["trained_distribution_model"],
    "strategy": src["best_strategy"],
    "threshold": src["best_threshold"],
    "meantau": src["best_mean_steps"],
    "tokbest": tok,
    "top5best": top5,
    "nlltau": src["best_nll"],
})
out.to_csv("tex/tables/wikitext_lm_base.csv", index=False)
print("written: tex/tables/wikitext_lm_base.csv")
PY
```

Рисунки «качество--стоимость'' и гистограмма \(\tau^\star\) для WikiText:

```bash
uv run python scripts/render_wikitext_figs.py --img-dir tex/img --tables-dir tex/tables --dpi 200
```

Три панели динамики базового обучения в `tex/img/`: основная потеря; точность train/val на **одной** оси; валидационный loss на **отдельной** оси (без двойной шкалы). Для ARC по умолчанию из графиков убираются ряды с `exact_acc`; флаг `--plot-exact-acc` возвращает их.

```bash
uv run diplom-plot --run-dir runs/oracle_sweep_wikitext_falcon/finite_discrete \
  --out-loss tex/img/wikitext_trm_train_loss.png \
  --out-val-acc tex/img/wikitext_trm_train_val_acc.png \
  --out-val-loss tex/img/wikitext_trm_train_val_loss.png \
  --dpi 200
uv run diplom-plot --run-dir runs/oracle_sweep_arc_agi/delta \
  --out-loss tex/img/arc_trm_train_loss.png \
  --out-val-acc tex/img/arc_trm_train_val_acc.png \
  --out-val-loss tex/img/arc_trm_train_val_loss.png \
  --dpi 200
```

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
| `--honest-split-ratio` | float | `0.0` | Доля val, отведённая под подбор threshold/strategy без утечки (`0` отключает holdout). |
| `--selection-metric` | str | `token_acc` | Метрика выбора лучшей конфигурации на части val, зарезервированной под подбор. |
| `--selection-mode` | choice | `max` | Режим оптимизации selection-метрики (`min`/`max`). |
| `--answer-policies` | str | `last,argmax_interval` | Как считать task-метрики после остановки: `last` (последний шаг) и/или `argmax_interval` (argmax PMF на интервале `1..stop_step`). |
| `--progress-bar` | bool, опц. | `null` | Включить/выключить progress bar во время `eval-stopping`. Если не задано, берётся `train.progress_bar` из YAML. |

Оракульное \(\tau^\*\) для полей отчёта, сравнивающих остановку с шагом истинного минимума потерь (`nll`, `brier`, `corr` и др.): \(\tau^\*=\mathrm{argmax}_t\) маскированной потокенной accuracy на шаге \(t\) того же rollout. Поле `mean_regret` считает **лишний CE** относительно CE на шаге \(\tau^\*\): \(\mathrm{CE}(\text{stop})-\mathrm{CE}(\tau^\*)\) по каждому примеру (усреднение по батчу как раньше).

Выход:
- `records`: агрегированные метрики по комбинациям `distribution_model × strategy × threshold/budget` (включая budget-варианты по `selected_threshold`);
- `last_step_metrics`: baseline при выборе только последнего шага (`N_sup`): средние по val-батчам `token_acc`, `exact_acc`, `val_loss` на **финальном** шаге;
- `per_step_token_acc`: список длины `N_sup` — для каждого supervision-шага \(1..N_\mathrm{sup}\) то же batch-pooled `token_acc`, усреднённое по тем же батчам, что и остальной eval (удобно для кривых accuracy по глубине);
- `honest_selection`: при `--honest-split-ratio > 0` — строка, выбранная на части подбора, и её качество на части итоговой оценки.

Поле **`token_acc_*`** в записях стратегий (например `token_acc_last`) считается **с тем же batch-pooling**, что и `last_step_metrics["token_acc"]`: на каждом val-батче суммируются верные маскированные токены по всем примерам (у каждого свой шаг ответа из `--answer-policies`) и делится на сумму масок; затем усреднение по батчам как для остальных полей `records`. Раньше там было среднее per-sample rate по примеру — из‑за этого на ARC возникал разрыв ~0.66 vs ~0.73 с `last_token_acc`.

### Распределение \(\tau^\*\) на валидации (гистограмма)

Файл: `scripts/plot_optimal_stop_distribution.py`

Строит **одну фигуру из трёх панелей** на батчах val-loader: (1) \(\tau^\*\) по **каждому примеру** — \(\mathrm{argmax}_t\) per-sample помаскированной `token_acc` (как в `diplom-eval-stopping` при определении оракульного \(\tau^\*\)); (2) \(\tau^\*\) по **каждому батчу** — \(\mathrm{argmax}_t\) траектории batch-pooled `token_acc` (одно значение \(\tau^\*\) на батч; нормировка гистограммы по числу батчей); (3) две кривые `token_acc` по шагам (batch-pooled macro по батчам vs равный вес примеров). В `--out-json`: `counts`/`probabilities` — per-sample \(\tau^\*\); `counts_tau_per_batch_pooled_trajectory`/`probabilities_tau_per_batch_pooled_trajectory` — per-batch \(\tau^\*\); плюс два массива accuracy по шагам и `n_batches_used`.

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--config` | — | Путь к YAML эксперимента (**обязателен**). |
| `--checkpoint` | — | Путь к `.pt` чекпоинту (`state_dict` под ключом `model`) (**обязателен**). |
| `--out` | — | Путь к выходному PNG (**обязателен**). |
| `--out-json` | выкл. | JSON: две гистограммы \(\tau^\*\), два массива accuracy по шагам, см. описание выше. |
| `--max-steps` | из модели (`N_sup`) | Переопределить длину rollout (число supervision-шагов). |
| `--max-batches` | `50` | Сколько батчей val просмотреть. |
| `--title` | `Optimal stop-step distribution (validation)` | Заголовок графика. |

Пример:

```bash
uv run python scripts/plot_optimal_stop_distribution.py \
  --config configs/text_wikitext103_trm_oracle.yaml \
  --checkpoint runs/text_wikitext103_trm_oracle/checkpoints/best.pt \
  --out runs/text_wikitext103_trm_oracle/tau_star_hist_val.png \
  --out-json runs/text_wikitext103_trm_oracle/tau_star_hist_val.json \
  --max-batches 100
```

### Рисунок для диплома: одна панель (per-batch \(\tau^\*\), seaborn)

Файл: `scripts/plot_arc_arcagi_figures.py`

Строит **один** PNG (по умолчанию стилизованный средний ряд гистограммы из трёхпанельного графика ``per-batch \(\tau^\*\)''): столбцы \(\tau^\star\in\{1,\ldots,K\}\), нормированные доли. Данные по умолчанию зафиксированы в коде; опционально подставляются из структурированного вывода утилиты оценки (аргумент ``аналогично полю массива вероятностей per-batch \(\tau^\*\)'' в промежуточном дампе `plot_optimal_stop_distribution`), без упоминания формата в тексте работы.

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--from-json` | путь | выкл. | Входной дамп статистики: ожидается массив вероятностей per-batch \(\tau^\*\) (совместимо с ключом вероятностей из `plot_optimal_stop_distribution --out-json`). |
| `--out` | путь | `tex/img/arc_tau_star_hist_val.png` | Выходной PNG для вставки в \LaTeX. |
| `--dpi` | int | `160` | Разрешение при сохранении PNG. |
| `--write-hist-csv` | путь | выкл. | При указании записать файл со столбцами `step`, `count`, `probability` для сверки с таблицей в главе (`tex/tables/…`). |
| `--hist-weight` | int | `400` | Сумма целых счётчиков в колонке `count` при `--write-hist-csv` (метод наибольших остатков относительно нормированных долей). |
| `--extra-token-acc` | путь | выкл. | При указании пути PNG и наличии `--from-json`: дополнительный график токеновой точности по шагам (macro по батчам), если в дампе есть соответствующий массив. |
| `--title` | str | см. `--help` | Заголовок рисунка (можно русифицировать). |
| `--ylabel` | str | `Probability` | Подпись оси \(y\). |

Пример сборки центральной панели для диплома и синхронной таблицы в `tex/tables/` (загрузка через \texttt{datatool} в \LaTeX):

```bash
uv run python scripts/plot_arc_arcagi_figures.py \
  --out tex/img/arc_tau_star_hist_val.png \
  --write-hist-csv tex/tables/tau_star_hist_finite_discrete_val.csv \
  --hist-weight 400
```

Подстановка данных из промежуточного дампа (после `plot_optimal_stop_distribution`):

```bash
uv run python scripts/plot_arc_arcagi_figures.py \
  --from-json runs/oracle_sweep_arc_agi/finite_discrete/tau_star_hist_val.json \
  --out tex/img/arc_tau_star_hist_val.png \
  --write-hist-csv tex/tables/tau_star_hist_finite_discrete_val.csv \
  --hist-weight 400
```

### Качество и среднее число шагов для диплома (по точке на семейство распределений)

Файл: `scripts/plot_arc_quality_cost.py`

Строит диаграмму рассеяния: по горизонтали \(\bar{\tau}\), по вертикали \(\mathrm{tok}_{\mathrm{best}}\) — строки совпадают с `tex/tables/arc_stopping_base.csv`: столбцы `family`, `meantau`, `tokbest`; русские названия семейств задаются встроенным словарём. Положение точек должно совпадать с таблицей лучших строк по семействам в главе про ARC.

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--from-csv` | путь | `tex/tables/arc_stopping_base.csv` | Входная сводная таблица (после синхронизации с LaTeX). |
| `--out` | путь | `tex/img/arc_quality_cost_best.png` | Выход PNG. |
| `--dpi` | int | `160` | DPI. |
| `--title` | str | `Точность` | Заголовок области рисунка. |

```bash
uv run python scripts/plot_arc_quality_cost.py \
  --from-csv tex/tables/arc_stopping_base.csv \
  --out tex/img/arc_quality_cost_best.png
```

---

### Скрипт сводной таблицы stopping-eval

Файл: `scripts/summarize_stopping_eval.py`

Собирает `stopping_eval.json` по sweep-конфигам, фильтрует записи по `trained_distribution_model` из YAML и строит:
- CSV-таблицу;
- Markdown-таблицу;
- сравнение лучшей стратегии (по выбранной метрике) с baseline по шагам: колонки **`tok_s1`…`tok_sK`** (Markdown) / **`token_acc_step_1`…** (CSV) берутся из `per_step_token_acc` в JSON (если поля нет в старом файле — только один столбец из `last_step_metrics.token_acc`).
- для языковой задачи в CSV дополнительно выводятся **`best_top5_acc_*`** и **`best_top5_acc_best_of_policies`**, если в записях `records` есть поля `top5_acc_*` (см. `text_lm_task.compute_metrics`).

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--group` | `all` | Какие наборы сводить: `wikitext`, `arc`, `all`. |
| `--metric` | `token_acc` | Метрика выбора лучшей стратегии внутри каждого `run_dir`. |
| `--mode` | `max` | Режим оптимизации метрики: `min` или `max`. |
| `--use-honest-selection` | выкл. | Брать кандидатов из `honest_selection` (если есть), а не из полного `records`. |
| `--run-subdir` | пусто | Подкаталог внутри `train.run_dir`, откуда читать `stopping_eval` (например `oracle_finetune`). |
| `--eval-filename` | `stopping_eval.json` | Имя файла с результатами eval внутри `run_dir`/`run-subdir`. |
| `--out-csv` | авто | Путь к CSV (если не задан: `wikitext` → `runs/oracle_sweep_wikitext_falcon/stopping_summary.csv`, `arc` → `runs/oracle_sweep_arc_agi/...`, `all` → `runs/oracle_sweep_summary.csv`). |
| `--out-md` | авто | Путь к Markdown-таблице (аналогично CSV). |

Примеры:

```bash
# Сводка по ARC sweep (лучшее по минимальному regret)
uv run python scripts/summarize_stopping_eval.py --group arc --metric mean_regret --mode min

# Сводка по ARC из honest_selection (holdout-оценка)
uv run python scripts/summarize_stopping_eval.py --group arc --metric token_acc --mode max --use-honest-selection

# Сводка по ARC sweep (лучшее по максимальному token_acc) в явные файлы
uv run python scripts/summarize_stopping_eval.py \
  --group arc \
  --metric token_acc \
  --mode max \
  --out-csv runs/oracle_sweep_arc_agi/stopping_summary_token_acc.csv \
  --out-md runs/oracle_sweep_arc_agi/stopping_summary_token_acc.md

# Сводка по oracle_finetune stopping_eval в отдельный md/csv
uv run python scripts/summarize_stopping_eval.py \
  --group arc \
  --metric token_acc_best_of_policies \
  --mode max \
  --run-subdir oracle_finetune \
  --eval-filename stopping_eval.json \
  --out-csv runs/oracle_sweep_arc_agi/oracle_finetune/stopping_summary.csv \
  --out-md runs/oracle_sweep_arc_agi/oracle_finetune/stopping_summary.md
```

---

### Пересчёт после фикса budget-метрик

После обновления `eval_stopping` (глобальный budget-constraint без отбрасывания батчей) пересчитай сначала `stopping_eval`, затем summary.

```bash
# 1) Пересчитать базовые ARC stopping_eval (честный split + две answer-политики)
./scripts/validate_oracle_sweep.sh arc --continue-on-error --honest-split-ratio=0.5

# 2) Сводка базовых ARC run'ов
uv run python scripts/summarize_stopping_eval.py \
  --group arc \
  --metric token_acc_best_of_policies \
  --mode max \
  --out-csv runs/oracle_sweep_arc_agi/stopping_summary.csv \
  --out-md runs/oracle_sweep_arc_agi/stopping_summary.md

# 3) Пересчитать ARC oracle_finetune stopping_eval
for cfg in configs/oracle_sweep_arc_agi/arc_agi_trm_oracle_*.yaml; do
  run_dir="$(uv run python - <<PY
import yaml
cfg = yaml.safe_load(open('${cfg}', 'r', encoding='utf-8'))
print(cfg.get('train', {}).get('run_dir', ''))
PY
)"
  dist="$(uv run python - <<PY
import yaml
cfg = yaml.safe_load(open('${cfg}', 'r', encoding='utf-8'))
m = cfg.get('model', {}) or {}
mode = str(m.get('oracle_target_mode', 'delta')).lower()
d = str(m.get('oracle_distribution_model', 'finite_discrete'))
print('finite_discrete' if mode == 'delta' else d)
PY
)"
  ckpt="${run_dir}/oracle_finetune/checkpoints/best.pt"
  out="${run_dir}/oracle_finetune/stopping_eval.json"
  [[ -f "${ckpt}" ]] || { echo "[oracle-ft-validate] missing checkpoint: ${ckpt}" >&2; continue; }
  uv run diplom-eval-stopping \
    --config "${cfg}" \
    --checkpoint "${ckpt}" \
    --distribution-models "${dist}" \
    --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
    --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
    --budget-grid 2,4,6,8 \
    --honest-split-ratio 0.5 \
    --selection-metric token_acc \
    --selection-mode max \
    --answer-policies last,argmax_interval \
    --progress-bar true \
    --out "${out}"
done

# 4) Сводка oracle_finetune
uv run python scripts/summarize_stopping_eval.py \
  --group arc \
  --metric token_acc_best_of_policies \
  --mode max \
  --run-subdir oracle_finetune \
  --eval-filename stopping_eval.json \
  --out-csv runs/oracle_sweep_arc_agi/oracle_finetune/stopping_summary.csv \
  --out-md runs/oracle_sweep_arc_agi/oracle_finetune/stopping_summary.md
```

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
- при отсутствии `latexmk` выполняется заданное число прогонов выбранного движка, затем при наличии `thesis.bcf` вызывается `biber`, иначе при `biblatex`+`backend=bibtex` — `bibtex` из каталога `build/` с `BIBINPUTS`, включающим `workdir`, чтобы находился `references.bib`;
- в переменные окружения процесса подставляются `PAPER=a4` и `PAPERSIZE=a4` (обход поломанного системного `paper`, из‑за которого `xdvipdfmx` падает с «Unrecognized paper format»);
- в `tex/.latexmkrc` дублируется `PAPER`/`PAPERSIZE` для `latexmk`.

---

## `diplom-draw-arch`

Экспорт PDF/PNG блок-схем архитектуры сети (чистый TRM и sLLM + проектор + ядро TRM), без изображения пайплайна обучения с оракулом.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--out-dir` | str | `tex/img` | Каталог вывода: `arch_trm_network.pdf`/`png`, `arch_sllm_trm_network.pdf`/`png`. |
| `--dpi` | int | `220` | Разрешение растрового экспорта (`png`). |
| `--formats` | str | `pdf,png` | Форматы через запятую: `pdf`, `png` (любое подмножество). |

Примеры:

```bash
uv run diplom-draw-arch --out-dir tex/img --dpi 220 --formats pdf,png
uv run python scripts/draw_arch_figures.py --out-dir tex/img --dpi 220 --formats pdf,png
```

В `tex/chapters/03_architecture_models.tex` включаются **ручные** файлы `tex/img/simple.png` (чистый TRM) и `tex/img/frozen.png` (замороженная sLLM + проектор + ядро). Команда `diplom-draw-arch` генерирует отдельно `arch_trm_network` и `arch_sllm_trm_network` для черновых схем; смена рисунков в тексте задаётся аргументом `\includegraphics` или заменой `simple.png` / `frozen.png`.

---

## `render_wikitext_figs.py`

Генерирует рисунки для раздела результатов WikiText: гистограмму \(\tau^\star\) (`wikitext_tau_star_hist_val.pdf` / `.png`) и точечную диаграмму качество–глубину (`wikitext_quality_cost_best.pdf` / `.png`) из CSV в `tex/tables/` (`wikitext_tau_hist_val.csv`, `wikitext_lm_base.csv`).

| Аргумент | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `--img-dir` | `Path` | `tex/img` | Каталог выхода PNG/PDF. |
| `--tables-dir` | `Path` | `tex/tables` | Каталог входных CSV. |
| `--dpi` | `int` | `200` | DPI для PNG. |

Пример:

```bash
uv run python scripts/render_wikitext_figs.py --img-dir tex/img --tables-dir tex/tables --dpi 200
```

## `export_wikitext_lm_tex_tables.py`

Строит `tex/tables/wikitext_lm_base.csv` и `tex/tables/wikitext_lm_ft_compare.csv` из `oracle_finetune/stopping_summary.csv` (колонки `top5best`, сравнение до/после FT). Для строк FT-таблицы столбцы «до'' по метрикам берутся из `run_dir/stopping_eval.json`, если файл есть; иначе --- из предыдущего `wikitext_lm_ft_compare.csv` (аргумент `--legacy-ft-csv`).

| Аргумент | По умолчанию | Описание |
|----------|---------------|-----------|
| `--finetune-summary` | `runs/oracle_sweep_wikitext_falcon/oracle_finetune/stopping_summary.csv` | Входная сводка. |
| `--tables-dir` | `tex/tables` | Куда писать CSV. |
| `--legacy-ft-csv` | `tex/tables/wikitext_lm_ft_compare.csv` | Fallback для базовых столбцов при отсутствии корневого `stopping_eval.json`. |

```bash
uv run python scripts/export_wikitext_lm_tex_tables.py
```

---

## `diplom-plot`

Построение PNG по `metrics.jsonl` из каталога прогона.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--run-dir` | str | **обязателен** | Каталог с `metrics.jsonl` (например `runs/trm_1h`). |
| `--out` | str, опц. | — | Путь к выходному PNG; если не задан — `{run-dir}/plots.png` (четырёхпанельный дашборд). |
| `--out-loss` | str, опц. | — | Панель основной обучающей потери (`train_main_loss`, при наличии `train_loss_total`). |
| `--out-val-acc` | str, опц. | — | Только точности: val-метрики без «loss-like'' имён + train `*_acc` (одна ось \(y\)). |
| `--out-val-loss` | str, опц. | — | Только val loss-like (`val_loss`, \ldots) на одной оси. |
| `--plot-exact-acc` | флаг | выкл. | Не отфильтровывать ключи, содержащие `exact_acc`. |
| `--dpi` | int | `150` | DPI для PNG при использовании флагов `--out-*`. |
| `--no-ema-band` | флаг | выкл. | Отключить EMA и полосу ±σ по скользящему std. |
| `--ema-alpha` | float | `0.06` | Коэффициент EMA (больше — быстрее следует за сырым рядом). |
| `--val-ema-alpha` | float | `0.35` | EMA для валидационных точек на панелях `--out-val-acc` / `--out-val-loss`. |
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
  --oracle-policy greedy --oracle-max-steps 32 --progress-bar true

# График после прогона
uv run diplom-plot --run-dir runs/dev_trm_cgar_sudoku

# WikiText-103 materialization
uv run diplom-data text --name wikitext --dataset-config wikitext-103-raw-v1 --split train --dry-run false --materialize

# ARC-AGI materialization (готовит data/arc_agi/train.jsonl и data/arc_agi/val.jsonl)
uv run diplom-data arc-agi --name lordspline/arc-agi --split-train training --split-val evaluation --dry-run false --materialize

# TRM + Oracle на ARC-AGI
uv run diplom-train --config configs/arc_agi_trm_oracle.yaml

# Oracle-only дообучение от готового чекпоинта
uv run diplom-train --config configs/arc_agi_trm_oracle.yaml \
  --init-checkpoint runs/arc_agi_trm_oracle/checkpoints/best.pt \
  --oracle-only

# TRM + Oracle на WikiText-103
uv run diplom-train --config configs/text_wikitext103_trm_oracle.yaml

# TRM без oracle с фиксированным числом итераций
uv run diplom-train --config configs/text_wikitext103_trm_fixed.yaml

# Frozen LLM embeddings -> TRM correction module (Falcon-H1-Tiny-90M; `task.train_fraction: 0.1`)
uv run diplom-train --config configs/text_qwen_correction_trm.yaml

# LoRA ablation (тот же Falcon; `task.train_fraction: 0.1`)
uv run diplom-train --config configs/text_qwen_lora_ablation.yaml

# Сравнение всех стратегий остановки для заданной oracle-модели
uv run diplom-eval-stopping --config configs/text_wikitext103_trm_oracle.yaml \
  --checkpoint runs/text_wikitext103_trm_oracle/checkpoints/step_1000.pt \
  --distribution-models finite_discrete,smoothed_loss,mixture_geometric,mixture_exponential,power,negative_binomial,lognormal,hybrid \
  --strategies cumulative_probability,future_improvement,hazard,quantile,budget \
  --threshold-grid 0.5,0.6,0.7,0.8,0.9 \
  --budget-grid 2,4,6,8 \
  --out runs/text_wikitext103_trm_oracle/stopping_eval.json \
  --progress-bar true

# Сборка LaTeX-диплома
uv run diplom-tex-build --workdir tex --main thesis.tex --out-dir build --engine xelatex --passes 2
```
