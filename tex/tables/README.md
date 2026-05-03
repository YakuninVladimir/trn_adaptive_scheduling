# Табличные данные для диплома (CSV)

Файлы в UTF-8, разделитель поля — запятая. Числа с десятичной **точкой** (как в `stopping_summary.csv` репозитория); в PDF запятая как десятичный разделитель задаётся через `siunitx`.

| Файл | Назначение |
|------|------------|
| `arc_stopping_base.csv` | Лучшая строка свода остановки после базового обучения ARC-AGI. |
| `arc_stopping_finetune_compare.csv` | Сравнение базы и `oracle_finetune` (политика и метрики). |
| `nlp_illustrative_params.csv` | Иллюстративные параметры языкового эксперимента (плейсхолдеры); колонки `param`, `colvalue`. |
| `tau_star_hist_finite_discrete_val.csv` | Гистограмма оракульного \(\tau^\star\) (ветка `finite_discrete`, val); синхронизировать с `runs/oracle_sweep_arc_agi/finite_discrete/tau_star_hist_val.json`. |

Обновление из репозитория: скопировать/скриптовать строки из `runs/oracle_sweep_arc_agi/stopping_summary.csv` и `oracle_finetune/stopping_summary.csv` в первые два файла при новых прогонах.
