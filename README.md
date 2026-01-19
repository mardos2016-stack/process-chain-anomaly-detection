# Process Chain Anomaly Detection (Markov)

Обнаружение аномалий в цепочках процессов Windows на основе марковских моделей (порядок 1–5).

Код проекта собран из твоего исходного скрипта `process_chain_model.py` fileciteturn0file0 и разложен по модулям, чтобы было удобно поддерживать, тестировать и публиковать на GitHub.

## Возможности
- Марковские модели 1–5 порядка
- Laplace smoothing (`alpha`)
- Автопорог по квантилю (по умолчанию 0.95)
- CLI: train / test / evaluate / visualize / compare / update
- Метрики: MCC, F1, Precision, Recall, ROC-AUC, PR-AUC
- Визуализация графа переходов

## Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Запуск
### Обучение
```bash
python -m process_chain_model.cli --mode train --input train.xlsx --order 1
```

### Тестирование
```bash
python -m process_chain_model.cli --mode test --input test.xlsx --model-file markov_order1.pkl
```

### Оценка (нужна колонка `label`, где 1=норма, -1=аномалия)
```bash
python -m process_chain_model.cli --mode evaluate --input test_labeled.xlsx --model-file markov_order1.pkl
```

### Визуализация
```bash
python -m process_chain_model.cli --mode visualize --model-file markov_order1.pkl --output-dir results
```

## Формат входных данных
Ожидается колонка `chain_proc_names` в виде:

`svchost.exe ← services.exe ← wininit.exe`

Она будет превращена в список:

`[wininit.exe, services.exe, svchost.exe]`

## Структура репозитория
- `src/process_chain_model/` — библиотека + CLI
- `tests/` — тесты (pytest)
- `data/examples/` — маленькие примеры

