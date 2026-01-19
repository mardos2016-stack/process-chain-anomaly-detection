# Process Chain Anomaly Detection (Markov)

Обнаружение аномалий в цепочках процессов Windows  
на основе **марковских моделей порядка 1–5**.

Проект предназначен для анализа process tree / process chains  
(например, из EDR, Sysmon, SIEM) и выявления нетипичных последовательностей
запуска процессов.

Код проекта основан на исходном скрипте `process_chain_model.py` и
приведён в поддерживаемый, модульный вид, удобный для использования,
тестирования и публикации на GitHub.

---

## Возможности

- Марковские модели **1–5 порядка**
- Сглаживание Лапласа (**Laplace smoothing**, `alpha`)
- **Автоматический порог** аномальности по квантилю (по умолчанию `0.95`)
- Полноценный **CLI-интерфейс**:
  - `train` — обучение
  - `test` — тестирование
  - `evaluate` — оценка качества
  - `visualize` — визуализация модели
  - `compare` — сравнение моделей разных порядков
  - `update` — инкрементальное дообучение
- Метрики для несбалансированных данных:
  - **MCC (основная)**
  - F1-score
  - Precision / Recall
  - ROC-AUC, PR-AUC
- Визуализация графа переходов марковской модели
- Поддержка инкрементального обучения без полного переобучения

---

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/mardos2016-stack/process-chain-anomaly-detection.git
cd process-chain-anomaly-detection
```

### 2. Создание виртуального окружения

```bash
python -m venv .venv
```

Активация:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```powershell
.venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Быстрый старт

### Обучение модели

```bash
python -m process_chain_model.cli --mode train --input examples/train.xlsx --order 1
```

Будет создан файл модели:

```
markov_order1.pkl
```

---

### Тестирование

```bash
python -m process_chain_model.cli --mode test --input examples/test.xlsx --model-file markov_order1.pkl
```

---

### Оценка качества (с метками)

⚠️ Входной файл должен содержать колонку `label`:
- `1` — норма  
- `-1` — аномалия  

```bash
python -m process_chain_model.cli \
  --mode evaluate \
  --input test_labeled.xlsx \
  --model-file markov_order1.pkl
```

---

### Визуализация модели

```bash
python -m process_chain_model.cli \
  --mode visualize \
  --model-file markov_order1.pkl \
  --output-dir results
```

---

## Формат входных данных

Ожидается колонка:

```
chain_proc_names
```

Пример значения:

```
svchost.exe ← services.exe ← wininit.exe
```

После парсинга цепочка преобразуется в:

```python
['wininit.exe', 'services.exe', 'svchost.exe']
```

---

## Структура репозитория

```
process-chain-anomaly-detection/
│
├── src/process_chain_model/
│   ├── model.py
│   ├── parser.py
│   ├── metrics.py
│   ├── io.py
│   └── cli.py
│
├── tests/
├── data/examples/
│   ├── test.xlsx
│   └── train.xlsx
├── results/
├── requirements.txt
├── README.md
└── LICENSE
```

---
