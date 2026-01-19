from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, precision_recall_curve,
    roc_auc_score, auc, f1_score, precision_score, recall_score,
    average_precision_score
)

# =============================================================================
#                           КОНСТАНТЫ
# =============================================================================

DEFAULT_MODEL_FILE = "markov_model.pkl"
SMOOTHING_EPS = 1e-12  # Защита от log(0)


# =============================================================================
#                           УТИЛИТЫ
# =============================================================================

def parse_chains(df: pd.DataFrame, column_name: str = "chain_proc_names") -> pd.Series:
    """
    Преобразует строку 'A ← B ← C' в ['C', 'B', 'A'].

    Формат входных данных Excel:
    chain_proc_names: "svchost.exe ← services.exe ← wininit.exe"

    Результат парсинга:
    ['wininit.exe', 'services.exe', 'svchost.exe']

    Args:
        df: DataFrame с данными
        column_name: Название колонки с цепочками

    Returns:
        Series с распарсенными цепочками (списками процессов)
    """

    def _parse_single_chain(chain_str):
        if not isinstance(chain_str, str):
            return []
        # Разделяем по стрелке и переворачиваем (от родителя к потомку)
        return [p.strip() for p in chain_str.split("←")][::-1]

    return df[column_name].apply(_parse_single_chain)


# =============================================================================
#                       МАРКОВСКАЯ МОДЕЛЬ
# =============================================================================

class MarkovChainModel:
    """
    Марковская модель для обнаружения аномалий в цепочках процессов.

    """

    def __init__(self, order: int = 1, alpha: float = 1.0):
        """
        Создаёт марковскую модель.

        Args:
            order: Порядок модели (1-5)
                  1 = классическая цепь Маркова
                  2-5 = модели высших порядков (учитывают больше контекста)
            alpha: Параметр сглаживания Лапласа
                  1.0 = добавляем "виртуальный" 1 переход ко всем комбинациям
                  0.0 = без сглаживания (не рекомендуется)
        """
        if order < 1 or order > 5:
            raise ValueError("order должен быть от 1 до 5")

        self.order = order
        self.alpha = alpha

        # Словарь переходов: {состояние: {следующий_процесс: количество}}
        # Для order=1: {'explorer.exe': {'cmd.exe': 10, 'chrome.exe': 5}}
        # Для order=2: {('explorer.exe', 'cmd.exe'): {'ipconfig.exe': 3}}
        self.transition_counts = {}

        # Словарь вероятностей: {состояние: {следующий_процесс: вероятность}}
        self.transition_probs = {}

        # Множество всех наблюдаемых процессов
        self.vocab = set()

        # Автоматический порог для классификации
        self.threshold = None

    def _get_state(self, chain: List[str], idx: int) -> Optional[str]:
        """
        Возвращает состояние (n-грамму) для индекса в цепочке.

        Для order=1: состояние = один процесс
        Для order=2: состояние = пара (tuple) процессов

        Пример для order=2, chain=['A', 'B', 'C', 'D'], idx=2:
            Состояние = ('A', 'B')
            Следующий процесс (chain[idx+1]) = 'C'

        Args:
            chain: Цепочка процессов
            idx: Индекс в цепочке

        Returns:
            Состояние (str или tuple) или None если недостаточно контекста
        """
        if idx < self.order - 1:
            return None

        if self.order == 1:
            return chain[idx]
        else:
            # Для высших порядков состояние = tuple предыдущих процессов
            return tuple(chain[idx - self.order + 1:idx + 1])

    def fit(self, chains: List[List[str]],
            threshold_quantile: float = 0.95):
        """
        Обучает модель на цепочках процессов.

        Args:
            chains: Список цепочек процессов для обучения
            threshold_quantile: Квантиль для автоматического порога (0.0-1.0)
                              0.95 = 95% обучающих данных считаются нормой
        """
        logging.info(f"Обучение марковской модели порядка {self.order} "
                     f"на {len(chains)} цепочках...")

        # Проверяем короткие цепочки
        short_chains = [c for c in chains if len(c) < self.order + 1]
        if short_chains:
            logging.warning(f"Найдено {len(short_chains)} коротких цепочек "
                            f"(< {self.order + 1} процессов), они будут пропущены")

        # ШАГ 1: Подсчёт переходов
        valid_chains = 0
        for chain in chains:
            if len(chain) < self.order + 1:
                continue

            valid_chains += 1

            # Добавляем процессы в словарь
            self.vocab.update(chain)

            # Считаем переходы
            for i in range(self.order, len(chain)):
                current_state = self._get_state(chain, i - 1)
                next_proc = chain[i]

                if current_state is not None:
                    # Инициализируем вложенный словарь если нужно
                    if current_state not in self.transition_counts:
                        self.transition_counts[current_state] = {}
                    if next_proc not in self.transition_counts[current_state]:
                        self.transition_counts[current_state][next_proc] = 0

                    # Увеличиваем счётчик
                    self.transition_counts[current_state][next_proc] += 1

        if valid_chains == 0:
            raise ValueError(f"Нет цепочек длиной >= {self.order + 1} для обучения "
                             f"модели порядка {self.order}")

        # ШАГ 2: Вычисляем вероятности с сглаживанием
        self._compute_probabilities()

        # ШАГ 3: Оцениваем порог на обучающих данных
        if threshold_quantile is not None:
            scores = [self.score(chain) for chain in chains]
            scores = [s for s in scores if s is not None]
            if scores:
                self.threshold = float(np.quantile(scores, threshold_quantile))
                logging.info(f"Автоматический порог: {self.threshold:.4f}")
                logging.info(f"Скоры: min={min(scores):.4f}, max={max(scores):.4f}, "
                             f"median={np.median(scores):.4f}")

        # Статистика обучения
        logging.info(f"Модель обучена:")
        logging.info(f"  - Валидных цепочек: {valid_chains}/{len(chains)}")
        logging.info(f"  - Уникальных состояний: {len(self.transition_counts)}")
        logging.info(f"  - Уникальных процессов (V): {len(self.vocab)}")
        logging.info(f"  - Всего переходов: "
                     f"{sum(sum(v.values()) for v in self.transition_counts.values())}")

    def _compute_probabilities(self):
        """
        Вычисляет вероятности переходов со сглаживанием Лапласа.

        """
        self.transition_probs = {}
        V = len(self.vocab)

        for state, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            denom = total + self.alpha * V

            self.transition_probs[state] = {
                next_proc: (count + self.alpha) / denom
                for next_proc, count in next_counts.items()
            }

    def update(self, new_chains: List[List[str]]):
        """
        Обновляет модель новыми данными (инкрементальное обучение).

        Используется для дообучения на новых нормальных цепочках без
        полного переобучения.

        Args:
            new_chains: Новые цепочки для добавления в модель
        """
        logging.info(f"Обновление модели на {len(new_chains)} новых цепочках...")

        for chain in new_chains:
            if len(chain) < self.order + 1:
                continue

            self.vocab.update(chain)

            for i in range(self.order, len(chain)):
                current_state = self._get_state(chain, i - 1)
                next_proc = chain[i]

                if current_state is not None:
                    if current_state not in self.transition_counts:
                        self.transition_counts[current_state] = {}
                    if next_proc not in self.transition_counts[current_state]:
                        self.transition_counts[current_state][next_proc] = 0
                    self.transition_counts[current_state][next_proc] += 1

        self._compute_probabilities()
        logging.info("Модель обновлена")

    def get_transition_prob(self, state, next_proc: str) -> float:
        """
        Возвращает вероятность перехода state → next_proc.

        Обрабатывает 3 случая:
        1. Известное состояние, известный переход → возвращаем вычисленную вероятность
        2. Известное состояние, неизвестный переход → сглаживание Лапласа
        3. Неизвестное состояние → минимальная вероятность

        Args:
            state: Состояние (процесс или tuple процессов)
            next_proc: Следующий процесс

        Returns:
            Вероятность перехода (0.0 до 1.0)
        """
        if state not in self.transition_probs:
            # Случай 3: Никогда не видели это состояние
            # Возвращаем минимальную вероятность α / (α×V)
            return self.alpha / (self.alpha * len(self.vocab))

        if next_proc not in self.transition_probs[state]:
            # Случай 2: Видели состояние, но не видели этот переход из него
            # Применяем сглаживание Лапласа
            total = sum(self.transition_counts[state].values())
            return self.alpha / (total + self.alpha * len(self.vocab))

        # Случай 1: Известный переход
        return self.transition_probs[state][next_proc]

    def score(self, chain: List[str], normalize: bool = True) -> Optional[float]:
        """
        Вычисляет скор аномальности цепочки.


        Args:
            chain: Цепочка процессов
            normalize: Нормализовать на количество переходов

        Returns:
            Скор аномальности или None если цепочка слишком короткая
        """
        if len(chain) < self.order + 1:
            return None

        log_likelihood = 0.0
        count = 0

        for i in range(self.order, len(chain)):
            current_state = self._get_state(chain, i - 1)
            next_proc = chain[i]

            if current_state is not None:
                prob = self.get_transition_prob(current_state, next_proc)
                # Защита от log(0)
                prob = max(prob, SMOOTHING_EPS)
                # Накапливаем отрицательный log-likelihood
                log_likelihood += -math.log(prob)
                count += 1

        if count == 0:
            return None

        # Нормализация на количество переходов
        if normalize:
            return log_likelihood / count
        return log_likelihood

    def predict(self, chain: List[str], threshold: Optional[float] = None) -> int:
        """
        Предсказывает класс цепочки: норма или аномалия.

        Алгоритм:
        =========

        1. Вычисляем скор цепочки
        2. Сравниваем с порогом
        3. Если скор > порог → аномалия

        Args:
            chain: Цепочка процессов
            threshold: Порог (если None, используется автоматический)

        Returns:
            1 = норма, -1 = аномалия
        """
        score = self.score(chain)

        if score is None:
            # Слишком короткая цепочка = подозрительно
            return -1

        thresh = threshold if threshold is not None else self.threshold

        if thresh is None:
            logging.warning("Порог не установлен, возвращаю норму")
            return 1

        # Для марковской модели: больше = хуже
        return -1 if score > thresh else 1

    def get_top_transitions(self, state, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Возвращает топ-K наиболее вероятных переходов из состояния.

        Полезно для анализа и отладки модели.

        Args:
            state: Состояние (процесс или tuple)
            top_k: Количество топовых переходов

        Returns:
            Список (процесс, вероятность) отсортированный по убыванию
        """
        if state not in self.transition_probs:
            return []

        transitions = self.transition_probs[state].items()
        return sorted(transitions, key=lambda x: x[1], reverse=True)[:top_k]

    def visualize(self, min_prob: float = 0.01,
                  max_nodes: int = 50,
                  save_path: Optional[str] = None):
        """
        Визуализирует граф переходов марковской модели.

        Создаёт направленный граф где:
        - Узлы = состояния (процессы или n-граммы)
        - Рёбра = переходы с вероятностями

        Args:
            min_prob: Минимальная вероятность для отображения ребра
            max_nodes: Максимальное количество узлов (для читаемости)
            save_path: Путь для сохранения PNG (если None, показывает окно)
        """
        logging.info("Создание визуализации графа переходов...")

        # Собираем рёбра
        edges = []
        for state, transitions in self.transition_probs.items():
            # Конвертируем состояние в строку для отображения
            state_str = state if isinstance(state, str) else "→".join(state)

            for next_proc, prob in transitions.items():
                if prob >= min_prob:
                    edges.append((state_str, next_proc, prob))

        if not edges:
            logging.warning("Нет рёбер для визуализации (попробуй уменьшить min_prob)")
            return

        # Ограничиваем количество узлов для читаемости
        all_nodes = set(e[0] for e in edges) | set(e[1] for e in edges)
        if len(all_nodes) > max_nodes:
            # Берём только самые частые состояния
            state_counts = Counter(e[0] for e in edges)
            top_states = set(s for s, _ in state_counts.most_common(max_nodes // 2))
            edges = [e for e in edges if e[0] in top_states]
            logging.info(f"Граф ограничен топ-{max_nodes // 2} состояниями для читаемости")

        # Создаём граф
        G = nx.DiGraph()
        for src, dst, prob in edges:
            G.add_edge(src, dst, weight=prob)

        # Визуализация
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

        # Рисуем узлы
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=2000, alpha=0.9)

        # Рисуем рёбра
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               arrows=True, arrowsize=15, alpha=0.6)

        # Подписи узлов
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        # Подписи рёбер (вероятности)
        edge_labels = {(e[0], e[1]): f"{e[2]:.2f}" for e in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

        plt.title(f"Граф переходов марковской модели (order={self.order})",
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"График сохранён: {save_path}")
        else:
            plt.show()

        plt.close()


# =============================================================================
#                           МЕТРИКИ
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Вычисляет метрики для оценки модели.

    Метрики для несбалансированных данных:
    ======================================

    1. MCC (Matthews Correlation Coefficient) — ПЕРВИЧНАЯ МЕТРИКА
       - Учитывает все 4 категории confusion matrix
       - Не зависит от дисбаланса классов
       - Значения: -1 (худшее) до +1 (идеально)

    2. F1-Score
       - Гармоническое среднее Precision и Recall
       - Хорошо для несбалансированных данных

    3. Precision (точность)
       - TP / (TP + FP)
       - Доля правильных среди предсказанных аномалий

    4. Recall (полнота)
       - TP / (TP + FN)
       - Доля найденных аномалий

    5. ROC-AUC, PR-AUC
       - Threshold-independent метрики
       - Требуют скоры (y_scores)

    Args:
        y_true: Истинные метки (1 = норма, -1 = аномалия)
        y_pred: Предсказанные метки
        y_scores: Скоры аномальности (опционально)

    Returns:
        Словарь с метриками
    """
    # Преобразуем в бинарные (0 = норма, 1 = аномалия)
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)

    metrics = {}

    # Confusion Matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)

    # MCC (первичная метрика!)
    metrics['mcc'] = matthews_corrcoef(y_true_binary, y_pred_binary)

    # Precision, Recall, F1
    metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics['f1_score'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # Accuracy (используй осторожно для несбалансированных данных!)
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # FPR (False Positive Rate)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

    # AUC метрики (если есть скоры)
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_scores)
            metrics['pr_auc'] = average_precision_score(y_true_binary, y_scores)
        except:
            pass

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Красиво выводит метрики."""
    print("\n" + "=" * 70)
    print("МЕТРИКИ МОДЕЛИ")
    print("=" * 70)

    print(f"\n Confusion Matrix:")
    print(f"   TP: {metrics.get('true_positives', 0):>5} | FP: {metrics.get('false_positives', 0):>5}")
    print(f"   FN: {metrics.get('false_negatives', 0):>5} | TN: {metrics.get('true_negatives', 0):>5}")

    print(f"\n Основные метрики:")
    print(f"   MCC (Matthews Correlation):  {metrics.get('mcc', 0):>7.4f}   ПЕРВИЧНАЯ")
    print(f"   F1-Score:                    {metrics.get('f1_score', 0):>7.4f}")
    print(f"   Precision:                   {metrics.get('precision', 0):>7.4f}")
    print(f"   Recall:                      {metrics.get('recall', 0):>7.4f}")

    if 'roc_auc' in metrics:
        print(f"\n AUC метрики:")
        print(f"   ROC-AUC:                     {metrics.get('roc_auc', 0):>7.4f}")
        print(f"   PR-AUC:                      {metrics.get('pr_auc', 0):>7.4f}")

    print(f"\n Дополнительно:")
    print(f"   Accuracy:                    {metrics.get('accuracy', 0):>7.4f}")
    print(f"   False Positive Rate:         {metrics.get('fpr', 0):>7.4f}")
    print("=" * 70 + "\n")


# =============================================================================
#                           СОХРАНЕНИЕ/ЗАГРУЗКА
# =============================================================================

def save_model(model: MarkovChainModel, filepath: str):
    """
    Сохраняет модель в файл.

    Что сохраняется в .pkl файле:
    ==============================

    1. model.order — порядок модели (1-5)
    2. model.alpha — параметр сглаживания
    3. model.transition_counts — словарь переходов с подсчётами
       Пример для order=1:
       {
           'explorer.exe': {'cmd.exe': 10, 'chrome.exe': 5},
           'cmd.exe': {'ipconfig.exe': 3, 'ping.exe': 2}
       }

    4. model.transition_probs — вычисленные вероятности
       Пример:
       {
           'explorer.exe': {'cmd.exe': 0.375, 'chrome.exe': 0.250, ...},
           'cmd.exe': {'ipconfig.exe': 0.286, 'ping.exe': 0.190, ...}
       }

    5. model.vocab — set всех уникальных процессов
       {'explorer.exe', 'cmd.exe', 'chrome.exe', ...}

    6. model.threshold — порог для классификации (float)
       Например: 2.5431

    Формат: Python pickle (бинарный)
    Размер: обычно 10-500 KB в зависимости от количества процессов

    Args:
        model: Обученная модель
        filepath: Путь для сохранения (например, 'markov_order1.pkl')
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Модель сохранена: {filepath}")

    # Выводим статистику сохранённой модели
    file_size = Path(filepath).stat().st_size / 1024
    logging.info(f"  Размер файла: {file_size:.1f} KB")
    logging.info(f"  Порядок: {model.order}")
    logging.info(f"  Состояний: {len(model.transition_counts)}")
    logging.info(f"  Процессов в словаре: {len(model.vocab)}")
    logging.info(f"  Порог: {model.threshold}")

def save_predictions_json(results: List[dict], filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Результаты сохранены: {filepath}")
def load_model(filepath: str) -> MarkovChainModel:
    """
    Загружает модель из файла.

    Args:
        filepath: Путь к файлу модели

    Returns:
        Загруженная модель
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Модель загружена: {filepath}")
    logging.info(f"  Порядок: {model.order}")
    logging.info(f"  Состояний: {len(model.transition_counts)}")
    logging.info(f"  Процессов: {len(model.vocab)}")
    logging.info(f"  Порог: {model.threshold}")
    return model


# =============================================================================
#                           CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Обнаружение аномалий в цепочках процессов Windows на основе марковских моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Обучение модели первого порядка
  python process_chain_model.py --mode train --input train_data.xlsx --order 1

  # Обучение модели второго порядка
  python process_chain_model.py --mode train --input train_data.xlsx --order 2

  # Тестирование
  python process_chain_model.py --mode test --input test_data.xlsx --model-file markov_order1.pkl

  # Оценка с метриками
  python process_chain_model.py --mode evaluate --input test_data.xlsx --model-file markov_order1.pkl

  # Визуализация
  python process_chain_model.py --mode visualize --model-file markov_order1.pkl --output-dir results

  # Сравнение моделей
  python process_chain_model.py --mode compare --input test_data.xlsx
        """
    )

    parser.add_argument("--mode", required=True,
                        choices=['train', 'test', 'evaluate', 'visualize', 'compare', 'update'],
                        help="Режим работы")

    parser.add_argument("--results-json", default=None,
                        help="Путь для сохранения результатов (цепочка, score, статус) в JSON")

    parser.add_argument("--input", help="Excel файл с данными")
    parser.add_argument("--sheet", default=None, help="Лист Excel")

    parser.add_argument("--order", type=int, default=1,
                        help="Порядок марковской модели (1-5), по умолчанию 1")

    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Параметр сглаживания Лапласа, по умолчанию 1.0")

    parser.add_argument("--model-file", default=None,
                        help="Файл для сохранения/загрузки модели")

    parser.add_argument("--threshold", type=float, default=None,
                        help="Ручной порог для классификации")

    parser.add_argument("--output-dir", default="results",
                        help="Директория для результатов")

    parser.add_argument("--log-level", default="INFO",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Файл модели по умолчанию
    if args.model_file is None:
        args.model_file = f"markov_order{args.order}.pkl"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # =========================================================================
    # ОБУЧЕНИЕ
    # =========================================================================
    if args.mode == 'train':
        if not args.input:
            logging.error("Укажите --input для обучения")
            return

        logging.info("=" * 70)
        logging.info("ОБУЧЕНИЕ МАРКОВСКОЙ МОДЕЛИ")
        logging.info("=" * 70)

        # Читаем данные
        df = pd.read_excel(args.input, sheet_name=args.sheet) \
            if args.sheet else pd.read_excel(args.input)

        df['parsed_chains'] = parse_chains(df)
        chains = df['parsed_chains'].tolist()

        logging.info(f"Загружено {len(chains)} цепочек из {args.input}")

        # Создаём и обучаем модель
        model = MarkovChainModel(order=args.order, alpha=args.alpha)
        model.fit(chains, threshold_quantile=0.95)

        # Сохраняем
        save_model(model, args.model_file)

        logging.info("=" * 70)
        logging.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        logging.info("=" * 70)

    # =========================================================================
    # ТЕСТИРОВАНИЕ
    # =========================================================================
    elif args.mode == 'test':
        if not args.input:
            logging.error("Укажите --input для тестирования")
            return

        # Загружаем модель
        model = load_model(args.model_file)

        # Читаем данные
        df = pd.read_excel(args.input, sheet_name=args.sheet) \
            if args.sheet else pd.read_excel(args.input)

        df['parsed_chains'] = parse_chains(df)

        # Тестируем
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 70 + "\n")

        results = []
        anomalies = []
        normals = []
        short = 0

        for idx, row in df.iterrows():
            chain = row['parsed_chains']
            label = row.get('chain_label', f'Chain_{idx + 1}')

            score = model.score(chain)
            pred = model.predict(chain, args.threshold)

            status = "✅ НОРМА" if pred == 1 else "⚠️ АНОМАЛИЯ"

            # печать как раньше
            print(f"{label}: {status}")
            print(f"  Цепочка: {' → '.join(chain)}")

            if score is not None:
                print(f"  Скор: {score:.4f}")
            else:
                min_length = model.order + 1
                print(f"  Скор: N/A (цепочка слишком короткая: {len(chain)} < {min_length})")
                short += 1

            print()

            # сбор результатов
            rec = {
                "index": int(idx),
                "label": str(label),
                "pred": int(pred),
                "status": "anomaly" if pred == -1 else "normal",
                "score": None if score is None else float(score),
                "chain": " → ".join(chain),
                "chain_len": int(len(chain)),
            }
            results.append(rec)

            if pred == -1:
                anomalies.append(rec)
            else:
                normals.append(rec)

        # --------------------- СВОДКА ---------------------
        print("\n" + "=" * 70)
        print("СВОДКА ТЕСТИРОВАНИЯ")
        print("=" * 70)

        total = len(results)
        print(f"Всего проверено цепочек: {total}")
        print(f"Обнаружено аномалий:     {len(anomalies)}")
        print(f"Нормальных цепочек:      {len(normals)}")
        print(f"Слишком коротких:        {short} (автоматически помечаются как аномалия)")

        # выводим список аномалий (лучше отсортировать по score убыванию)
        if anomalies:
            anomalies_sorted = sorted(
                anomalies,
                key=lambda x: (x["score"] is None, x["score"]),  # None в конец
                reverse=True
            )

            print("\nТОП аномальных цепочек:")
            for a in anomalies_sorted[:20]:  # ограничим вывод 20, чтобы не засорять консоль
                sc = "N/A" if a["score"] is None else f"{a['score']:.4f}"
                print(f"- {a['label']} (score={sc}): {a['chain']}")

    # =========================================================================
    # ОЦЕНКА
    # =========================================================================
    elif args.mode == 'evaluate':
        if not args.input:
            logging.error("Укажите --input для оценки")
            return

        # Загружаем модель
        model = load_model(args.model_file)

        # Читаем данные
        df = pd.read_excel(args.input, sheet_name=args.sheet) \
            if args.sheet else pd.read_excel(args.input)

        if 'label' not in df.columns:
            logging.error("Для evaluate нужна колонка 'label' (1=норма, -1=аномалия)")
            return

        df['parsed_chains'] = parse_chains(df)

        # Предсказываем
        y_true = df['label'].values
        y_pred = []
        y_scores = []

        for chain in df['parsed_chains']:
            pred = model.predict(chain, args.threshold)
            score = model.score(chain)

            y_pred.append(pred)
            y_scores.append(score if score is not None else 0)

        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        # Вычисляем метрики
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        print_metrics(metrics)

        # Сохраняем
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Метрики сохранены: {metrics_file}")

    # =========================================================================
    # ВИЗУАЛИЗАЦИЯ
    # =========================================================================
    elif args.mode == 'visualize':
        model = load_model(args.model_file)

        save_path = output_dir / f'markov_order{model.order}_graph.png'
        model.visualize(min_prob=0.01, save_path=str(save_path))

    # =========================================================================
    # СРАВНЕНИЕ
    # =========================================================================
    elif args.mode == 'compare':
        if not args.input:
            logging.error("Укажите --input для сравнения")
            return

        print("\n" + "=" * 70)
        print("СРАВНЕНИЕ МАРКОВСКИХ МОДЕЛЕЙ РАЗНЫХ ПОРЯДКОВ")
        print("=" * 70 + "\n")

        # Читаем данные
        df = pd.read_excel(args.input, sheet_name=args.sheet) \
            if args.sheet else pd.read_excel(args.input)

        df['parsed_chains'] = parse_chains(df)

        # Разделяем на train/test
        train_size = int(len(df) * 0.7)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        train_chains = train_df['parsed_chains'].tolist()

        # Обучаем модели разных порядков
        results = []

        for order in [1, 2, 3]:
            logging.info(f"\nОбучение марковской модели порядка {order}...")

            model = MarkovChainModel(order=order, alpha=args.alpha)
            model.fit(train_chains, threshold_quantile=0.95)

            # Если есть метки - оцениваем
            if 'label' in test_df.columns:
                y_true = test_df['label'].values
                y_pred = [model.predict(chain) for chain in test_df['parsed_chains']]
                y_scores = [model.score(chain) or 0 for chain in test_df['parsed_chains']]

                metrics = calculate_metrics(y_true, np.array(y_pred), np.array(y_scores))

                results.append({
                    'order': order,
                    'mcc': metrics['mcc'],
                    'f1': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'roc_auc': metrics.get('roc_auc', 0)
                })

        # Выводим таблицу
        if results:
            print("\n" + "=" * 80)
            print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
            print("=" * 80)
            print(f"\n{'Order':<8} {'MCC':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'ROC-AUC':<8}")
            print("-" * 80)
            for r in results:
                print(f"{r['order']:<8} {r['mcc']:<8.4f} {r['f1']:<8.4f} "
                      f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['roc_auc']:<8.4f}")

            # Находим лучшую модель
            best = max(results, key=lambda x: x['mcc'])
            print("\n" + "=" * 80)
            print(f" Лучшая модель: Order {best['order']} (MCC = {best['mcc']:.4f})")
            print("=" * 80 + "\n")

    # =========================================================================
    # ДОOБУЧЕНИЕ МОДЕЛИ
    # =========================================================================
    elif args.mode == 'update':
        if not args.input:
            logging.error("Укажите --input для дообучения модели")
            return

        logging.info("=" * 70)
        logging.info("ДОOБУЧЕНИЕ МАРКОВСКОЙ МОДЕЛИ")
        logging.info("=" * 70)

        # Загружаем модель
        model = load_model(args.model_file)

        # Загружаем новые данные
        df = pd.read_excel(args.input, sheet_name=args.sheet) \
            if args.sheet else pd.read_excel(args.input)

        df['parsed_chains'] = parse_chains(df)
        new_chains = df['parsed_chains'].tolist()

        logging.info(f"Загружено {len(new_chains)} новых цепочек")

        # Дообучение
        model.update(new_chains)
        logging.info("Переходные вероятности обновлены")

        # (Опционально) обновляем threshold
        scores = [model.score(c) for c in new_chains if model.score(c) is not None]
        if scores:
            new_threshold = float(np.quantile(scores, 0.95))
            logging.info(f"Обновление порога: old={model.threshold}, new={new_threshold}")
            model.threshold = new_threshold

        # Сохраняем обновлённую модель
        save_model(model, args.model_file)

        logging.info("=" * 70)
        logging.info("✅ ДОOБУЧЕНИЕ ЗАВЕРШЕНО!")
        logging.info("=" * 70)

if __name__ == "__main__":
    main()