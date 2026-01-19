from __future__ import annotations

import logging
import math
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .constants import SMOOTHING_EPS


class MarkovChainModel:
    """Markov model for anomaly detection in Windows process chains.

    Supports order 1-5 (higher-order chains keep more context) and Laplace smoothing.
    """

    def __init__(self, order: int = 1, alpha: float = 1.0):
        if order < 1 or order > 5:
            raise ValueError("order должен быть от 1 до 5")

        self.order = order
        self.alpha = float(alpha)

        # Transition counts/probs:
        # order=1: {"explorer.exe": {"cmd.exe": 10, ...}}
        # order=2: {("explorer.exe","cmd.exe"): {"ipconfig.exe": 3, ...}}
        self.transition_counts = {}
        self.transition_probs = {}

        self.vocab = set()
        self.threshold: Optional[float] = None

    def _get_state(self, chain: List[str], idx: int):
        if idx < self.order - 1:
            return None
        if self.order == 1:
            return chain[idx]
        return tuple(chain[idx - self.order + 1 : idx + 1])

    def fit(self, chains: List[List[str]], threshold_quantile: float = 0.95):
        logging.info(
            f"Обучение марковской модели порядка {self.order} на {len(chains)} цепочках..."
        )

        short_chains = [c for c in chains if len(c) < self.order + 1]
        if short_chains:
            logging.warning(
                f"Найдено {len(short_chains)} коротких цепочек (< {self.order + 1} процессов), они будут пропущены"
            )

        valid_chains = 0
        for chain in chains:
            if len(chain) < self.order + 1:
                continue

            valid_chains += 1
            self.vocab.update(chain)

            for i in range(self.order, len(chain)):
                current_state = self._get_state(chain, i - 1)
                next_proc = chain[i]

                if current_state is None:
                    continue

                self.transition_counts.setdefault(current_state, {})
                self.transition_counts[current_state].setdefault(next_proc, 0)
                self.transition_counts[current_state][next_proc] += 1

        if valid_chains == 0:
            raise ValueError(
                f"Нет цепочек длиной >= {self.order + 1} для обучения модели порядка {self.order}"
            )

        self._compute_probabilities()

        if threshold_quantile is not None:
            scores = [self.score(chain) for chain in chains]
            scores = [s for s in scores if s is not None]
            if scores:
                self.threshold = float(np.quantile(scores, threshold_quantile))
                logging.info(f"Автоматический порог: {self.threshold:.4f}")
                logging.info(
                    f"Скоры: min={min(scores):.4f}, max={max(scores):.4f}, median={np.median(scores):.4f}"
                )

        logging.info("Модель обучена:")
        logging.info(f"  - Валидных цепочек: {valid_chains}/{len(chains)}")
        logging.info(f"  - Уникальных состояний: {len(self.transition_counts)}")
        logging.info(f"  - Уникальных процессов (V): {len(self.vocab)}")
        logging.info(
            f"  - Всего переходов: {sum(sum(v.values()) for v in self.transition_counts.values())}"
        )

    def _compute_probabilities(self):
        self.transition_probs = {}
        V = len(self.vocab)
        if V == 0:
            return

        for state, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            denom = total + self.alpha * V
            self.transition_probs[state] = {
                next_proc: (count + self.alpha) / denom
                for next_proc, count in next_counts.items()
            }

    def update(self, new_chains: List[List[str]]):
        logging.info(f"Обновление модели на {len(new_chains)} новых цепочках...")

        for chain in new_chains:
            if len(chain) < self.order + 1:
                continue

            self.vocab.update(chain)

            for i in range(self.order, len(chain)):
                current_state = self._get_state(chain, i - 1)
                next_proc = chain[i]

                if current_state is None:
                    continue

                self.transition_counts.setdefault(current_state, {})
                self.transition_counts[current_state].setdefault(next_proc, 0)
                self.transition_counts[current_state][next_proc] += 1

        self._compute_probabilities()
        logging.info("Модель обновлена")

    def get_transition_prob(self, state, next_proc: str) -> float:
        if state not in self.transition_probs:
            # never saw this state
            return self.alpha / (self.alpha * len(self.vocab)) if self.vocab else 0.0

        if next_proc not in self.transition_probs[state]:
            total = sum(self.transition_counts[state].values())
            return self.alpha / (total + self.alpha * len(self.vocab))

        return self.transition_probs[state][next_proc]

    def score(self, chain: List[str], normalize: bool = True) -> Optional[float]:
        if len(chain) < self.order + 1:
            return None

        log_likelihood = 0.0
        count = 0

        for i in range(self.order, len(chain)):
            current_state = self._get_state(chain, i - 1)
            next_proc = chain[i]

            if current_state is None:
                continue

            prob = self.get_transition_prob(current_state, next_proc)
            prob = max(prob, SMOOTHING_EPS)
            log_likelihood += -math.log(prob)
            count += 1

        if count == 0:
            return None

        return (log_likelihood / count) if normalize else log_likelihood

    def predict(self, chain: List[str], threshold: Optional[float] = None) -> int:
        score = self.score(chain)
        if score is None:
            return -1

        thresh = threshold if threshold is not None else self.threshold
        if thresh is None:
            logging.warning("Порог не установлен, возвращаю норму")
            return 1

        return -1 if score > thresh else 1

    def get_top_transitions(self, state, top_k: int = 5) -> List[Tuple[str, float]]:
        if state not in self.transition_probs:
            return []
        transitions = self.transition_probs[state].items()
        return sorted(transitions, key=lambda x: x[1], reverse=True)[:top_k]

    def visualize(
        self,
        min_prob: float = 0.01,
        max_nodes: int = 50,
        save_path: Optional[str] = None,
    ):
        """Visualize transition graph of the Markov model."""
        logging.info("Создание визуализации графа переходов...")

        edges = []
        for state, transitions in self.transition_probs.items():
            state_str = state if isinstance(state, str) else "→".join(state)
            for next_proc, prob in transitions.items():
                if prob >= min_prob:
                    edges.append((state_str, next_proc, prob))

        if not edges:
            logging.warning("Нет рёбер для визуализации (попробуй уменьшить min_prob)")
            return

        all_nodes = set(e[0] for e in edges) | set(e[1] for e in edges)
        if len(all_nodes) > max_nodes:
            state_counts = Counter(e[0] for e in edges)
            top_states = set(s for s, _ in state_counts.most_common(max_nodes // 2))
            edges = [e for e in edges if e[0] in top_states]
            logging.info(f"Граф ограничен топ-{max_nodes // 2} состояниями для читаемости")

        G = nx.DiGraph()
        for src, dst, prob in edges:
            G.add_edge(src, dst, weight=prob)

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=2000, alpha=0.9)
        nx.draw_networkx_edges(
            G, pos, edge_color="gray", arrows=True, arrowsize=15, alpha=0.6
        )
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

        edge_labels = {(e[0], e[1]): f"{e[2]:.2f}" for e in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

        plt.title(
            f"Граф переходов марковской модели (order={self.order})",
            fontsize=14,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"График сохранён: {save_path}")
        else:
            plt.show()

        plt.close()
