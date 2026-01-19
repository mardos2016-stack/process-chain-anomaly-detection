from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List

from .model import MarkovChainModel


def save_model(model: MarkovChainModel, filepath: str) -> None:
    """Save model as a pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Модель сохранена: {filepath}")

    file_size_kb = Path(filepath).stat().st_size / 1024
    logging.info(f"  Размер файла: {file_size_kb:.1f} KB")
    logging.info(f"  Порядок: {model.order}")
    logging.info(f"  Состояний: {len(model.transition_counts)}")
    logging.info(f"  Процессов в словаре: {len(model.vocab)}")
    logging.info(f"  Порог: {model.threshold}")


def load_model(filepath: str) -> MarkovChainModel:
    """Load a model from a pickle file."""
    with open(filepath, "rb") as f:
        model: MarkovChainModel = pickle.load(f)
    logging.info(f"Модель загружена: {filepath}")
    logging.info(f"  Порядок: {model.order}")
    logging.info(f"  Состояний: {len(model.transition_counts)}")
    logging.info(f"  Процессов: {len(model.vocab)}")
    logging.info(f"  Порог: {model.threshold}")
    return model


def save_predictions_json(results: List[dict], filepath: str) -> None:
    """Save per-chain predictions/scores as JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Результаты сохранены: {filepath}")
