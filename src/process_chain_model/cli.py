from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .parser import parse_chains
from .model import MarkovChainModel
from .io import load_model, save_model, save_predictions_json
from .metrics import calculate_metrics, print_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обнаружение аномалий в цепочках процессов Windows на основе марковских моделей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nПримеры использования:\n\n"
            "  # Обучение модели первого порядка\n"
            "  process-chain-model --mode train --input train_data.xlsx --order 1\n\n"
            "  # Тестирование\n"
            "  process-chain-model --mode test --input test_data.xlsx --model-file markov_order1.pkl\n\n"
            "  # Оценка с метриками\n"
            "  process-chain-model --mode evaluate --input test_data.xlsx --model-file markov_order1.pkl\n\n"
            "  # Визуализация\n"
            "  process-chain-model --mode visualize --model-file markov_order1.pkl --output-dir results\n\n"
            "  # Сравнение моделей\n"
            "  process-chain-model --mode compare --input train_data.xlsx\n"
        ),
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "test", "evaluate", "visualize", "compare", "update"],
        help="Режим работы",
    )

    parser.add_argument(
        "--results-json",
        default=None,
        help="Путь для сохранения результатов (цепочка, score, статус) в JSON",
    )

    parser.add_argument("--input", help="Excel файл с данными")
    parser.add_argument("--sheet", default=None, help="Лист Excel")

    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Порядок марковской модели (1-5), по умолчанию 1",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Параметр сглаживания Лапласа, по умолчанию 1.0",
    )

    parser.add_argument(
        "--model-file",
        default=None,
        help="Файл для сохранения/загрузки модели",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ручной порог для классификации",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Директория для результатов",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.model_file is None:
        args.model_file = f"markov_order{args.order}.pkl"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- TRAIN -----------------------------
    if args.mode == "train":
        if not args.input:
            logging.error("Укажите --input для обучения")
            return

        df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
        df["parsed_chains"] = parse_chains(df)
        chains = df["parsed_chains"].tolist()

        logging.info(f"Загружено {len(chains)} цепочек из {args.input}")

        model = MarkovChainModel(order=args.order, alpha=args.alpha)
        model.fit(chains, threshold_quantile=0.95)
        save_model(model, args.model_file)
        logging.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")

    # ----------------------------- TEST ------------------------------
    elif args.mode == "test":
        if not args.input:
            logging.error("Укажите --input для тестирования")
            return

        model = load_model(args.model_file)

        df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
        df["parsed_chains"] = parse_chains(df)

        results = []
        anomalies = []
        normals = []
        short = 0

        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 70 + "\n")

        for idx, row in df.iterrows():
            chain = row["parsed_chains"]
            label = row.get("chain_label", f"Chain_{idx + 1}")

            score = model.score(chain)
            pred = model.predict(chain, args.threshold)
            status = "✅ НОРМА" if pred == 1 else "⚠️ АНОМАЛИЯ"

            print(f"{label}: {status}")
            print(f"  Цепочка: {' → '.join(chain)}")

            if score is not None:
                print(f"  Скор: {score:.4f}")
            else:
                min_length = model.order + 1
                print(f"  Скор: N/A (цепочка слишком короткая: {len(chain)} < {min_length})")
                short += 1
            print()

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
            (anomalies if pred == -1 else normals).append(rec)

        # summary
        print("\n" + "=" * 70)
        print("СВОДКА ТЕСТИРОВАНИЯ")
        print("=" * 70)
        print(f"Всего проверено цепочек: {len(results)}")
        print(f"Обнаружено аномалий:     {len(anomalies)}")
        print(f"Нормальных цепочек:      {len(normals)}")
        print(f"Слишком коротких:        {short} (автоматически помечаются как аномалия)")

        if anomalies:
            anomalies_sorted = sorted(
                anomalies,
                key=lambda x: (x["score"] is None, x["score"]),
                reverse=True,
            )
            print("\nТОП аномальных цепочек:")
            for a in anomalies_sorted[:20]:
                sc = "N/A" if a["score"] is None else f"{a['score']:.4f}"
                print(f"- {a['label']} (score={sc}): {a['chain']}")

        if args.results_json:
            save_predictions_json(results, args.results_json)

    # --------------------------- EVALUATE ----------------------------
    elif args.mode == "evaluate":
        if not args.input:
            logging.error("Укажите --input для оценки")
            return

        model = load_model(args.model_file)

        df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
        if "label" not in df.columns:
            logging.error("Для evaluate нужна колонка 'label' (1=норма, -1=аномалия)")
            return

        df["parsed_chains"] = parse_chains(df)

        y_true = df["label"].values
        y_pred = []
        y_scores = []

        for chain in df["parsed_chains"]:
            pred = model.predict(chain, args.threshold)
            score = model.score(chain)
            y_pred.append(pred)
            y_scores.append(score if score is not None else 0)

        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)

        metrics = calculate_metrics(y_true, y_pred, y_scores)
        print_metrics(metrics)

        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info(f"Метрики сохранены: {metrics_file}")

    # --------------------------- VISUALIZE ---------------------------
    elif args.mode == "visualize":
        model = load_model(args.model_file)
        save_path = output_dir / f"markov_order{model.order}_graph.png"
        model.visualize(min_prob=0.01, save_path=str(save_path))

    # ---------------------------- COMPARE ----------------------------
    elif args.mode == "compare":
        if not args.input:
            logging.error("Укажите --input для сравнения")
            return

        df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
        df["parsed_chains"] = parse_chains(df)

        train_size = int(len(df) * 0.7)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        train_chains = train_df["parsed_chains"].tolist()

        results = []
        for order in [1, 2, 3]:
            logging.info(f"\nОбучение марковской модели порядка {order}...")
            m = MarkovChainModel(order=order, alpha=args.alpha)
            m.fit(train_chains, threshold_quantile=0.95)

            if "label" in test_df.columns:
                y_true = test_df["label"].values
                y_pred = [m.predict(c) for c in test_df["parsed_chains"]]
                y_scores = [m.score(c) or 0 for c in test_df["parsed_chains"]]

                met = calculate_metrics(y_true, np.array(y_pred), np.array(y_scores))
                results.append(
                    {
                        "order": order,
                        "mcc": met["mcc"],
                        "f1": met["f1_score"],
                        "precision": met["precision"],
                        "recall": met["recall"],
                        "roc_auc": met.get("roc_auc", 0),
                    }
                )

        if results:
            print("\n" + "=" * 80)
            print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
            print("=" * 80)
            print(f"\n{'Order':<8} {'MCC':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'ROC-AUC':<8}")
            print("-" * 80)
            for r in results:
                print(
                    f"{r['order']:<8} {r['mcc']:<8.4f} {r['f1']:<8.4f} "
                    f"{r['precision']:<10.4f} {r['recall']:<8.4f} {r['roc_auc']:<8.4f}"
                )

            best = max(results, key=lambda x: x["mcc"])
            print("\n" + "=" * 80)
            print(f" Лучшая модель: Order {best['order']} (MCC = {best['mcc']:.4f})")
            print("=" * 80 + "\n")

    # ----------------------------- UPDATE ----------------------------
    elif args.mode == "update":
        if not args.input:
            logging.error("Укажите --input для дообучения модели")
            return

        model = load_model(args.model_file)

        df = pd.read_excel(args.input, sheet_name=args.sheet) if args.sheet else pd.read_excel(args.input)
        df["parsed_chains"] = parse_chains(df)
        new_chains = df["parsed_chains"].tolist()

        model.update(new_chains)

        scores = [model.score(c) for c in new_chains]
        scores = [s for s in scores if s is not None]
        if scores:
            model.threshold = float(np.quantile(scores, 0.95))
            logging.info(f"Обновлённый порог: {model.threshold}")

        save_model(model, args.model_file)
        logging.info("✅ ДОOБУЧЕНИЕ ЗАВЕРШЕНО!")


if __name__ == "__main__":
    main()
