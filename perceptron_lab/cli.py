# запуск экспериментов с перцептроном из командной строки

import argparse
from pathlib import Path

from perceptron_lab.datasets import Dataset, make_iris, make_synthetic, split_dataset
from perceptron_lab.model import Perceptron
from perceptron_lab.plots import save_plot


def _run_experiment(
    name: str,
    data: Dataset,
    learning_rate: float,
    epochs: int,
    seed: int,
    output: Path,
) -> None:
    # обучаем модель и проверяем её на отложенных данных
    train, test = split_dataset(*data, seed=seed)
    model = Perceptron(n_features=2, learning_rate=learning_rate, epochs=epochs)
    model.train(*train)

    path = output / f"{name}.png"
    save_plot(model, train, test, path)

    print(f"\nэксперимент : {name}")
    print(f"эпох выполнено : {len(model.errors)}")
    print(
        f"точность на обучении : {model.score(*train):.1%} ({len(train[1])} примеров)"
    )
    print(f"точность на тесте : {model.score(*test):.1%} ({len(test[1])} примеров)")
    print(f"графики : {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="обучение бинарного перцептрона")
    parser.add_argument(
        "--samples", type=int, default=200, help="число случайных точек"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="скорость обучения")
    parser.add_argument("--epochs", type=int, default=50, help="максимум эпох")
    parser.add_argument("--seed", type=int, default=42, help="зерно генератора")
    parser.add_argument(
        "--dataset", choices=("synthetic", "iris", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    samples: int = args.samples
    learning_rate: float = args.lr
    epochs: int = args.epochs
    seed: int = args.seed
    dataset: str = args.dataset
    output: Path = args.output

    try:
        if dataset in ("synthetic", "all"):
            data = make_synthetic(samples, seed)
            _run_experiment("synthetic", data, learning_rate, epochs, seed, output)
        if dataset in ("iris", "all"):
            _run_experiment("iris", make_iris(), learning_rate, epochs, seed, output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"ошибка : {error}\n")
