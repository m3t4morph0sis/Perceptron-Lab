# генерация синтетических точек и подготовка данных iris

from collections.abc import Iterable, Sequence
from random import Random
from typing import cast

from sklearn.datasets import load_iris

type Point = tuple[float, float]
type Dataset = tuple[list[Point], list[int]]


def make_synthetic(samples: int, seed: int = 42) -> Dataset:
    if samples < 4:
        raise ValueError("укажи не менее четырёх синтетических точек")

    rng = Random(seed)
    points: list[Point] = []
    labels: list[int] = []

    # классы лежат по разные стороны от прямой x1 = 0
    for i in range(samples):
        label = i % 2
        center = -3.0 if label == 0 else 3.0
        points.append((center + rng.uniform(-1.0, 1.0), rng.uniform(-2.0, 2.0)))
        labels.append(label)

    return points, labels


def make_iris() -> Dataset:
    iris = load_iris()
    rows = cast("Iterable[Sequence[float]]", iris.data)
    targets = cast("Iterable[int]", iris.target)
    points: list[Point] = []
    labels: list[int] = []

    # берём два вида и два признака : длину и ширину лепестка
    for row, target in zip(rows, targets, strict=True):
        label = int(target)
        if label in (0, 1):
            points.append((float(row[2]), float(row[3])))
            labels.append(label)

    return points, labels


def split_dataset(
    points: list[Point], labels: list[int], seed: int = 42
) -> tuple[Dataset, Dataset]:
    rng = Random(seed)
    rows = list(zip(points, labels, strict=True))
    train: list[tuple[Point, int]] = []
    test: list[tuple[Point, int]] = []

    # делим каждый класс отдельно примерно в пропорции 65/35
    for label in (0, 1):
        group = [row for row in rows if row[1] == label]
        if len(group) < 2:
            raise ValueError("для разбиения нужно минимум два примера каждого класса")
        rng.shuffle(group)
        cut = int(0.65 * len(group))
        train.extend(group[:cut])
        test.extend(group[cut:])

    rng.shuffle(train)
    rng.shuffle(test)
    return _unpack(train), _unpack(test)


def _unpack(rows: list[tuple[Point, int]]) -> Dataset:
    # раскладываем точки и их метки по отдельным спискам
    return [point for point, _ in rows], [label for _, label in rows]
