"""графики границы решения и ошибок по эпохам"""

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from perceptron_lab.datasets import Dataset
from perceptron_lab.model import Perceptron


def save_plot(model: Perceptron, train: Dataset, test: Dataset, path: Path) -> None:
    # создаём два графика в одном изображении
    figure = Figure(figsize=(11, 4), layout="constrained")
    boundary_ax = figure.add_subplot(1, 2, 1)
    errors_ax = figure.add_subplot(1, 2, 2)

    _draw_points(boundary_ax, train, "обучение", "o")
    _draw_points(boundary_ax, test, "тест", "^")
    boundary_ax.set(
        title="граница решения",
        xlabel="первый признак",
        ylabel="второй признак",
    )
    _draw_boundary(boundary_ax, model)
    boundary_ax.legend(fontsize=8)

    epochs = list(range(1, len(model.errors) + 1))
    errors_ax.plot(epochs, model.errors, marker="o")
    errors_ax.set(title="ошибки по эпохам", xlabel="эпоха", ylabel="число ошибок")
    errors_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    errors_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    errors_ax.grid(alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", backend="agg")


def _draw_points(ax: Axes, dataset: Dataset, name: str, marker: str) -> None:
    # цвет показывает класс, а форма маркера показывает выборку
    points, labels = dataset
    for label, color in enumerate(("tab:blue", "tab:orange")):
        group = [
            point
            for point, target in zip(points, labels, strict=True)
            if target == label
        ]
        ax.scatter(
            [x1 for x1, _ in group],
            [x2 for _, x2 in group],
            color=color,
            marker=marker,
            alpha=0.7,
            label=f"{name}, класс {label}",
        )


def _draw_boundary(ax: Axes, model: Perceptron) -> None:
    # на границе взвешенная сумма со смещением равна нулю
    w1, w2 = model.w
    if w1 == 0 and w2 == 0:
        ax.set_title("модель предсказывает один класс")
        return

    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()

    # выражаем координату, у которой вес больше по модулю
    if abs(w1) >= abs(w2):
        ax.plot(
            [-(w2 * bottom + model.bias) / w1, -(w2 * top + model.bias) / w1],
            [bottom, top],
            "k--",
            label="граница модели",
        )
    else:
        ax.plot(
            [left, right],
            [-(w1 * left + model.bias) / w2, -(w1 * right + model.bias) / w2],
            "k--",
            label="граница модели",
        )

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
