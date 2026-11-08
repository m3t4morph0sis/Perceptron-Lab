"""тесты предсказания и обучения перцептрона"""

from perceptron_lab.datasets import make_iris, make_synthetic, split_dataset
from perceptron_lab.model import Perceptron


def test_activate() -> None:
    # проверяем порог с обеих сторон и на нуле
    assert Perceptron.activate(-1.0) == 0
    assert Perceptron.activate(0.0) == 1
    assert Perceptron.activate(1.0) == 1


def test_initial_weights() -> None:
    # число весов должно совпадать с числом признаков
    model = Perceptron(n_features=3)
    assert model.w == [0.0, 0.0, 0.0]
    assert model.bias == 0.0


def test_predict_with_known_weights() -> None:
    # проверяем предсказание с заранее заданными параметрами
    model = Perceptron(n_features=2)
    model.w = [1.0, -1.0]
    model.bias = -0.5

    assert model.predict((1.0, 4.0)) == 0
    assert model.predict((5.0, 2.0)) == 1


def test_training_separates_synthetic_classes() -> None:
    # проверяем обучение на двух линейно разделимых классах
    train, test = split_dataset(*make_synthetic(200, seed=42), seed=42)
    model = Perceptron(n_features=2)
    model.train(*train)

    assert model.score(*train) == 1.0
    assert model.score(*test) == 1.0
    assert model.errors[-1] == 0


def test_training_on_iris() -> None:
    # проверяем точность на цветах, которые модель не видела при обучении
    train, test = split_dataset(*make_iris(), seed=42)
    model = Perceptron(n_features=2, epochs=100)
    model.train(*train)

    assert model.score(*train) == 1.0
    assert model.score(*test) >= 0.95
