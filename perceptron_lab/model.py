"""пороговый перцептрон с обучением на размеченных примерах"""

from collections.abc import Sequence
from math import isfinite


class Perceptron:
    # классифицирует объект по заданному количеству признаков

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        if type(n_features) is not int or n_features < 1:
            raise ValueError("количество признаков должно быть положительным целым")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("количество эпох должно быть положительным целым")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("скорость обучения должна быть конечной и положительной")

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs

        # каждому признаку соответствует один вес
        self.w: list[float] = [0.0] * n_features
        self.bias = 0.0
        self.errors: list[int] = []

    @staticmethod
    def activate(z: float) -> int:
        # нулевая сумма относится к классу 1
        return int(z >= 0)

    def predict(self, x: Sequence[float]) -> int:
        if len(x) != self.n_features:
            raise ValueError(f"ожидается признаков : {self.n_features}")
        if any(not isfinite(feature) for feature in x):
            raise ValueError("признаки должны быть конечными числами")

        # считаем взвешенную сумму со смещением
        z = self.bias + sum(
            weight * feature for weight, feature in zip(self.w, x, strict=True)
        )
        if not isfinite(z):
            raise ValueError("взвешенная сумма вышла за пределы конечных чисел")
        return self.activate(z)

    def train(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        self._validate_dataset(X, y)

        # каждый вызов начинает обучение с нуля
        self.w = [0.0] * self.n_features
        self.bias = 0.0
        self.errors.clear()

        for _ in range(self.epochs):
            mistakes = 0
            for x, target in zip(X, y, strict=True):
                error = target - self.predict(x)
                if error == 0:
                    continue

                # обновляем параметры только при ошибке
                mistakes += 1
                step = self.learning_rate * error
                for i, feature in enumerate(x):
                    self.w[i] += step * feature
                self.bias += step

            # сохраняем число ошибок и останавливаемся после чистой эпохи
            self.errors.append(mistakes)
            if mistakes == 0:
                break

    def score(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        self._validate_dataset(X, y)
        correct = sum(self.predict(x) == target for x, target in zip(X, y, strict=True))
        return correct / len(y)

    @staticmethod
    def _validate_dataset(X: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        # проверяем размер выборки и допустимые метки
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("выборка и метки должны быть непустыми и одной длины")
        if any(type(label) is not int or label not in (0, 1) for label in y):
            raise ValueError("метки классов должны быть целыми числами 0 или 1")
