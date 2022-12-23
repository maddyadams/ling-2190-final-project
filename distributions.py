from __future__ import annotations
import random
import numpy as np

class BernouilliDistribution:
    """
    Static distribution that returns True with probability p,
    and False with probability 1 - p
    """

    def __init__(self, p: float):
        self.p = p

    def __repr__(self) -> str:
        return f"BernouilliDistribution({self.p})"

    def __str__(self) -> str:
        return f"BernouilliDistribution({self.p:.4f})"

    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.p

class UniformDistribution:
    """
    Static distribution that returns a number chosen uniformly at random
    from [minValue, maxValue]
    """
    def __init__(self, minValue: float, maxValue: float):
        self.minValue = minValue
        self.maxValue = maxValue

    def __repr__(self) -> str:
        return f"UniformDistribution({self.minValue}, {self.maxValue})"

    def __str__(self) -> str:
        return f"UniformDistribution({self.minValue:.4f}, {self.maxValue:.4f})"

    def sample(self) -> float:
        return random.uniform(self.minValue, self.maxValue)

    def getMass(self, granularity: int) -> list[float]:
        return [self.minValue + x * (self.maxValue - self.minValue) / granularity
                for x in range(granularity)]

class UpdatableThreshold:
    """
    A threshold, clamped to the range [minValue, maxValue],
    that can be increased or decreased in multiples of granularity
    """

    def __init__(self, granularity: float, initialValue: float = 0.5,
                 minValue: float = 0, maxValue: float = 1):
        self.granularity = granularity
        self.threshold = initialValue
        self.minValue = minValue
        self.maxValue = maxValue

    def __repr__(self) -> str:
        return (
            f"UpdatableThreshold({self.granularity}, {self.threshold}, "
            f"{self.minValue}, {self.maxValue})"
        )

    def __str__(self) -> str:
        return (
            f"UpdatableThreshold({self.granularity}, {self.threshold:.4f}, "
            f"{self.minValue:.4f}, {self.maxValue:.4f})"
        )
    
    def increase(self, byMultiple: float):
        self.threshold += byMultiple / self.granularity

    def decrease(self, byMultiple: float):
        self.threshold -= byMultiple / self.granularity

    def valuePasses(self, value: float) -> bool:
        return self.threshold <= value

class UpdatableDistribution:
    """
    An arbitrary probability distribution that allows shifting weight towards 1 or 0. 
    After calling `sample()`, calls to `increase()` or `decrease()` will shift the
    weight that contributed to the value returned by `sample()` by a multiple of granularity
    until the next time `sample()` is called. 
    """

    def __init__(self, granularity: int):
        self.granularity = granularity
        self.mass = [i / self.granularity for i in range(self.granularity)]
        self.sampledIndex = None

    @staticmethod
    def fromMass(granularity: int, mass: list[float]) -> UpdatableDistribution:
        result = UpdatableDistribution(granularity)
        result.mass = mass
        return result

    def __repr__(self) -> string:
        return f"UpdatableDistribution.fromMass({self.granularity}, {self.mass})"

    def sample(self) -> float:
        self.sampledIndex = random.randrange(0, len(self.mass))
        return self.mass[self.sampledIndex]

    def _clamp(self):
        self.mass[self.sampledIndex] = min(1, self.mass[self.sampledIndex])
        self.mass[self.sampledIndex] = max(0, self.mass[self.sampledIndex])

    def increase(self, byMultiple: float):
        self.mass[self.sampledIndex] += byMultiple / self.granularity
        self._clamp()

    def decrease(self, byMultiple: float):
        self.mass[self.sampledIndex] -= byMultiple / self.granularity
        self._clamp()


