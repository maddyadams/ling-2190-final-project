import random
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ENABLE_VPRINT = False

def vprint(s: str):
    if ENABLE_VPRINT:
        print(s)

class UpdatableDistribution:
    def __init__(self, divisions: int):
        self.divisions = divisions
        self.weights = [i / (self.divisions - 1) for i in range(self.divisions)]
        self.k = None
    
    def sample(self) -> float:
        self.k = random.randrange(0, len(self.weights))
        return self.weights[self.k]

    def increase(self, amt: float):
        self.weights[self.k] = min(1, self.weights[self.k] + amt / (self.divisions - 1))

    def decrease(self, amt: float):
        self.weights[self.k] = max(0, self.weights[self.k] - amt / (self.divisions - 1))

    def getStats(self) -> str:
        l = []
        mean = sum(self.weights) / self.divisions
        variance = sum([(x - mean) * (x - mean) for x in self.weights]) / (self.divisions - 1)
        minimum = min(self.weights)
        maximum = max(self.weights)

        l.append(f"mean: {mean:.4f}")
        l.append(f"variance: {variance:.4f}")
        l.append(f"minimum: {minimum:.4f}")
        l.append(f"maximum: {maximum:.4f}")

        return "; ".join(l)

    def draw(self, ax, color, label):
        ax.hist(self.weights, bins=self.divisions, range=(0, 1),
                density=True, histtype="step",
                color=color, label=label)

class UpdatableThreshold:
    def __init__(self, initialValue: float, divisions: float):
        self.threshold = initialValue
        self.divisions = divisions
    
    def valuePasses(self, value: float) -> bool:
        return self.threshold <= value
    
    def increase(self, amt: float):
        self.threshold = min(1, self.threshold + amt / (self.divisions - 1))
    
    def decrease(self, amt: float):
        self.threshold = max(0, self.threshold - amt / (self.divisions - 1))

    def draw(self, ax, color, label, rect=None, divideBy2=False):
        x = self.threshold / (2 if divideBy2 else 1)
        if rect is None:
            ax.axvline(x, color=color, label=label)
        elif isinstance(rect, UniformDistribution):
            if rect.a == rect.b:
                ax.axvline(self.threshold, color=color, label=label)
            else:
                rect = patches.Rectangle((self.threshold, 0),
                                         rect.b - rect.a, 1,
                                         color=color)
                ax.add_patch(rect)
        else:
            raise ValueError(f"rect: {rect}")

class UniformDistribution:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    
    def sample(self) -> float:
        return random.uniform(self.a, self.b)

    def draw(self, ax, color, label):
        rect = patches.Rectangle((self.a, 0), self.b - self.a, 0.1, color=color)
        ax.add_patch(rect)

class BernouilliDistribution:
    def __init__(self, p: float):
        self.p = p
    
    def sample(self) -> bool:
        return random.uniform(0, 1) <= self.p

    def draw(self, ax, color, label):
        ax.axvline(self.p, color=color, label=label)

class CombinationFormula:
    def __init__(self, kind: str):
        self.kind = kind

    def combine(self, efforts: list[float], noise: float) -> float:
        if self.kind == "+":
            return sum(efforts) - noise
        elif self.kind == "*":
            return reduce(mul, efforts, 1 - noise)
        else:
            raise Exception(f"unknown kind {self.kind}")

class Simulation:
    def __init__(self, divisions: float, successThreshold: float,
                 combinationFormula: CombinationFormula,
                 honestGuessThresholdSensitivity: int,
                 honestSuccessSensitivity: int,
                 honestLowEffortSensitivity: int,
                 honestPerceptionSensitivity: int,
                 dishonestFailureSensitivity: int,
                 dishonestPerceptionSensitivity: int,

                 honestGuessThreshold: float, honestAssignmentProbability: float,
                 noiseDistribution: UniformDistribution
                 ):
        self.divisions = divisions
        self.successThreshold = UpdatableThreshold(successThreshold, self.divisions)
        self.combinationFormula = combinationFormula
        self.honestGuessThresholdSensitivity = honestGuessThresholdSensitivity
        self.honestSuccessSensitivity = honestSuccessSensitivity
        self.honestLowEffortSensitivity = honestLowEffortSensitivity
        self.honestPerceptionSensitivity = honestPerceptionSensitivity
        self.dishonestFailureSensitivity = dishonestFailureSensitivity
        self.dishonestPerceptionSensitivity = dishonestPerceptionSensitivity
        self.honestGuessThreshold = UpdatableThreshold(honestGuessThreshold, self.divisions)
        self.honestAssignmentProbability = BernouilliDistribution(honestAssignmentProbability)
        self.noiseDistribution = noiseDistribution

        self.honestDistribution = UpdatableDistribution(self.divisions)
        self.dishonestDistribution = UpdatableDistribution(self.divisions)


    def getStats(self) -> str:
        l = []
        l.append(f"honest guess threshold: {self.honestGuessThreshold.threshold}")
        l.append(f"honest distribution: {self.honestDistribution.getStats()}")
        l.append(f"dishonest distribution: {self.dishonestDistribution.getStats()}")

        return "\n".join(l)

    def draw(self, ax):
        ax.clear()
        self.successThreshold.draw(ax, "tab:red", "success threshold over 2", divideBy2=True)
        self.honestGuessThreshold.draw(ax, "tab:orange", "honest guess threshold", rect=self.noiseDistribution)
        self.honestAssignmentProbability.draw(ax, "tab:olive", "honest assignment probability")
        self.noiseDistribution.draw(ax, "tab:green", "noise distribution")
        self.honestDistribution.draw(ax, "tab:blue", "honest distribution")
        self.dishonestDistribution.draw(ax, "tab:pink", "dishonest distribution")
        ax.legend()
        plt.pause(0.001)

    def runManyTrials(self, nBatches, batchSize,
                      validationNBatches, drawPlot: bool=True):
        if drawPlot:
            fig = plt.figure()
            ax = plt.subplot()
            plt.ion()
            plt.show()
        
        for i in range(batchSize * nBatches):
            self.trial()
            if (i + 1) % batchSize == 0:
                print(f"{i + 1} / {batchSize * nBatches} ({100 * (i + 1) / (batchSize * nBatches)}%)")

                if drawPlot:
                    self.draw(ax)
                else:
                    print(self.getStats())
                    print("")

                # input()
        
        gameOutcomes = [self.gameOutcome() for _ in range(batchSize * validationNBatches)]
        honestHonestGames = [g for g in gameOutcomes if g[0] and g[1]]
        successfulHonestHonestGames = len([g for g in honestHonestGames if g[2]])
        successfulGames = len([g for g in gameOutcomes if g[2]])
        honestPerceivedHonest = len(
            [g for g in gameOutcomes if g[0] and g[4]]
            + [g for g in gameOutcomes if g[1] and g[3]]
        )
        honestPerceivedDishonest = len(
            [g for g in gameOutcomes if g[0] and not g[4]]
            + [g for g in gameOutcomes if g[1] and not g[3]]
        )
        dishonestPerceivedHonest = len(
            [g for g in gameOutcomes if not g[0] and g[4]]
            + [g for g in gameOutcomes if not g[1] and g[3]]
        )
        dishonestPerceivedDishonest = len(
            [g for g in gameOutcomes if not g[0] and not g[4]]
            + [g for g in gameOutcomes if not g[1] and not g[3]]
        )

        hhPercent = 100 * len(honestHonestGames) / (batchSize * validationNBatches)
        shhgPercent = 100 * successfulHonestHonestGames / len(honestHonestGames) if len(honestHonestGames) != 0 else "na"
        sPercent = 100 * successfulGames / (batchSize * validationNBatches)

        honestPrecision = "na"
        honestRecall = "na"
        honestFscore = "na"

        if honestPerceivedHonest != 0:
            honestPrecision = honestPerceivedHonest / (honestPerceivedHonest + dishonestPerceivedHonest)
            honestRecall = honestPerceivedHonest / (honestPerceivedHonest + honestPerceivedDishonest)
            honestFscore = 2 * honestPrecision * honestRecall / (honestPrecision + honestRecall)
        
        print(f"validationSize: {batchSize * validationNBatches}")
        print(f"hh games: {len(honestHonestGames)} ({hhPercent}% of games)")
        print(f"shh games: {successfulHonestHonestGames} ({shhgPercent}% of hhg)")
        print(f"s games: {successfulGames} ({sPercent}% of games)")
        print(f"hph: {honestPerceivedHonest}")
        print(f"hpd: {honestPerceivedDishonest}")
        print(f"dph: {dishonestPerceivedHonest}")
        print(f"dpd: {dishonestPerceivedDishonest}")
        print(f"hPrec (ie /dph): {honestPrecision}")
        print(f"hRec (ie /hpd): {honestRecall}")
        print(f"hFscore: {honestFscore}")

    def gameOutcome(self):
        # assign roles
        player1IsHonest = self.honestAssignmentProbability.sample()
        player2IsHonest = self.honestAssignmentProbability.sample()

        # communication occurs
        player1Effort = self.honestDistribution.sample() if player1IsHonest else self.dishonestDistribution.sample()
        player2Effort = self.honestDistribution.sample() if player2IsHonest else self.dishonestDistribution.sample()
        noiseEffort = self.noiseDistribution.sample()

        communicationEffort = self.combinationFormula.combine([player1Effort, player2Effort], noiseEffort)
        communicationSucceeds = self.successThreshold.valuePasses(communicationEffort)
        player1InformationForGuess = self.combinationFormula.combine([player2Effort], noiseEffort)
        player2InformationForGuess = self.combinationFormula.combine([player1Effort], noiseEffort)

        player1GuessesHonest = self.honestGuessThreshold.valuePasses(player1InformationForGuess)
        player2GuessesHonest = self.honestGuessThreshold.valuePasses(player2InformationForGuess)

        return (player1IsHonest, player2IsHonest,
                communicationSucceeds,
                player1GuessesHonest, player2GuessesHonest)
    
    def trial(self):
        (player1IsHonest, player2IsHonest,
         communicationSucceeds,
         player1GuessesHonest, player2GuessesHonest) = self.gameOutcome()

        # honest guess threshold updates
        if player2GuessesHonest and not player1IsHonest:
            self.honestGuessThreshold.increase(self.honestGuessThresholdSensitivity)
        elif not player2GuessesHonest and player1IsHonest:
            self.honestGuessThreshold.decrease(self.honestGuessThresholdSensitivity)

        # player 2 effort update
        if player2IsHonest:
            if communicationSucceeds:
                self.honestDistribution.decrease(self.honestLowEffortSensitivity)

            else:
                self.honestDistribution.increase(self.honestSuccessSensitivity)

            if not player1GuessesHonest:
                self.honestDistribution.increase(self.honestPerceptionSensitivity)

        else:
            if communicationSucceeds:
                self.dishonestDistribution.decrease(self.dishonestFailureSensitivity)

            if not player1GuessesHonest:
                self.dishonestDistribution.increase(self.dishonestPerceptionSensitivity)


if __name__ == "__main__":
    import time
    
    divisions = 500
    successThreshold = 0.5
    combinationFormula = CombinationFormula("+")
    honestGuessThresholdSensitivity = 1
    honestSuccessSensitivity = 1
    honestLowEffortSensitivity = 0.5
    honestPerceptionSensitivity = 0
    dishonestFailureSensitivity = 1
    dishonestPerceptionSensitivity = 0.5

    honestGuessThreshold = 0.5
    honestAssignmentProbability = 0.5
    noiseDistribution = UniformDistribution(0, 0.5)

    nBatches = 1_000
    batchSize = 1000
    validationNBatches = 10

    startTime = time.time()
    
    Simulation(divisions, successThreshold, combinationFormula,
               honestGuessThresholdSensitivity,
               honestSuccessSensitivity, honestLowEffortSensitivity,
               honestPerceptionSensitivity,
               dishonestFailureSensitivity, dishonestPerceptionSensitivity,
               honestGuessThreshold, honestAssignmentProbability,
               noiseDistribution
               ).runManyTrials(nBatches, batchSize, validationNBatches)


    endTime = time.time()
    print(f"duration: {endTime - startTime}s")

    input()
