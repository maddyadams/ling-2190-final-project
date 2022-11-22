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
    
    def sample(self) -> (float, int):
        k = random.randrange(0, len(self.weights))
        return self.weights[k], k

    def increase(self, k: int):
        self.weights[k] = min(1, self.weights[k] + 1 / (self.divisions - 1))

    def decrease(self, k: int):
        self.weights[k] = max(0, self.weights[k] - 1 / (self.divisions - 1))

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
    
    def increase(self):
        self.threshold = min(1, self.threshold + 1 / (self.divisions - 1))
    
    def decrease(self):
        self.threshold = max(0, self.threshold - 1 / (self.divisions - 1))

    def draw(self, ax, color, label):
        ax.axvline(self.threshold, color=color, label=label)

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
                 failureOverridesHiding: bool, successOverridesPerception: bool,
                 honestGuessThreshold: float, honestAssignmentProbability: float,
                 noiseDistribution: UniformDistribution
                 ):
        self.divisions = divisions
        self.successThreshold = UpdatableThreshold(successThreshold, self.divisions)
        self.combinationFormula = combinationFormula
        self.failureOverridesHiding = failureOverridesHiding
        self.successOverridesPerception = successOverridesPerception
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
        self.successThreshold.draw(ax, "tab:red", "success threshold")
        self.honestGuessThreshold.draw(ax, "tab:orange", "honest guess threshold")
        self.honestAssignmentProbability.draw(ax, "tab:olive", "honest assignment probability")
        self.noiseDistribution.draw(ax, "tab:green", "noise distribution")
        self.honestDistribution.draw(ax, "tab:blue", "honest distribution")
        self.dishonestDistribution.draw(ax, "tab:pink", "dishonest distribution")
        ax.legend()
        plt.pause(0.001)

    def runManyTrials(self, nBatches, batchSize, drawPlot: bool=True):
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
    
    def trial(self):
        # assign roles
        player1IsHonest = self.honestAssignmentProbability.sample()
        player2IsHonest = self.honestAssignmentProbability.sample()

        vprint(f"roles: 1honest: {player1IsHonest}, 2honest: {player2IsHonest}")
        
        # communication occurs
        player1Effort, player1Key = self.honestDistribution.sample() if player1IsHonest else self.dishonestDistribution.sample()
        player2Effort, player2Key = self.honestDistribution.sample() if player2IsHonest else self.dishonestDistribution.sample()        
        noiseEffort = self.noiseDistribution.sample()
        vprint(f"efforts: 1: {player1Effort}, 2: {player2Effort}, noise: {noiseEffort}")
        communicationEffort = self.combinationFormula.combine([player1Effort, player2Effort], noiseEffort)
        communicationSucceeds = self.successThreshold.valuePasses(communicationEffort)
        vprint(f"communication: {communicationEffort}, threshold: {self.successThreshold.threshold}, success: {communicationSucceeds}")
        
        player1InformationForGuess = self.combinationFormula.combine([player2Effort], noiseEffort)
        player2InformationForGuess = self.combinationFormula.combine([player1Effort], noiseEffort)
        vprint(f"info: 1: {player1InformationForGuess}, 2: {player2InformationForGuess}, threshold: {self.honestGuessThreshold.threshold}")
        
        player1GuessesHonest = self.honestGuessThreshold.valuePasses(player1InformationForGuess)
        player2GuessesHonest = self.honestGuessThreshold.valuePasses(player2InformationForGuess)
        vprint(f"guesses: 1: {player1GuessesHonest}, 2: {player2GuessesHonest}")

        # honest guess threshold updates
        if player1GuessesHonest and not player2IsHonest:
            vprint(f"p1 wrong, increase")
            self.honestGuessThreshold.increase()
        elif not player1GuessesHonest and player2IsHonest:
            vprint("p1 wrong, decrease")
            self.honestGuessThreshold.decrease()

        if player2GuessesHonest and not player1IsHonest:
            vprint("p2 wrong, increase")
            self.honestGuessThreshold.increase()
        elif not player2GuessesHonest and player1IsHonest:
            vprint("p2 wrong, decrease")
            self.honestGuessThreshold.decrease()

        # player 1 effort update
        if player1IsHonest:
            if communicationSucceeds and player2GuessesHonest:
                vprint("honest decr")
                self.honestDistribution.decrease(player1Key)
            elif not communicationSucceeds and not player2GuessesHonest:
                vprint("honest incr")
                self.honestDistribution.increase(player1Key)
            elif communicationSucceeds and not player2GuessesHonest:
                if self.successOverridesPerception:
                    vprint("honest decr (sop)")
                    self.honestDistribution.decrease(player1Key)
                else:
                    vprint("honest incr (not sop)")
                    self.honestDistribution.increase(player1Key)
            elif not communicationSucceeds and player2GuessesHonest:
                if self.successOverridesPerception:
                    vprint("honest incr (sop)")
                    self.honestDistribution.increase(player1Key)
                else:
                    vprint("honest decr (not sop)")
                    self.honestDistribution.decrease(player1Key)
            else:
                raise Exception("impossible! player1IsHonest")
        else:
            if communicationSucceeds and player2GuessesHonest:
                vprint("dishonest decr")
                self.dishonestDistribution.decrease(player1Key)
            elif communicationSucceeds and not player2GuessesHonest:
                if self.failureOverridesHiding:
                    vprint("dishonest decr (foh)")
                    self.dishonestDistribution.decrease(player1Key)
                else:
                    vprint("dishonest incr (not foh)")
                    self.dishonestDistribution.increase(player1Key)
            elif not communicationSucceeds and player2GuessesHonest:
                vprint("dishonest pass")
                pass
            elif not communicationSucceeds and not player2GuessesHonest:
                if self.failureOverridesHiding:
                    vprint("dishonest pass (foh)")
                    pass
                else:
                    vprint("dishonest incr (not foh)")
                    self.dishonestDistribution.increase(player1Key)
            else:
                raise Exception("impossible: not player1IsHonest")

        # player 2 effort update
        if player2IsHonest:
            if communicationSucceeds and player1GuessesHonest:
                self.honestDistribution.decrease(player2Key)
            elif not communicationSucceeds and not player1GuessesHonest:
                self.honestDistribution.increase(player2Key)
            elif communicationSucceeds and not player1GuessesHonest:
                if self.successOverridesPerception:
                    self.honestDistribution.decrease(player2Key)
                else:
                    self.honestDistribution.increase(player2Key)
            elif not communicationSucceeds and player1GuessesHonest:
                if self.successOverridesPerception:
                    self.honestDistribution.increase(player2Key)
                else:
                    self.honestDistribution.decrease(player2Key)
            else:
                raise Exception("impossible! player2IsHonest")
        else:
            if communicationSucceeds and player1GuessesHonest:
                self.dishonestDistribution.decrease(player2Key)
            elif communicationSucceeds and not player1GuessesHonest:
                if self.failureOverridesHiding:
                    self.dishonestDistribution.decrease(player2Key)
                else:
                    self.dishonestDistribution.increase(player2Key)
            elif not communicationSucceeds and player1GuessesHonest:
                pass
            elif not communicationSucceeds and not player1GuessesHonest:
                if self.failureOverridesHiding:
                    pass
                else:
                    self.dishonestDistribution.increase(player2Key)
            else:
                raise Exception("impossible: not player1IsHonest")


if __name__ == "__main__":
    import time
    
    divisions = 500
    successThreshold = 0.5
    combinationFormula = CombinationFormula("+")
    failureOverridesHiding = False
    successOverridesPerception = True
    honestGuessThreshold = 0.5
    honestAssignmentProbability = 0.5
    noiseDistribution = UniformDistribution(0, 0)

    nBatches = 1_000
    batchSize = 1000


    startTime = time.time()
    
    Simulation(divisions, successThreshold, combinationFormula,
               failureOverridesHiding, successOverridesPerception,
               honestGuessThreshold, honestAssignmentProbability,
               noiseDistribution
               ).runManyTrials(nBatches, batchSize)


    endTime = time.time()
    print(f"duration: {endTime - startTime}s")
