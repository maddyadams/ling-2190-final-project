from simulator import *
from distributions import *
import argparse
import matplotlib.pyplot as plt

def getStats(d: UpdatableDistribution) -> str:
    mean = sum(d.mass) / len(d.mass)
    variance = sum([(x - mean) * (x - mean) for x in d.mass]) / (len(d.mass) - 1)
    minValue = min(d.mass)
    maxValue = max(d.mass)

    return (
        f"mean: {mean:.4f}; "
        f"variance: {variance:.4f}; "
        f"min: {minValue:.4f}; "
        f"max: {maxValue:.4f}"
    )

def printAnalysis(s: Simulator, nTrials: int):
    hhGames = [s.getSingleGameOutcome(True, True) for _ in range(nTrials)]
    hdGames = [s.getSingleGameOutcome(True, False) for _ in range(nTrials)]

    hhSuccessPercent = 100 * sum([g[2] for g in hhGames]) / len(hhGames)
    hdRawFailurePercent = 100 * sum([not g[2] for g in hdGames]) / len(hdGames)
    hdFailHidePercent = 100 * sum([(not g[2]) and g[3] for g in hdGames]) / len(hdGames)
    
    honestPerceivedHonest = [g[3] for g in hhGames] + [g[4] for g in hhGames] + [g[4] for g in hdGames]
    honestPerceivedHonestDenom = len(honestPerceivedHonest)
    honestPerceivedHonest = sum(honestPerceivedHonest)
    
    dishonestPerceivedHonest = [g[3] for g in hdGames]
    dishonestPerceivedHonestDenom = len(dishonestPerceivedHonest)
    dishonestPerceivedHonest = sum(dishonestPerceivedHonest)
    
    honestPerceivedDishonest = [not g[3] for g in hhGames] + [not g[4] for g in hhGames] + [not g[4] for g in hdGames]
    honestPerceivedDishonestDenom = len(honestPerceivedDishonest)
    honestPerceivedDishonest = sum(honestPerceivedDishonest)
    
    honestPrecision = honestPerceivedHonest / (honestPerceivedHonest + dishonestPerceivedHonest)
    honestRecall = honestPerceivedHonest / (honestPerceivedHonest + honestPerceivedDishonest)
    honestFscore = 2 * honestPrecision * honestRecall / (honestPrecision + honestRecall)

    print(f"  Analysis:")
    print(f"    nTrials: {nTrials}")
    print(f"    hhSuccess: {hhSuccessPercent:.4f}%")
    print(f"    hdRawFailure: {hdRawFailurePercent:.4f}%")
    print(f"    hdFailHide: {hdFailHidePercent:.4f}%")
    print(f"    ")
    print(f"    honestPerceivedHonest: {honestPerceivedHonest} / {honestPerceivedHonestDenom} ({(100*honestPerceivedHonest/honestPerceivedHonestDenom):.4f}%)")
    print(f"    honestPerceivedDishonest: {honestPerceivedDishonest} / {honestPerceivedDishonestDenom} ({(100*honestPerceivedDishonest/honestPerceivedDishonestDenom):.4f}%)")
    print(f"    dishonestPerceivedHonest: {dishonestPerceivedHonest} / {dishonestPerceivedHonestDenom} ({(100*dishonestPerceivedHonest/dishonestPerceivedHonestDenom):.4f}%)")
    print(f"    ")
    print(f"    honestPrecision: {honestPrecision:.4f}")
    print(f"    honestRecall: {honestRecall:.4f}")
    print(f"    honestFscore: {honestFscore:.4f}")

def printInitialConditions(s: Simulator):
    print(f"  Environment:")
    print(f"    successThreshold: {simulator.successThreshold}")
    print(f"    honestAssignmentDistribution: {simulator.honestAssignmentDistribution}")
    print(f"    noiseDistribution: {simulator.noiseDistribution}")
    print(f"  Sensitivities:")
    print(f"    honestThreshold: {simulator.honestThresholdSensitivity}")
    print(f"    honestSuccess: {simulator.honestSuccessSensitivity}")
    print(f"    honestAvoidsEffort: {simulator.honestAvoidsEffortSensitivity}")
    print(f"    honestPerception: {simulator.honestPerceptionSensitivity}")
    print(f"    dishonestFailure: {simulator.dishonestFailureSensitivity}")
    print(f"    dishonestPerception: {simulator.dishonestPerceptionSensitivity}")
    print(f"  Results:")
    print(f"    numberOfTrialsRun: {simulator.numberOfTrialsRun:,}")
    print(f"    guessesHonestThreshold: {simulator.guessesHonestThreshold}")
    print(f"    honestDistribution: {getStats(simulator.honestDistribution)}")
    print(f"    dishonestDistribution: {getStats(simulator.dishonestDistribution)}")

def plotSimulator(s: Simulator):
    """
    guessesHonestThreshold - left and right edge?
    """
    fig = plt.figure()
    ax = plt.subplot()

    ax.hist(s.honestDistribution.mass, bins=s.honestDistribution.granularity,
            range=(0, 1), density=True, histtype="bar",
            label="Honest Distribution",
            alpha=0.8)
    ax.hist(s.dishonestDistribution.mass, bins=s.dishonestDistribution.granularity,
            range=(0, 1), density=True, histtype="bar",
            label="Dishonest Distribution",
            alpha=0.8)
    ax.hist(s.noiseDistribution.getMass(s.granularity * 10), bins=s.granularity,
            range=(0, 1), density=True, histtype="bar",
            label="Noise Distribution",
            alpha=0.8)
    
    ax.axvline(s.successThreshold, label="Success Threshold",
               color="tab:gray")
    ax.axvline(s.guessesHonestThreshold.threshold, label="Guesses Honest Threshold",
               color="tab:olive")

    ax.axvspan((s.successThreshold + s.noiseDistribution.minValue) / 2,
               (s.successThreshold + s.noiseDistribution.maxValue) / 2,
               label="Region of uncertain outcome",
               alpha=0.2,
               color="tab:purple")

    ax.axvspan(s.guessesHonestThreshold.threshold + s.noiseDistribution.minValue,
               s.guessesHonestThreshold.threshold + s.noiseDistribution.maxValue,
               label="Region of uncertain alignment",
               alpha=0.2,
               color="tab:red")

    ax.legend()
    text = (
        f"Honest assignment: {s.honestAssignmentDistribution.p}; "
        f"Threshold sensitivity: {s.honestThresholdSensitivity}\n"
        f"Success: {s.honestSuccessSensitivity} "
        f"Effort: {s.honestAvoidsEffortSensitivity} Perception: {s.honestPerceptionSensitivity}; "
        f"Failure: {s.dishonestFailureSensitivity} "
        f"Hiding: {s.dishonestPerceptionSensitivity}\n"
        f"{s.numberOfTrialsRun:,} trials"
    )

    plt.title(text)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile")
    args = parser.parse_args()

    with open(args.inputFile) as f:
        simulator = eval(f.read())
        
        print(f"{args.inputFile}:")
        printInitialConditions(simulator)
        printAnalysis(simulator, 10_000)
        plotSimulator(simulator)
