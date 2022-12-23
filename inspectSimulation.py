from simulator import *
from distributions import *
import argparse
import matplotlib.pyplot as plt
import os
import multiprocessing

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

def getAnalysis(s: Simulator, nTrials: int) -> str:
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

    result = []
    result.append("Analysis:")
    result.append(f"nTrials: {nTrials}")
    result.append(f"hhSuccess: {hhSuccessPercent:.4f}%")
    result.append(f"hdRawFailure: {hdRawFailurePercent:.4f}%")
    result.append(f"hdFailHide: {hdFailHidePercent:.4f}%")
    result.append(f"")
    result.append(f"honestPerceivedHonest: {(100*honestPerceivedHonest/honestPerceivedHonestDenom):.4f}%")
    result.append(f"honestPerceivedDishonest: {(100*honestPerceivedDishonest/honestPerceivedDishonestDenom):.4f}%")
    result.append(f"dishonestPerceivedHonest: {(100*dishonestPerceivedHonest/dishonestPerceivedHonestDenom):.4f}%")
    result.append(f"")
    result.append(f"honestPrecision: {honestPrecision:.4f}")
    result.append(f"honestRecall: {honestRecall:.4f}")
    result.append(f"honestFscore: {honestFscore:.4f}")

    return "\n".join(result)

def getInitialConditions(simulator: Simulator) -> str:
    result = []
    result.append(f"Environment:")
    result.append(f"successThreshold: {simulator.successThreshold}")
    result.append(f"honestAssignmentProbability: {simulator.honestAssignmentDistribution.p}")
    
    result.append(f"noise: {simulator.noiseDistribution}")
    result.append(f"honestThresholdStepSize: {simulator.honestThresholdSensitivity}")
    result.append(f"")
    result.append(f"Sensitivities:")
    result.append(f"honestSuccess: {simulator.honestSuccessSensitivity}")
    result.append(f"honestAvoidsEffort: {simulator.honestAvoidsEffortSensitivity}")
    result.append(f"honestPerception: {simulator.honestPerceptionSensitivity}")
    result.append(f"dishonestFailure: {simulator.dishonestFailureSensitivity}")
    result.append(f"dishonestPerception: {simulator.dishonestPerceptionSensitivity}")
    result.append(f"")
    result.append(f"Results:")
    result.append(f"numberOfTrialsRun: {simulator.numberOfTrialsRun:,}")
    result.append(f"guessesHonestThreshold: {simulator.guessesHonestThreshold.threshold}")
    result.append(f"honestDistribution: {getStats(simulator.honestDistribution)}")
    result.append(f"dishonestDistribution: {getStats(simulator.dishonestDistribution)}")

    return "\n".join(result)

def plotSimulator(s: Simulator, inputFile: str, outputFileDir: str):
    """
    guessesHonestThreshold - left and right edge?
    """
    fig = plt.figure(figsize=(12, 4))
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
    plt.title(inputFile)


    subplotText = getInitialConditions(s) + "\n\n" + getAnalysis(s, 100_000)
    ax.text(1.02, 1, subplotText, va="top", wrap=True, fontsize=6, transform=ax.transAxes)

    plt.tight_layout()

    if outputFileDir is None:
        plt.show()
    else:
        outPath = os.path.join(outputFileDir, os.path.splitext(inputFile)[0] + ".png")
        outPathDir = os.path.dirname(outPath)
        if not os.path.exists(outPathDir):
            os.makedirs(outPathDir)
        plt.savefig(outPath, dpi=400)

class Job:
    def __init__(self, inputFilePath, outputFilePath):
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath
        self.nTotalJobs = None

    @staticmethod
    def _initializer(_nCompletedJobs):
        global nCompletedJobs
        nCompletedJobs = _nCompletedJobs
    
    def _run(self):
        global nCompletedJobs

        print(f"Starting {self.inputFilePath}")

        with open(self.inputFilePath) as f:
            simulator = eval(f.read())
            plotSimulator(simulator, self.inputFilePath, self.outputFilePath)
            with nCompletedJobs.get_lock():
                nCompletedJobs.value += 1
                print(f"Completed {nCompletedJobs.value} of {self.nTotalJobs} ({self.inputFilePath})")

class JobSystem:
    @staticmethod
    def run(jobs: list[Job], nProcesses: int = None):
        if nProcesses is None:
            nProcesses = (multiprocessing.cpu_count() - 1) or 1
        
        nCompletedJobs = multiprocessing.Value("i", 0)
        print(f"Starting {len(jobs)} jobs with {nProcesses} processes")

        for job in jobs:
            job.nTotalJobs = len(jobs)

        with multiprocessing.Pool(processes=nProcesses,
                                  initializer=Job._initializer,
                                  initargs=(nCompletedJobs,)) as pool:
            pool.map(Job._run, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", nargs="+")
    parser.add_argument("--outputFileDir", nargs="?", default=None)
    args = parser.parse_args()

    if len(args.inputFile) == 1:
        with open(inputFile) as f:
            simulator = eval(f.read())
            print(f"{args.inputFile[0]}:")
            print(getInitialConditions(simulator))
            print(getAnalysis(simulator, 10_000))

    else:
        JobSystem.run([Job(i, args.outputFileDir)
                       for i in args.inputFile])
