from jobSystem import *
from distributions import *
from simulator import *

if __name__ == "__main__":
    jobs = []
    ctr = 0

    nIterations = 100_000_000
    saveFilePathExistsStrategy = "skip"
    simulator = Simulator(
        granularity=5000,

        successThreshold=0.5,
        honestAssignmentDistribution=BernouilliDistribution(1),
        noiseDistribution=UniformDistribution(0.2, 0.4),

        guessesHonestThreshold=UpdatableThreshold(granularity=5000),
        honestThresholdSensitivity=1,

        honestSuccessSensitivity=1,
        honestAvoidsEffortSensitivity=1,
        honestPerceptionSensitivity=0,
        dishonestFailureSensitivity=1,
        dishonestPerceptionSensitivity=1
    )

    def addJob(saveFilePathPrefix: str, **kwargs):
        global ctr
        ctr += 1
        jobs.append(Job(nIterations=nIterations,
                        simulator=simulator.copyWith(**kwargs),
                        saveFilePath=f"./output_effort/{saveFilePathPrefix}_{ctr}.txt",
                        saveFilePathExistsStrategy=saveFilePathExistsStrategy))

    """
    1 noise
    2 success threshold
    3 success/effort
    4 honest assigment decreases
    5 dishonest/dishonest
    6 honest perception
    7 noise
    8 success threshold
    9 success/effort
    """

    POST_DECREASE_HONEST_ASSIGNMENT = 0.5

    for noiseMin in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for noiseMax in [0.2, 0.4, 0.6, 0.8]:
            if noiseMin < noiseMax:
                addJob(f"1_noise_{noiseMin}_{noiseMax}",
                       noiseDistribution=(noiseMin, noiseMax))

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        addJob(f"2_successThreshold_{threshold}", successThreshold=threshold)

    for (success, effort) in [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                              (1, 0), (1, 2), (3, 2), (1, 3), (2, 3), (4, 3),
                              (1, 4), (3, 4)]:
        addJob(f"3_successEffort_{success}_{effort}",
               honestSuccessSensitivity=success,
               honestAvoidsEffortSensitivity=effort)

    for honestAssignment in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        addJob(f"4_honestAssignment_{honestAssignment}",
               honestAssignmentDistribution=honestAssignment)

    for (failure, perception) in [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                              (1, 0), (1, 2), (3, 2), (1, 3), (2, 3), (4, 3),
                              (1, 4), (3, 4)]:
        addJob(f"5_failurePerception_{failure}_{perception}",
               honestAssignmentDistribution=POST_DECREASE_HONEST_ASSIGNMENT,
               dishonestFailureSensitivity=failure,
               dishonestPerceptionSensitivity=perception)

    for perception in [0, 1, 2, 3, 4]:
        addJob(f"6_honestPerception_{perception}",
               honestAssignmentDistribution=POST_DECREASE_HONEST_ASSIGNMENT,
               honestPerceptionSensitivity=perception)

    for noiseMin in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for noiseMax in [0.2, 0.4, 0.6, 0.8]:
            if noiseMin < noiseMax:
                addJob(f"7_post_noise_{noiseMin}_{noiseMax}",
                       honestAssignmentDistribution=POST_DECREASE_HONEST_ASSIGNMENT,
                       noiseDistribution=(noiseMin, noiseMax))

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        addJob(f"8_post_successThreshold_{threshold}",
               honestAssignmentDistribution=POST_DECREASE_HONEST_ASSIGNMENT,
               successThreshold=threshold)

    for (success, effort) in [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                              (1, 0), (1, 2), (3, 2), (1, 3), (2, 3), (4, 3),
                              (1, 4), (3, 4)]:
        addJob(f"9_post_successEffort_{success}_{effort}",
               honestAssignmentDistribution=POST_DECREASE_HONEST_ASSIGNMENT,
               honestSuccessSensitivity=success,
               honestAvoidsEffortSensitivity=effort)
    


    JobSystem.run(jobs)
