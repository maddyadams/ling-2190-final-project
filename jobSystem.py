import json
import os
import multiprocessing
from simulator import *
from distributions import *

class Job:
    def __init__(self, nIterations: int, simulator: Simulator,
                 saveFilePath: str, saveFilePathExistsStrategy: str, verbose: bool=False):
        self.nIterations = nIterations
        self.simulator = simulator
        self.saveFilePath = saveFilePath
        self.saveFilePathExistsStrategy = saveFilePathExistsStrategy
        self.nTotalJobs = None
        self.verbose = verbose

    @staticmethod
    def _initializer(_nCompletedJobs):
        global nCompletedJobs
        nCompletedJobs = _nCompletedJobs
    
    def _run(self):
        global nCompletedJobs
        def onExit(action: str):
            with nCompletedJobs.get_lock():
                nCompletedJobs.value += 1
                print(f"{action} {nCompletedJobs.value} of {self.nTotalJobs} ({self.saveFilePath})")

        # load the job from the json...
        
        fileExisted = False

        if os.path.exists(self.saveFilePath):
            fileExisted = True
            if self.saveFilePathExistsStrategy == "skip":
                onExit("Skipping")
                return

            elif self.saveFilePathExistsStrategy == "resume":
                with open(self.saveFilePath) as f:
                    self.simulator = f.read()

        if fileExisted:
            if self.saveFilePathExistsStrategy == "resume":
                print(f"Resuming {self.saveFilePath}")
            else:
                print(f"Restarting {self.saveFilePath}")
        else:
            print(f"Starting {self.saveFilePath}")

        # run the simulation...

        for _ in range(self.nIterations):
            self.simulator.updateUsingGameOutcome(verbose=self.verbose)

        # save the simulation...

        with open(self.saveFilePath, "w") as f:
            f.write(repr(self.simulator))

        onExit("Completed")
        return

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
