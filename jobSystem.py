import json
import os
import multiprocessing
from simulator import *
from distributions import *

def fetch(json, path, defaultValue=None):
    for p in path.split("."):
        if p in json:
            json = json[p]
        else:
            return defaultValue
    return json

class JobSystem:
    """
    Runs one or more jobs as described by a job file, writing them to disk. 
    Call JobSystem.runJobFile() instead of calling the initializer. 

    The job file should be an object with the following keys:
        metadata: an object with the following keys:
            nProcesses: int. Default: (multiprocessing.cpu_count() - 1) or 1
            nIterations: int. Optional
            saveFilePathExistsStrategy: str. Optional.
                Must be one of "skip", "restart", "resume". Default: "skip"
        jobs: list where each element is an object with the following keys:
            nIterations: int. If present, overrides the value in metadata
            setup: str or list[str]. Should `eval()` to a `Simulator`.
                (If a list, it will be joined with a comma before being `eval()`ed.)
            saveFilePath: str
            saveFilePathExistsStrategy: str. 
                Must be one of "skip", "restart", "resume".
                If present, overrides the strategy in metadata. Default: "skip"
    """
    
    def __init__(self, nTotalJobs, nCompletedJobs,
                 nIterations, saveFilePathExistsStrategy,
                 verbose):
        self.nTotalJobs = nTotalJobs
        self.nCompletedJobs = nCompletedJobs
        self.nIterations = nIterations
        self.saveFilePathExistsStrategy = saveFilePathExistsStrategy
        self.verbose = verbose

    @staticmethod
    def makeWorker(*args):
        global self
        self = JobSystem(*args)

    @staticmethod
    def runJobFile(jobFile: str):
        """
        jobFile: str. The path to the job file to run
        """
        
        print(f"Loading {jobFile}...")
        jobList = json.load(open(jobFile))
        
        nProcesses = fetch(jobList, "metadata.nProcesses",
                           (multiprocessing.cpu_count() - 1) or 1)
        nIterations = fetch(jobList, "metadata.nIterations")
        saveFilePathExistsStrategy = fetch(jobList, "metadata.saveFilePathExistsStrategy")
        verbose = fetch(jobList, "metadata.verbose")
        
        
        nTotalJobs = len(jobList["jobs"])
        nCompletedJobs = multiprocessing.Value("i", 0)
        print(f"Starting {nTotalJobs} jobs with {nProcesses} processes")

        initargs = (nTotalJobs,
                    nCompletedJobs,
                    nIterations,
                    saveFilePathExistsStrategy,
                    verbose)

        with multiprocessing.Pool(processes=nProcesses,
                                  initializer=JobSystem.makeWorker,
                                  initargs=initargs) as pool:
            pool.map(JobSystem.runJob, jobList["jobs"])

    def runJob(job):
        global self
        def onExit(action: str):
            with self.nCompletedJobs.get_lock():
                self.nCompletedJobs.value += 1
                print(f"{action} {self.nCompletedJobs.value} of {self.nTotalJobs} ({saveFilePath})")

        # load the job from the json...
        
        saveFilePath = job["saveFilePath"]
        self.saveFilePathExistsStrategy = fetch(job, "saveFilePathExistsStrategy", self.saveFilePathExistsStrategy)

        setup = None
        fileExisted = False

        if os.path.exists(saveFilePath):
            fileExisted = True
            if self.saveFilePathExistsStrategy == "skip":
                onExit("Skipping")
                return

            elif self.saveFilePathExistsStrategy == "resume":
                with open(saveFilePath) as f:
                    setup = f.read()

        if setup is None:
            # no previous save file, or restart strategy
            setup = job["setup"]
            if isinstance(setup, list):
                setup = ",".join(setup)

        setup = eval(setup)

        if fileExisted:
            if self.saveFilePathExistsStrategy == "resume":
                print(f"Resuming {saveFilePath}")
            else:
                print(f"Restarting {saveFilePath}")
        else:
            print(f"Starting {saveFilePath}")

        # run the simulation...

        for _ in range(fetch(job, "nIterations", self.nIterations)):
            setup.updateUsingGameOutcome(verbose=self.verbose)

        # save the simulation...

        with open(saveFilePath, "w") as f:
            f.write(repr(setup))

        onExit("Completed")
        return
