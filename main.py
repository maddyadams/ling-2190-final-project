import argparse
from jobSystem import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("jobFile")
    args = parser.parse_args()
    
    JobSystem.runJobFile(args.jobFile)
