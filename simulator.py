from __future__ import annotations
from distributions import *
import copy

class Simulator:
    """
    Holds the environment for one simulation, and can simulate the communication game
    """

    def copyWith(self, **kwargs) -> Simulator:
        result = copy.deepcopy(self)
        result.granularity = kwargs.get("granularity", self.granularity)
        result.successThreshold = kwargs.get("successThreshold", self.successThreshold)
        result.honestAssignmentDistribution.p = kwargs.get("honestAssignmentDistribution", self.honestAssignmentDistribution.p)
        result.noiseDistribution.minValue = kwargs.get("noiseDistribution", (self.noiseDistribution.minValue, self.noiseDistribution.maxValue))[0]
        result.noiseDistribution.maxValue = kwargs.get("noiseDistribution", (self.noiseDistribution.minValue, self.noiseDistribution.maxValue))[1]
        result.guessesHonestThreshold.granularity = kwargs.get("guessesHonestThreshold", self.guessesHonestThreshold.granularity)
        result.honestThresholdSensitivity = kwargs.get("honestThresholdSensitivity", self.honestThresholdSensitivity)
        result.honestSuccessSensitivity = kwargs.get("honestSuccessSensitivity", self.honestSuccessSensitivity)
        result.honestAvoidsEffortSensitivity = kwargs.get("honestAvoidsEffortSensitivity", self.honestAvoidsEffortSensitivity)
        result.honestPerceptionSensitivity = kwargs.get("honestPerceptionSensitivity", self.honestPerceptionSensitivity)
        result.dishonestFailureSensitivity = kwargs.get("dishonestFailureSensitivity", self.dishonestFailureSensitivity)
        result.dishonestPerceptionSensitivity = kwargs.get("dishonestPerceptionSensitivity", self.dishonestPerceptionSensitivity)

        return result
    
    def __init__(self,
                 granularity: float,
                 
                 successThreshold: float,
                 honestAssignmentDistribution: BernouilliDistribution,
                 noiseDistribution: UniformDistribution,
                 
                 guessesHonestThreshold: UpdatableThreshold,
                 honestThresholdSensitivity: float,
                 
                 honestSuccessSensitivity: float,
                 honestAvoidsEffortSensitivity: float,
                 honestPerceptionSensitivity: float,
                 dishonestFailureSensitivity: float,
                 dishonestPerceptionSensitivity: float,
                 
                 numberOfTrialsRun: int = 0,
                 honestDistribution: UpdatableDistribution = None,
                 dishonestDistribution: UpdatableDistribution = None):
        
        self.granularity = granularity
        
        self.successThreshold = successThreshold
        self.honestAssignmentDistribution = honestAssignmentDistribution
        self.noiseDistribution = noiseDistribution
        
        self.guessesHonestThreshold = guessesHonestThreshold
        self.honestThresholdSensitivity = honestThresholdSensitivity
 
        self.honestSuccessSensitivity = honestSuccessSensitivity
        self.honestAvoidsEffortSensitivity = honestAvoidsEffortSensitivity
        self.honestPerceptionSensitivity = honestPerceptionSensitivity
        self.dishonestFailureSensitivity = dishonestFailureSensitivity
        self.dishonestPerceptionSensitivity = dishonestPerceptionSensitivity
        
        self.numberOfTrialsRun = numberOfTrialsRun

        self.honestDistribution = honestDistribution or UpdatableDistribution(self.granularity)
        self.dishonestDistribution = dishonestDistribution or UpdatableDistribution(self.granularity)

    def __repr__(self) -> str:
        return (
            f"Simulator({self.granularity},\n"
            f"{self.successThreshold},\n"
            f"{self.honestAssignmentDistribution},\n"
            f"{self.noiseDistribution},\n"
            f"{self.guessesHonestThreshold},\n"
            f"{self.honestThresholdSensitivity},\n"
            f"{self.honestSuccessSensitivity},\n"
            f"{self.honestAvoidsEffortSensitivity},\n"
            f"{self.honestPerceptionSensitivity},\n"
            f"{self.dishonestFailureSensitivity},\n"
            f"{self.dishonestPerceptionSensitivity},\n"
            f"{self.numberOfTrialsRun},\n"
            f"{self.honestDistribution},\n"
            f"{self.dishonestDistribution},)"
        )

    def getSingleGameOutcome(self,
                             player1IsHonest: bool = None,
                             player2IsHonest: bool = None):
        """
        Pass `player1IsHonest` or `player2IsHonest` to force a specific assignment

        returns: (player1IsHonest: bool,
                  player2IsHonest: bool,
                  communicationSucceeds: bool,
                  player1GuessesHonest: bool,
                  player2GuessesHonest: bool)

        """

        # note: due to the way sampling from an UpdatableDistribution works,
        # we will only end up doing a "half-update" due to how player 2 acts and is perceived

        if player1IsHonest is None:
            player1IsHonest = self.honestAssignmentDistribution.sample()
        if player2IsHonest is None:
            player2IsHonest = self.honestAssignmentDistribution.sample()

        player1Effort = (self.honestDistribution if player1IsHonest else self.dishonestDistribution).sample()
        player2Effort = (self.honestDistribution if player2IsHonest else self.dishonestDistribution).sample()
        noiseAmount = self.noiseDistribution.sample()

        communicationValue = player1Effort + player2Effort - noiseAmount
        communicationSucceeds = self.successThreshold < communicationValue
        player1InformationForGuess = player2Effort - noiseAmount
        player2InformationForGuess = player1Effort - noiseAmount
        player1GuessesHonest = self.guessesHonestThreshold.valuePasses(player1InformationForGuess)
        player2GuessesHonest = self.guessesHonestThreshold.valuePasses(player2InformationForGuess)
        return (
            player1IsHonest,
            player2IsHonest,
            communicationSucceeds,
            player1GuessesHonest,
            player2GuessesHonest
        )


    def updateUsingGameOutcome(self, verbose: bool = False):
        self.numberOfTrialsRun += 1
        if verbose:
            print(self.numberOfTrialsRun)
        
        (player1IsHonest,
         player2IsHonest,
         communicationSucceeds,
         player1GuessesHonest,
         player2GuessesHonest) = self.getSingleGameOutcome()

        # note: due to the way sampling from an UpdatableDistribution works,
        # we only end up doing a "half-update" due to how player 2 acts and is perceived

        if player2GuessesHonest and not player1IsHonest:
            self.guessesHonestThreshold.increase(self.honestThresholdSensitivity)
        elif not player2GuessesHonest and player1IsHonest:
            self.guessesHonestThreshold.decrease(self.honestThresholdSensitivity)

        if player2IsHonest:
            if communicationSucceeds:
                self.honestDistribution.decrease(self.honestAvoidsEffortSensitivity)

            else:
                self.honestDistribution.increase(self.honestSuccessSensitivity)

            if not player1GuessesHonest:
                self.honestDistribution.increase(self.honestPerceptionSensitivity)

        else:
            if communicationSucceeds:
                self.dishonestDistribution.decrease(self.dishonestFailureSensitivity)

            if not player1GuessesHonest:
                self.dishonestDistribution.increase(self.dishonestPerceptionSensitivity)

         
