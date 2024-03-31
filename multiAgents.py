# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        evaluation = 0

        # Distance to the nearest food pellet
        distances_to_food = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        if distances_to_food:
            min_distance_to_food = min(distances_to_food)
            evaluation += 1 / min_distance_to_food

        # Distance to ghosts
        for i in range(len(newGhostStates)):
            ghostState = newGhostStates[i]
            scaredTime = newScaredTimes[i]
            ghost_position = ghostState.getPosition()
            distance_to_ghost = util.manhattanDistance(newPos, ghost_position)
            if scaredTime == 0:
                # Ghost is not scared
                    if distance_to_ghost <= 1:
                        evaluation -= 100
            else:
                # Ghost is scared, Pacman can go closer to it
                evaluation += 15 / (distance_to_ghost + 1)

        # Remaining food pellets
        evaluation -= len(newFood.asList())

        # Increase evaluation if new state will increase score
        evaluation += successorGameState.getScore()

        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:  # Pacman's turn, maximize score
                value = float("-inf")
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    newValue, newAction = minimax(1, depth, successorGameState)  # Next agent is a ghost
                    if newValue > value:
                        value, bestAction = newValue, action
                return value, bestAction
            else:  # Ghosts' turn, minimize score
                value = float("inf")
                bestAction = None
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents(): # Pacman is next, all agents have gone
                    nextAgent = 0
                    depth -= 1  # Decrement depth after all agents have had their turn
                for action in gameState.getLegalActions(agentIndex):
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    newValue, newAction = minimax(nextAgent, depth, successorGameState)
                    if newValue < value:
                        value, bestAction = newValue, action
                return value, bestAction

        _, action = minimax(0, self.depth, gameState)  # Start from Pacman
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        Here are some method calls that might be useful.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(gameState, alpha, beta, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            if agentIndex == 0: # Pacman's turn, maximize score
                v = float("-inf")
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    newValue, newAction = alphaBeta(successorGameState, alpha, beta,depth, 1)  # Next agent is a ghost
                    if newValue > v:
                        bestAction = action
                        v = newValue
                    if v > beta:
                        bestAction = action
                        return v, bestAction
                    alpha = max(alpha, v)
                return v, bestAction
            else: # Ghosts' turn, minimize score
                v = float("inf")
                bestAction = None
                nextAgent = agentIndex + 1
                if nextAgent >= gameState.getNumAgents(): # Pacman is next, all agents have gone
                    nextAgent = 0
                    depth -= 1  #all agents have had their turn, depth decrements
                for action in gameState.getLegalActions(agentIndex):
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    newValue, newAction = alphaBeta(successorGameState, alpha, beta, depth, nextAgent)
                    if newValue < v:
                        bestAction = action
                        v = newValue
                    if v < alpha:
                        bestAction = action
                        return v, bestAction
                    beta = min(beta, v)
                return v, bestAction
        
        _, action = alphaBeta(gameState, float("-inf"), float("inf"), self.depth, 0)  # Start with Pacman
        return action
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Call the expectimax function
        _, action = self.expectimax(gameState, self.depth, 0)  # Start from Pacman
        return action
    
    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:  # Pacman's turn, maximize score
            v = float("-inf")
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newValue, _ = self.expectimax(successorGameState, depth, 1)  # Next agent is a ghost
                if newValue > v:
                    bestAction = action
                    v = newValue
            return v, bestAction
        else:  # Ghosts' turn, average score
            averageValue = 0
            numActions = 0
            nextAgent = agentIndex + 1
            if nextAgent >= gameState.getNumAgents():  # Pacman is next, all agents have gone
                nextAgent = 0
                depth -= 1  # All agents have had their turn, depth decrements
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                newValue, _ = self.expectimax(successorGameState, depth, nextAgent)
                averageValue += newValue
                numActions += 1
            return averageValue / numActions, None

        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    # Initialize evaluation score
    evaluation = 0

    # Evaluate distance to the nearest food pellet
    distances_to_food = [util.manhattanDistance(curPos, food) for food in curFood.asList()]
    if distances_to_food:
        min_distance_to_food = min(distances_to_food)
        evaluation += 1.0 / min_distance_to_food

    # Evaluate proximity to ghosts
    for ghostState, scaredTime in zip(curGhostStates, curScaredTimes):
        ghost_position = ghostState.getPosition()
        distance_to_ghost = util.manhattanDistance(curPos, ghost_position)
        if scaredTime == 0:
            # Ghost is not scared, so Pacman should avoid it
            if distance_to_ghost < 2:
                    evaluation -= 100
        else:
            # Ghost is scared, Pacman can approach it
            evaluation += 10.0 / (distance_to_ghost + 1)

    # Consider remaining food pellets
    evaluation -= len(curFood.asList())

    # Consider game score
    evaluation += currentGameState.getScore()

    return evaluation
    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
