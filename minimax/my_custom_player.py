
import logging
from sample_players import DataPlayer
from collections import defaultdict, Counter
logger = logging.getLogger(__name__)
class CustomPlayer(DataPlayer):
    """ 
    Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        self.book = defaultdict(Counter)
        #self.randomAction(state)  
        #self.minimaxOrdering(state)
        self.minimaxNoOrdering(state)
    def randomAction(self, state):
        import random
        self.queue.put(random.choice(state.actions()))
    
    def minimaxNoOrdering(self, state):
        def minimax(state, depth):
            def min_value(state, alpha, beta, depth):
                nodesVisited = 1
                if state.terminal_test(): return (state.utility(self.player_id), nodesVisited)
                if depth <= 0: return (self.score(state), nodesVisited)
                value = float("inf")
                for action in state.actions():
                    newValue, childNodes = max_value(state.result(action), alpha, beta, depth - 1)
                    value = min(value, newValue)
                    nodesVisited += childNodes
                    if value < alpha:
                        return value, nodesVisited
                    beta = min(value, beta)
                return value, nodesVisited

            def max_value(state, alpha, beta, depth):
                nodesVisited = 1
                if state.terminal_test(): return (state.utility(self.player_id), nodesVisited)
                if depth <= 0: return (self.score(state), nodesVisited)
                value = float("-inf")
                for action in state.actions():
                    newValue, childNodes = min_value(state.result(action), alpha, beta, depth - 1)
                    value = max(value, newValue)
                    nodesVisited += childNodes
                    if value > beta:
                        return value, nodesVisited
                    alpha = max(value, alpha)
                return value, nodesVisited

            actionValues = []
            value, nodesVisited = min_value(state.result(action), float("-inf"), float("inf"), depth - 1)
            return value, nodesVisited
        totalNodesVisited = 0
        lastNodesVisited = 0
        for d in range(1, 10000):
            maxValue = None
            maxAction = None
            nodesAtDepth = 0
            for action in state.actions():
                value, nodesVisited = minimax(state, depth=d)
                nodesAtDepth += nodesVisited
                if value == float("inf"):
                    self.context = (state.ply_count, d, totalNodesVisited)
                    self.queue.put(action)
                    return
                if maxValue is None or value > maxValue:
                    maxValue = value
                    maxAction = action
            if nodesAtDepth == lastNodesVisited:
                return
            lastNodesVisited = nodesAtDepth
            totalNodesVisited += nodesAtDepth
            self.context = (state.ply_count, d, totalNodesVisited)
            self.queue.put(maxAction)
    
    
    def minimaxOrdering(self, state):
        def minimax(state, orderedActions, depth):
            def min_value(state, alpha, beta, depth):
                nodesVisited = 1
                if state.terminal_test(): return (state.utility(self.player_id), nodesVisited)
                if depth <= 0: return (self.score(state), nodesVisited)
                value = float("inf")
                for action in self.getActions(state, False):
                    newValue, childNodes = max_value(state.result(action), alpha, beta, depth - 1)
                    self.book[state][action] = newValue
                    value = min(value, newValue)
                    nodesVisited += childNodes
                    if value < alpha:
                        return value, nodesVisited
                    beta = min(value, beta)
                return value, nodesVisited

            def max_value(state, alpha, beta, depth):
                nodesVisited = 1
                if state.terminal_test(): return (state.utility(self.player_id), nodesVisited)
                if depth <= 0: return (self.score(state), nodesVisited)
                value = float("-inf")
                for action in self.getActions(state, True):
                    newValue, childNodes = min_value(state.result(action), alpha, beta, depth - 1)
                    self.book[state][action] = newValue
                    value = max(value, newValue)
                    nodesVisited += childNodes
                    if value > beta:
                        return value, nodesVisited
                    alpha = max(value, alpha)
                return value, nodesVisited

            totalNodeVisited = 0
            for action in orderedActions:
                self.book[state][action], nodesVisited = min_value(state.result(action), float("-inf"), float("inf"), depth - 1)
                totalNodeVisited += nodesVisited
            return self.getActions(state, True), totalNodeVisited
        #if self.handleFirstTwoPly(state):
            #return
        totalNodesVisited = 0
        lastNodesVisited = 0
        orderedActions = self.getActions(state, True)
        for d in range(1, 10000):
            orderedActions, nodesVisited = minimax(state, orderedActions, depth=d)
            if nodesVisited == lastNodesVisited:
                return
            lastNodesVisited = nodesVisited
            totalNodesVisited += nodesVisited
            self.context = (state.ply_count, d, totalNodesVisited)
            self.queue.put(orderedActions[0])

    def getActions(self, state, isMax):
        actions = self.book[state]
        if not actions:
            actions = state.actions()
        else:
            actions = sorted(actions, key=actions.get, reverse=isMax)
        defaultValue = float("-inf") if isMax else float("inf")
        self.book[state] = Counter({a: defaultValue for a in actions})
        return actions
    
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
