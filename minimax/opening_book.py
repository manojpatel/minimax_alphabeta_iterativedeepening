import random
from isolation import Isolation

def build_table(num_rounds):
    from collections import defaultdict, Counter
    book = defaultdict(Counter)
    for _ in range(num_rounds):
        gameState = Isolation()
        buildTree(gameState, book, 3)
    return {k: max(v, key=v.get) for k, v in book.items()}

def buildTree(state, book, depth):
    if depth <= 0 or state.terminal_test():
        return simulation(state)
    action = random.choice(state.actions())
    print(state, action, state.result(action))
    reward = buildTree(state.result(action), book, depth - 1)
    book[state][action] += reward
    return -reward
    
def simulation(state):
    player_id = state.player()
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
    return 1 if state.utility(player_id) else -1

book = build_table(5)
#print(book)