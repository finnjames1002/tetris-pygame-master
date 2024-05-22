import random
from ANN import main, runAgents

import matplotlib.pyplot as plt

def run_experiment(num_games):
    scores_agent1 = []
    scores_agent2 = []
    scores_agent3 = []
    game_numbers = list(range(1, num_games + 1))  # List of game numbers

    for _ in range(num_games):

        OptimalMoves = runAgents(False,True)
        randomMoves = runAgents(True,False)
        ANN = runAgents(False,False)
        
        scores_agent1.append(randomMoves)
        scores_agent2.append(ANN)
        scores_agent3.append(OptimalMoves)

    plt.plot(game_numbers, scores_agent1, label='Agent 1')
    plt.plot(game_numbers, scores_agent2, label='Agent 2')
    plt.plot(game_numbers, scores_agent3, label='Agent 3')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('scores.png')
    plt.close()

if __name__ == '__main__':
    num_games = 10
    run_experiment(num_games)