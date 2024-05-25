import random
from shutil import move
from ANN import getLoss, main, getScores

import matplotlib.pyplot as plt

def run_experiment(num_games):
    scores_agent1 = []
    scores_agent2 = []
    scores_agent3 = []
    game_numbers = list(range(1, num_games + 1))  # List of game numbers

    for _ in range(num_games):

        OptimalMoves = getScores(False,True)
        randomMoves = getScores(True,False)
        ANN = getScores(False,False)
        
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

def run_ann(num_games):
    avg_losses = []
    avg_moves = []
    game_numbers = list(range(1, num_games + 1))  # List of game numbers
    x = 1.0
    for i in range(num_games):
        moves, loss = getLoss(i, x)
        avg_loss = sum(loss) / len(loss)  # Calculate average loss for the game
        avg_losses.append(avg_loss)
        avg_moves.append(moves)

    plt.figure()
    plt.plot(game_numbers, avg_losses, label='Average Loss')  # Plot average loss per game
    plt.xlabel('Game')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig('avg_losses.png')

    plt.figure()
     # Plot average moves per game
    plt.plot(game_numbers, avg_moves, label='Average Moves')
    plt.xlabel('Game')
    plt.ylabel('Average Moves')
    plt.legend()
    plt.savefig('avg_moves.png')
    plt.close()


if __name__ == '__main__':
    num_games = 150
    run_ann(num_games)