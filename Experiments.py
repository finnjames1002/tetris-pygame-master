import random
from shutil import move
from ANN import getLoss, main, getScores

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import scatter

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
    final_scores = []
    total_rewards = []

    x = 0.0
    for i in range(num_games):
        moves, loss, finalScore, reward = getLoss(i, x)
        avg_loss = sum(loss) / len(loss)  # Calculate average loss for the game
        avg_losses.append(avg_loss)
        avg_moves.append(moves)
        final_scores.append(finalScore)
        total_rewards.append(reward)

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

    plt.figure()
    # Plot a scatter graph of the final scores
    plt.scatter(game_numbers, final_scores, label='Final Score')
    plt.xlabel('Game')
    plt.ylabel('Final Score')
    plt.legend()
    plt.savefig('final_score.png')

    plt.figure()
    # Plot total rewards per game with a trend line
    plt.scatter(game_numbers, total_rewards, label='Total Reward')
    z = np.polyfit(game_numbers, total_rewards, 1)
    p = np.poly1d(z)
    plt.plot(game_numbers, p(game_numbers), "r--")
    plt.xlabel('Game')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('total_reward.png')
    
    plt.figure()
    # Plot average moves per game
    ax1 = plt.gca()  # Get the current Axes instance
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Average Moves', color='b')
    ax1.tick_params('y', colors='b')
    ax1.plot(game_numbers, avg_moves, color='b', label='Average Moves')
    z = np.polyfit(game_numbers, avg_moves, 1)
    p = np.poly1d(z)
    ax1.plot(game_numbers, p(game_numbers), "b--")

    # Plot total rewards per game
    ax2 = ax1.twinx()  # Create a second y-axis that shares the x-axis with ax1
    ax2.scatter(game_numbers, total_rewards, color='r', label='Total Reward')
    ax2.set_ylabel('Total Reward', color='r')
    ax2.tick_params('y', colors='r')
    z = np.polyfit(game_numbers, total_rewards, 1)
    p = np.poly1d(z)
    ax2.plot(game_numbers, p(game_numbers), "r--")

    # Add a legend for each y-axis
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.savefig('avg_moves_and_total_rewards.png')
    plt.close()

    # Standard deviation between the average moves and total rewards
    std_dev = np.std([avg_moves, total_rewards])
    print('Standard deviation:', std_dev)

if __name__ == '__main__':
    num_games = 50
    run_ann(num_games)