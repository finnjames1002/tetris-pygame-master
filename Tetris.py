import os
import random
import pygame
import copy
import random
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(607, 128)  # Assuming a 10x20 grid as input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # Assuming 4 possible actions

    def forward(self, x):
        x = x.view(-1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Individual:
    def __init__(self, parameters, fitness=0):
        self.parameters = parameters
        self.fitness = fitness

"""
10 x 20 grid
play_height = 2 * play_width

tetriminos:
    0 - S - green
    1 - Z - red
    2 - I - cyan
    3 - O - yellow
    4 - J - blue
    5 - L - orange
    6 - T - purple
"""

pygame.font.init()

# global variables

col = 10  # 10 columns
row = 20  # 20 rows
s_width = 800  # window width
s_height = 750  # window height
play_width = 300  # play window width; 300/10 = 30 width per block
play_height = 600  # play window height; 600/20 = 20 height per block
block_size = 30  # size of block

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height - 50

filepath = './highscore.txt'
fontpath = './arcade.ttf'
fontpath_mario = './mario.ttf'

# shapes formats

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['.....',
      '..0..',
      '..0..',
      '..0..',
      '..0..'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

# index represents the shape
shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

# class to represent each of the pieces


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]  # choose color from the shape_color list
        self.rotation = 0  # chooses the rotation according to index
    
    def getShape(self):
        return shapes.index(self.shape)  # Assuming self.shape is a list of all shapes and self.rotation is the current shape


# initialise the grid
def create_grid(locked_pos={}):
    grid = [[(0, 0, 0) for x in range(col)] for y in range(row)]  # grid represented rgb tuples

    # locked_positions dictionary
    # (x,y):(r,g,b)
    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                color = locked_pos[
                    (x, y)]  # get the value color (r,g,b) from the locked_positions dictionary using key (x,y)
                grid[y][x] = color  # set grid position to color

    return grid


def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]  # get the desired rotated shape from piece

    '''
    e.g.
       ['.....',
        '.....',
        '..00.',
        '.00..',
        '.....']
    '''
    for i, line in enumerate(shape_format):  # i gives index; line gives string
        row = list(line)  # makes a list of char from string
        for j, column in enumerate(row):  # j gives index of char; column gives char
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)  # offset according to the input given with dot and zero

    return positions


# checks if current position of piece in grid is valid
def valid_space(piece, grid):
    accepted_pos = [[(x, y) for x in range(col) if grid[y][x] == (0, 0, 0)] for y in range(row)]
    accepted_pos = [x for item in accepted_pos for x in item]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_pos:
            if pos[1] >= 0:
                return False
    return True

# check if piece is out of board
def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


# chooses a shape randomly from shapes list
def get_shape():
    return Piece(5, 0, random.choice(shapes))


# draws text in the middle
def draw_text_middle(text, size, color, surface):
    font = pygame.font.Font(None, size)  # Use default font
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width()/2), top_left_y + play_height/2 - (label.get_height()/2)))


# draws the lines of the grid for the game
def draw_grid(surface):
    r = g = b = 0
    grid_color = (r, g, b)

    for i in range(row):
        # draw grey horizontal lines
        pygame.draw.line(surface, grid_color, (top_left_x, top_left_y + i * block_size),
                         (top_left_x + play_width, top_left_y + i * block_size))
        for j in range(col):
            # draw grey vertical lines
            pygame.draw.line(surface, grid_color, (top_left_x + j * block_size, top_left_y),
                             (top_left_x + j * block_size, top_left_y + play_height))


# clear a row when it is filled
def clear_rows(grid, locked):
    # need to check if row is clear then shift every other row above down one
    increment = 0
    for i in range(len(grid) - 1, -1, -1):      # start checking the grid backwards
        grid_row = grid[i]                      # get the last row
        if (0, 0, 0) not in grid_row:           # if there are no empty spaces (i.e. black blocks)
            increment += 1
            # add positions to remove from locked
            index = i                           # row index will be constant
            for j in range(len(grid_row)):
                try:
                    del locked[(j, i)]          # delete every locked element in the bottom row
                except ValueError:
                    continue

    # shift every row one step down
    # delete filled bottom row
    # add another empty row on the top
    # move down one step
    if increment > 0:
        # sort the locked list according to y value in (x,y) and then reverse
        # reversed because otherwise the ones on the top will overwrite the lower ones
        for key in sorted(list(locked), key=lambda a: a[1])[::-1]:
            x, y = key
            if y < index:                       # if the y value is above the removed index
                new_key = (x, y + increment)    # shift position to down
                locked[new_key] = locked.pop(key)

    return increment


# draws the upcoming piece
def draw_next_shape(piece, surface):
    font = pygame.font.Font(fontpath, 30)
    label = font.render('Next shape', 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color, (start_x + j*block_size, start_y + i*block_size, block_size, block_size), 0)

    surface.blit(label, (start_x, start_y - 30))

    # pygame.display.update()


# draws the content of the window
def draw_window(surface, grid, current_piece, score=0, last_score=0, level=0):
    surface.fill((0, 0, 0))  # fill the surface with black

    pygame.font.init()  # initialise font
    font = pygame.font.Font(fontpath_mario, 65)  # Use specific font
    label = font.render('TETRIS', 1, (255, 255, 255))  # initialise 'Tetris' text with white

    surface.blit(label, ((top_left_x + play_width / 2) - (label.get_width() / 2), 30))  # put surface on the center of the window

    # current score
    font = pygame.font.Font(fontpath, 30)
    label = font.render('SCORE   ' + str(score) , 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    surface.blit(label, (start_x, start_y + 200))

    # last score
    label_hi = font.render('HIGHSCORE   ' + str(last_score), 1, (255, 255, 255))

    start_x_hi = top_left_x - 240
    start_y_hi = top_left_y + 200

    surface.blit(label_hi, (start_x_hi + 20, start_y_hi + 200))

    # draw level
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Level: ' + str(level), 1, (255,255,255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100

    surface.blit(label, (sx + 20, sy + 160))

    # draw content of the grid
    for i in range(row):
        for j in range(col):
            # pygame.draw.rect()
            # draw a rectangle shape
            # rect(Surface, color, Rect, width=0) -> Rect
            pygame.draw.rect(surface, grid[i][j],
                             (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)
    

    # create a copy of the grid that doesn't include the current piece
    grid_copy = copy.deepcopy(grid)
    for x, y in convert_shape_format(current_piece):
        if y > -1 and x < 10:
            grid_copy[y][x] = (0, 0, 0)

    # draw shadow piece
    shadow_piece = copy.deepcopy(current_piece)
    while valid_space(shadow_piece, grid_copy):
        shadow_piece.y += 1
    shadow_piece.y -= 1
    for x, y in convert_shape_format(shadow_piece):
        if y > -1:
            pygame.draw.rect(surface, (127, 127, 127), (top_left_x + x * block_size, top_left_y + y * block_size, block_size, block_size), 0)

    # draw vertical and horizontal grid lines
    draw_grid(surface)

    # draw rectangular border around play area
    border_color = (255, 255, 255)
    pygame.draw.rect(surface, border_color, (top_left_x, top_left_y, play_width, play_height), 4)

    # pygame.display.update()


# update the score txt file with high score
def update_score(new_score):
    score = get_max_score()

    with open(filepath, 'w') as file:
        if new_score > score:
            file.write(str(new_score))
        else:
            file.write(str(score))


# get the high score from the file
def get_max_score():
    with open(filepath, 'r') as file:
        lines = file.readlines()        # reads all the lines and puts in a list
        score = int(lines[0].strip())   # remove \n

    return score

def game_logic(window, parameters):
    locked_positions = {}
    grid = create_grid(locked_positions)
    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.01
    level = 1
    level_time = 0
    score = 0
    best_score = get_max_score()
    ai_enabled = True
    global epsilon
    totalReward = 0

    
    while run:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        reward = 0

        clock.tick()

        if level_time/6000 > 5:
            level_time = 0
            level += 1
            if fall_speed > 0.15:
                fall_speed -= 0.01

        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        if ai_enabled:
            best_move = [None, None]
            
            if current_piece.y >= 3 and current_piece.shape != I:
                best_move[0], best_move[1], reward = find_best_move(grid, current_piece, next_piece)
            elif current_piece.shape == I and current_piece.y >= 4:
                best_move[0], best_move[1], reward = find_best_move(grid, current_piece, next_piece)
            
            if best_move[0] is not None and best_move[1] is not None:
                current_piece.x = best_move[0]
                current_piece.rotation = best_move[1]

        totalReward += reward
        piece_pos = convert_shape_format(current_piece)

        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y > 0:
                grid[y][x] = current_piece.color

        
        if change_piece:  # if the piece is locked
            piece_pos = convert_shape_format(current_piece)
            for pos in piece_pos:
                x, y = pos
                # Check if the space below is out of the grid or occupied by another piece
                if y + 1 >= row or (x, y + 1) in locked_positions:
                    for pos in piece_pos:
                        p = (pos[0], pos[1])
                        locked_positions[p] = current_piece.color  # add the key and value in the dictionary
                    last_grid = copy.deepcopy(grid)  # Store the last grid position
                    current_piece = next_piece
                    next_piece = get_shape()
                    change_piece = False
                    num_lines = clear_rows(grid, locked_positions)
                    last_score = score
                    if num_lines == 1:
                        score += 100 * level
                    elif num_lines == 2:
                        score += 300 * level
                    elif num_lines == 3:
                        score += 500 * level
                    elif num_lines == 4:
                        score += 800 * level
                    break  # Stop the loop if a block cannot be placed

        draw_window(window, grid, current_piece, score, best_score, level)
        draw_next_shape(next_piece, window)
        pygame.display.update()
        pygame.event.pump()
        if check_lost(locked_positions):
            run = False
            update_qnet(last_grid, grid, current_piece.getShape(), next_piece.getShape(), 0, -100, True)
    draw_text_middle('You Lost', 40, (255, 255, 255), window)
    pygame.display.update()
    return totalReward

# Initialize your PyTorch model
qnet = QNet()
optimizer = optim.Adam(qnet.parameters())
criterion = nn.MSELoss()
model_path = "model_weights.pth"

def epsilon_greedy(grid, piece, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)  # Take a random action
    else:
        with torch.no_grad():
            shape = piece.getShape()
            state = create_state(grid,shape)  # Convert to tensor and flatten
            action_values = qnet(state)  # Get predicted action values from the Q-network
            return action_values.argmax().item()  # Return the action with the highest value
        
def create_state(grid, piece):
    state = torch.tensor(grid, dtype=torch.float32).view(-1)  # Convert to tensor and flatten
    piece_one_hot = [0]*7  # Assuming there are 7 different pieces
    piece_one_hot[piece] = 1  # Set the element corresponding to the piece to 1
    piece_one_hot = torch.tensor(piece_one_hot, dtype=torch.float32)
    state = torch.cat((state, piece_one_hot))  # Append the one-hot encoding to the state
    return state

def update_qnet(grid, next_grid, current_piece, next_piece , action, reward, done):
    state = create_state(grid, current_piece)
    next_state = create_state(next_grid, next_piece)
    
    target = reward + 0.99 * torch.max(qnet(next_state)) * (not done)  # Calculate the target Q-value
    prediction = qnet(state)[action]  # Calculate the predicted Q-value
    loss = criterion(prediction, target)  # Calculate the loss
    optimizer.zero_grad()  # Reset the gradients
    loss.backward()  # Calculate the gradients
    optimizer.step()  # Update the weights

def worker(qnet, shared_weights, experiences_queue, epsilon):
    # Initialize Pygame and create the window
    pygame.init()
    window = pygame.display.set_mode((s_width, s_height))
    print(f'Worker process {os.getpid()} started')

    # Load the shared weights into the local Q-network
    qnet.load_state_dict(shared_weights)

    for episode in range(MAX_EPISODES):
        # Run the game logic and collect experiences
        experiences = game_logic(window, qnet.parameters())
        # Send the experiences to the main process
        experiences_queue.put(experiences)

        # Load the updated weights into the local Q-network
        qnet.load_state_dict(shared_weights)

        # Decrease epsilon
        epsilon -= 0.0025
        if epsilon < 0.00:
            epsilon = 0.00
        print ("Episode: ", episode, " Epsilon: {:.3f}".format(epsilon), "Process: ", os.getpid())
    print(f'Worker process {os.getpid()} finished')
    return

MAX_EPISODES = 1000
NUM_WORKERS = 5
epsilon = 0.2
def main(window=None):
    global epsilon
    # Load the trained model weights
    try: 
        qnet.load_state_dict(torch.load("model_weights.pth"))
    except FileNotFoundError:
        print("Model weights not found. Training a new model...")
        for param in qnet.parameters():
            torch.nn.init.normal_(param)
    qnet.eval()

    # Create a queue for collecting experiences
    experiences_queue = mp.Queue()

    # Create a copy of the Q-network weights in shared memory
    shared_weights = qnet.state_dict()
    for tensor in shared_weights.values():
        tensor.share_memory_()

    # Create and start the worker processes
    workers = []
    for _ in range(NUM_WORKERS):
        worker_process = mp.Process(target=worker, args=(qnet, shared_weights, experiences_queue, epsilon))
        worker_process.start()
        workers.append(worker_process)

    for worker_process in workers:
        worker_process.join()

    torch.save(qnet.state_dict(), "model_weights.pth")
    print("Training complete")

min_reward = -1
max_reward = 1

def find_best_move(grid, piece, next_piece):
    # Convert the grid into a suitable format for the ANN
    grid_for_ann = [cell for row in grid for cell in row]
    grid_for_ann = torch.tensor(grid_for_ann, dtype=torch.float32).view(-1)

    copy_piece = copy.deepcopy(piece)
    next_grid = copy.deepcopy(grid)
    
    action = epsilon_greedy(grid, piece, epsilon)
    if action == 0:
        copy_piece.x = piece.x - 1
        if not valid_space(copy_piece, grid):
            return None, None, 3
    elif action == 1:
        copy_piece.x = piece.x + 1
        if not valid_space(copy_piece, grid):
            return None, None, 3
    elif action == 2:
        copy_piece.rotation = piece.rotation + 1
        if not valid_space(copy_piece, grid):
            return None, None, 3
    
    # Drop the piece to the lowest valid position
    while valid_space(copy_piece, next_grid):
        copy_piece.y += 1
    copy_piece.y -= 1  # Adjust for the last increment

    height = get_aggregate_height(grid)
    complete_lines = get_complete_lines(grid)
    holes = get_holes(grid)
    bumpiness = get_bumpiness(grid)
    row_comp = row_completeness(grid)
    column_heights = get_column_heights(grid)
    height_dev = np.std(column_heights)

    # Update the Q-network
    reward = -0.4 * height + 0.9 * complete_lines + 0.7 * row_comp - 0.2 * holes - 0.18 * bumpiness - 0.5 * height_dev    # The reward is the score difference
    # Update the minimum and maximum rewards
    global min_reward, max_reward
    min_reward = min(min_reward, reward)
    max_reward = max(max_reward, reward)

    # Normalize the reward
    reward = 2 * ((reward - min_reward) / (max_reward - min_reward)) - 1
    done = False  # The game is over if run is False
    shapeInt = piece.getShape()
    nextShapeInt = next_piece.getShape()
    #print("Reward: ", reward)
    update_qnet(grid, next_grid, shapeInt, nextShapeInt, action, reward, done)  # Pass the last grid and piece as the state

    return copy_piece.x, copy_piece.rotation, reward

def get_aggregate_height(grid):
    aggregate_height = 0
    for j in range(len(grid[0])):
        for i in range(len(grid)):
            if grid[i][j] != (0, 0, 0):
                aggregate_height += len(grid) - i
                break
    return aggregate_height

def get_complete_lines(grid):
    complete_lines = 0
    for i in range(len(grid)):
        if (0, 0, 0) not in grid[i]:
            complete_lines += 1
    return complete_lines

def row_completeness(grid):
    total_score = 0
    for row in grid:
        filled_cells = sum(cell != (0, 0, 0) for cell in row)  # Count the number of filled cells in the row
        score = filled_cells / len(row)  # Calculate the score for the row
        total_score += score
    return total_score  # Convert the total score to an integer

def get_holes(grid):
    holes = 0
    for i in range(len(grid) - 1):  # -1 to ignore the bottom row
        for j in range(len(grid[i])):
            if grid[i][j] == (0, 0, 0) and grid[i + 1][j] != (0, 0, 0):
                holes += 1
    return holes

def get_bumpiness(grid):
    bumpiness = 0
    column_heights = [0] * len(grid[0])
    for j in range(len(grid[0])):
        for i in range(len(grid)):
            if grid[i][j] != (0, 0, 0):
                column_heights[j] = len(grid) - i
                break
    for j in range(len(column_heights) - 1):  # -1 to ignore the last column
        bumpiness += abs(column_heights[j] - column_heights[j + 1])
    return bumpiness

def get_column_heights(grid):
    column_heights = []
    for j in range(len(grid[0])):
        for i in range(len(grid)):
            if grid[i][j] != (0, 0, 0):
                column_heights.append(len(grid) - i)
                break
        else:
            column_heights.append(0)
    return column_heights

def lock_positions(grid, piece):
    formatted = convert_shape_format(piece)

    for pos in formatted:
        p = (pos[0], pos[1])
        if p[1] > -1:
            grid[p[1]][p[0]] = piece.color

    return grid

def select(fitnesses):
    # Normalize the fitness values so they sum to 1
    total_fitness = sum(fitnesses)
    normalized_fitnesses = [fitness / total_fitness for fitness in fitnesses]

    # Choose a random number between 0 and 1
    r = random.random()

    # Select an individual such that the probability of selecting it is proportional to its fitness
    cumulative_probability = 0.0
    for i, fitness in enumerate(normalized_fitnesses):
        cumulative_probability += fitness
        if r < cumulative_probability:
            return i

    # If no individual is selected (which should be very unlikely), return the last one
    return len(fitnesses) - 1

def main_menu(window):
    run = True
    while run:
        draw_text_middle('Press any key to begin', 50, (255, 255, 255), window)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                main(window)

    pygame.quit()


if __name__ == '__main__':
    win = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption('Tetris')
    draw_text_middle('Running workers', 50, (255, 255, 255), win)
    pygame.display.update()
    main(win)  # start game

