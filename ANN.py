from csv import writer
from math import exp
import os
import random
import stat
from tracemalloc import start
from typing import final
import pygame
import copy
import random
import threading
from Tetris import find_optimal_move

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter

import numpy as np

if (torch.cuda.is_available()):
    print("CUDA is available, using GPU")
    device = torch.device("cuda")
else:
    print("CUDA is not available, using CPU")
    device = torch.device("cpu")


class QNet(nn.Module):
    def __init__(self, max_x, max_rotation):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(607, 128)  # 10x20 grid as input
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, max_x * max_rotation)  # Output layer size is max_x * max_rotation

        self.max_x = max_x
        self.max_rotation = max_rotation

    def forward(self, x):
        x = x.view(-1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(self.max_x, self.max_rotation)  # Reshape the output into a 2D tensor
        return x
    

class CustomNet(nn.Module):
    def __init__(self, max_x, max_rotation):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        # Calculate the total number of features after the convolutional layers
        total_features = 25600  # Adjust this value based on your specific case

        self.fc1 = nn.Linear(total_features, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, max_x * max_rotation)  # Change this to 1 if the final output is 1-dimensional
        self.dropout = nn.Dropout(0.25)
        self.batch_norm = nn.BatchNorm2d(128)
        self.max_x = max_x
        self.max_rotation = max_rotation
        self.final_layer = self.fc3

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), 64, -1, 1)
        x = F.relu(self.conv4(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv5(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv6(x))
        x = self.batch_norm(x)

        # Flatten the tensor before passing it to the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # Reshape the output tensor to have one dimension for xs and another for rotations
        x = x.view(-1, self.max_x, self.max_rotation)

        # Reshape the tensor to be 1D and contain pairs [x, rotation]
        x = x.view(-1, 2)

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
                #print("Invalid space found: ", pos)
                return False
    return True

def find_valid_rotation(piece, grid):
    copy_piece = copy.deepcopy(piece)

    # Try each rotation
    for rotation in range(4):
        copy_piece.rotation = rotation

        # Check if the new rotation is valid
        if valid_space(copy_piece, grid):
            # If it's valid, return the rotation
            print("Valid rotation found: ", rotation)
            return rotation

    return None

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
        new_locked = {}
        for key in sorted(list(locked), key=lambda a: a[1])[::-1]:
            x, y = key
            if y < index:                       # if the y value is above the removed index
                new_key = (x, y + increment)    # shift position to down
                new_locked[new_key] = locked[key]
            else:
                new_locked[key] = locked[key]
        locked = new_locked

    return increment, locked


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
def draw_window(surface, grid, current_piece, score=0, last_score=0, level=0, bestX=0, bestRotation=0, episode=0, epsilon=0.0):
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

    # best move
    label = font.render('BEST MOVE   '+ str(bestX)+ " " + str(bestRotation), 1, (255,255,255))
    start_x_best = top_left_x + play_width + 50
    start_y_best = top_left_y + (play_height/2 - 100)
    surface.blit(label, (start_x_best - 10, start_y_best + 250))

    # episode number
    label = font.render('EPISODE   ' + str(episode), 1, (255,255,255))
    start_x_best = top_left_x + play_width + 50
    start_y_best = top_left_y + (play_height/2 - 100)
    surface.blit(label, (start_x_best - 10, start_y_best + 300))

    # epsilon number
    label = font.render('EPSILON   ' + str(epsilon), 1, (255,255,255))
    start_x_best = top_left_x + play_width + 50
    start_y_best = top_left_y + (play_height/2 - 100)
    surface.blit(label, (start_x_best - 10, start_y_best + 350))

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



gameScore = 0
def getFinalScore():
    global gameScore
    return gameScore

def setFinalScore(score):
    global gameScore
    gameScore = score

def emulate_placement(grid, piece):
    piece_copy = copy.deepcopy(piece)
    grid_copy = copy.deepcopy(grid)
    # Drop the piece to the lowest valid position
    while valid_space(piece_copy, grid_copy):
        piece_copy.y += 1
    piece_copy.y -= 1  # Adjust for the last increment
    if piece_copy.y < 0:
        piece.copy.y = 0
    if piece_copy.y == 0:
        return grid_copy
    lock_positions(grid_copy, piece_copy)
    return grid_copy

def game_logic(window, parameters, episode, id, randomize, agent):
    actions = 0
    actions_taken[id] = []
    experiences = []
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
    bestX = 0
    bestRotation = 0
    step = 0
    move = True
    global epsilon
    
    totalReward = 0

    if randomize:
        print("Randomizing Agent")
    if agent:
        print("Optimal Agent")
    if not randomize and not agent:
        print("ANN Agent")
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
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
            move = True

            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        if move:
            if ai_enabled:
                
                    best_move = [None, None]
                    
                    if current_piece.y >= 3 and current_piece.shape != I:
                        best_move[0], best_move[1], reward, action, bestX, bestRotation = find_best_move(grid, current_piece, next_piece, actions, randomize, agent)
                    elif current_piece.shape == I and current_piece.y >= 4:
                        best_move[0], best_move[1], reward, action, bestX, bestRotation = find_best_move(grid, current_piece, next_piece, actions, randomize, agent)
                    
                    if best_move[0] is not None and best_move[1] is not None:
                        current_piece.x = best_move[0]
                        current_piece.rotation = best_move[1]
                    
                    move = False
            
            next_grid = emulate_placement(grid, current_piece)
            shapeInt = current_piece.getShape()
            nextShapeInt = next_piece.getShape()
            done = False
            update_qnet(grid, next_grid, shapeInt, nextShapeInt, bestX, bestRotation, reward + 0.01, step, done)

            actions += 1
            
            totalReward += reward
        
        
        piece_pos = convert_shape_format(current_piece)


        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y > 0:
                grid[y][x] = current_piece.color

        #print("Reward: ", reward)
        if change_piece:  # if the piece is locked
            piece_pos = convert_shape_format(current_piece)
            for pos in piece_pos:
                x, y = pos
                # Check if the space below is out of the grid or occupied by another piece
                if y + 1 >= row or (x, y + 1) in locked_positions:
                    for pos in piece_pos:
                        p = (pos[0], pos[1])
                        locked_positions[p] = current_piece.color  # add the key and value in the dictionary
                    #print("REWARD FOR CHOSEN MOVE: ", reward)
                    last_grid = copy.deepcopy(grid)  # Store the last grid position
                    current_piece = next_piece
                    next_piece = get_shape()
                    change_piece = False
                    num_lines, locked_positions = clear_rows(grid, locked_positions)
                    last_score = score
                    shapeInt = current_piece.getShape()
                    nextShapeInt = next_piece.getShape()
                    done = False
                    step +=1
                    
                    if num_lines >= 1:
                        if actions > 2000:
                            print("Lines cleared: ", num_lines, " Episode: ", episode, "Cleared by Agent:", id, "In mode: ANN", "Actions taken: ", actions)
                        else:
                            print("Lines cleared: ", num_lines, " Episode: ", episode, "Cleared by Agent:", id, "In mode: OPT", "Actions taken: ", actions)
                    if num_lines == 1:
                        score += 100 * level
                    elif num_lines == 2:
                        score += 300 * level 
                    elif num_lines == 3:
                        score += 500 * level
                    elif num_lines == 4:
                        score += 800 * level
                    break  # Stop the loop if a block cannot be placed
        
        setFinalScore(score)
        finalScore = getFinalScore()
        #print("Final score: ", finalScore)
        draw_window(window, grid, current_piece, score, best_score, level, bestX, bestRotation, episode, epsilon)
        draw_next_shape(next_piece, window)
        pygame.display.update()
        pygame.event.pump()
        if check_lost(locked_positions):
            run = False
            done = True
            update_qnet(grid, next_grid, shapeInt, nextShapeInt, bestX, bestRotation, -1, step, done)
    draw_text_middle('You Lost', 40, (255, 255, 255), window)
    pygame.display.update()
    return []

# Initialize your PyTorch model
#qnet = QNet(10,4)
qnet = CustomNet(10,4)
qnet.to(device)
optimizer = optim.Adam(qnet.parameters(), lr = 0.001)
criterion = nn.MSELoss()
model_path = "model_weights.pth"

def epsilon_greedy(grid, piece, epsilon, moves, randomize, agent):
    if randomize and moves < 2000:
        actions = []
        actions.append(np.random.randint(0, 3))  # Move left or right
        return actions, False, 0, 0
    if randomize and moves >= 2000:
        actions = []
        actions.append(3)  # Move left or right
        return actions, False, 0, 0
    if agent and moves < 2000:
        best_x, best_rotation =  find_optimal_move(grid, piece)  # Take the best action
        return generate_actions(piece.x, piece.rotation, best_x, best_rotation), False, best_x, best_rotation
    if agent and moves >= 2000:
        actions = []
        actions.append(3)
    if random.random() < epsilon and moves < 2000:
        actions = []
        actions.append(np.random.randint(0, 3))  # Move left or right
        if actions[0] == 0:
            best_x = piece.x - 1
            best_rotation = piece.rotation
        elif actions[0] == 1:
            best_x = piece.x + 1
            best_rotation = piece.rotation
        elif actions[0] == 2:
            best_x = piece.x
            if (piece.rotation == 3):
                best_rotation = 0
            else:
                best_rotation = piece.rotation + 1
        else:
            best_x = piece.x
            best_rotation = piece.rotation
        best_x = max(0, best_x)
        best_x = min(9, best_x)
        best_rotation = max(0, best_rotation)
        best_rotation = min(3, best_rotation)
        return actions, False, best_x, best_rotation
        best_x, best_rotation =  find_optimal_move(grid, piece)  # Take the best action
        return generate_actions(piece.x, piece.rotation, best_x, best_rotation), False, best_x, best_rotation
    else:
        with torch.no_grad():
            copy_piece = copy.deepcopy(piece)
            # Convert to tensor and flatten
            state = create_state(grid).to(device)
            x = qnet(state)
            #print("X: ", x)
            # Find the index of the maximum score
            index = x.argmax()
            #print("Index: ", index)
            # Calculate the total number of rotations
            total_rotations = 4  # replace with the actual total number of rotations if it's not 4

            # Convert the index to best_x and best_rotation
            best_x = index // total_rotations
            best_rotation = index % total_rotations
            print("Best x: ", best_x.item(), " Best rotation: ", best_rotation.item())
            return generate_actions(copy_piece.x, copy_piece.rotation, best_x.item(), best_rotation.item()), True, best_x.item(), best_rotation.item()
        
def generate_actions(current_x, current_rotation, best_x, best_rotation):
    actions = []

    # Move the piece to the best x position
    while current_x < best_x:
        actions.append(1)  # Move right
        current_x += 1
    while current_x > best_x:
        actions.append(0)  # Move left
        current_x -= 1

    # Rotate the piece to the best rotation
    for _ in range((best_rotation - current_rotation) % 4):
        actions.append(2)  # Rotate

    # Drop the piece
    actions.append(3)  # Do nothing (drop)

    return actions
        
def create_state(grid):
    grid_height = 20
    grid_width = 10
    default_value = 0  # Change this to the grayscale value that represents an empty cell
    # Check if grid is already in the correct format
    if isinstance(grid[0], list):
        grid_for_ann = grid
    else:
        # Convert the grid into a suitable format for the ANN
        grid_for_ann = [grid[i*10:(i+1)*10] for i in range(20)]
    # Fill any empty row in the grid with the default value
    for i, row in enumerate(grid_for_ann):
        if len(row) == 0:
            grid_for_ann[i] = [default_value] * grid_width
            print("Empty row found,", i)

    # Convert the grid into grayscale
    grid_for_ann = [[0.299*cell[0] + 0.587*cell[1] + 0.114*cell[2] if isinstance(cell, tuple) else cell for cell in row] for row in grid_for_ann]

    state = torch.tensor(grid_for_ann, dtype=torch.float32).view(1, 1, grid_height, grid_width)  # Reshape the grid into a single-channel image
    return state

def update_qnet(grid, next_grid, current_piece, next_piece , target_position, target_rotation, reward, step, done):
    
    state = create_state(grid).to(device)
    next_state = create_state(next_grid).to(device)

    # Check if qnet and state are not None
    if qnet is None:
        print("qnet is None")
        return
    if state is None:
        print("state is None")
        return

    # Get the predicted Q-values for the current state
    predicted_q_values = qnet(state)

    # Get the predicted Q-values for the next state
    next_q_values = qnet(next_state)

    # Compute the target Q-values
    target_q_values = reward + (1 * next_q_values * (1 - done))

    # Compute the loss between the predicted and target Q-values
    loss = F.mse_loss(predicted_q_values, target_q_values)

    # Backpropagate the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

def worker(qnet, shared_weights, experiences_queue, epsilon, id, random, ag):
    # Initialize Pygame and create the window
    pygame.init()
    window = pygame.display.set_mode((s_width, s_height))
    print(f'Worker process {os.getpid()} started')
    # Load the shared weights into the local Q-network
    qnet.load_state_dict(shared_weights)


    writer = SummaryWriter(f'runs/worker_{id}')
    for episode in range(MAX_EPISODES):
        experiences = []
        # Run the game logic and collect experiences
        experiences = game_logic(window, qnet.parameters(), episode, id, random, ag)
        # Send the experiences to the main process
        for experience in experiences:
            grid = experience[0]
            next_grid = experience[1]
            current_piece = experience[2]
            next_piece = experience[3]
            bestX = experience[4]
            bestRotation = experience[5]
            reward = experience[6]
            step = experience[7]
            done = experience[8]
            

        experiences.clear()

        #experiences_queue.put(experiences)
        #print(f"Size of experiences_queue after episode {episode}: {experiences_queue.qsize()}")
        # Load the updated weights into the local Q-network
        qnet.load_state_dict(shared_weights)

        # Decrease epsilon
        epsilon -= 0.005
        if epsilon < 0.00:
            epsilon = 0.00
        print ("Episode: ", episode, " Epsilon: {:.3f}".format(epsilon), "Process: ", os.getpid())

    print(f'Worker process {os.getpid()} finished')
    writer.close()
    pygame.quit()
    os._exit(0)
    

MAX_EPISODES = 1500
NUM_WORKERS = 3
epsilon = 1
actions_taken = [[] for _ in range(NUM_WORKERS)]
def main(random, ag):
    global epsilon
    
    # Load the trained model weights
    try: 
        qnet.load_state_dict(torch.load("model_weights.pth"))
    except FileNotFoundError:
        print("Model weights not found. Training a new model...")
        for param in qnet.parameters():
            torch.nn.init.normal_(param)
    qnet.eval()

    print("Random: ", random, "Agent: ", ag)
    
    # Create a queue for collecting experiences
    experiences_queue = mp.Queue()

    # Create a copy of the Q-network weights in shared memory
    shared_weights = qnet.state_dict()
    for tensor in shared_weights.values():
        tensor.share_memory_()

    weights_before = copy.deepcopy(shared_weights)
    # Create and start the worker processes
    workers = []
    for id in range(NUM_WORKERS):
        worker_process = mp.Process(target=worker, args=(qnet, shared_weights, experiences_queue, epsilon, id, random, ag))
        worker_process.start()
        workers.append(worker_process)

    
    for worker_process in workers:
        worker_process.join()

    weights_after = qnet.state_dict()
    for name, param_before in weights_before.items():
        param_after = weights_after[name]
        if not torch.allclose(param_before, param_after):
            print(f'Parameter {name} has changed')
    torch.save(qnet.state_dict(), "model_weights.pth")
    print("Training complete")
    finalScore = getFinalScore()
    print("Final score: ", finalScore)
    return finalScore

def runAgents(random, ag):
    pygame.init()
    window = pygame.display.set_mode((s_width, s_height))
    print(f'Worker process {os.getpid()} started')
    try: 
        qnet.load_state_dict(torch.load("model_weights.pth"))
    except FileNotFoundError:
        print("Model weights not found. Training a new model...")
        for param in qnet.parameters():
            torch.nn.init.normal_(param)
    # Load the shared weights into the local Q-network
    qnet.load_state_dict(qnet.state_dict())
    finalScore = game_logic(window, qnet.parameters(), 0, 0, random, ag)
    print("Final score: ", finalScore)
    finalScore = getFinalScore()
    print("Final score: ", finalScore)
    return finalScore

def find_best_move(grid, piece, next_piece, moves, randomize ,agent):
    copy_piece = copy.deepcopy(piece)
    next_grid = copy.deepcopy(grid)
    actions, decider, bestX, bestRotation = epsilon_greedy(grid, piece, epsilon, moves, randomize, agent)
    reward = 0

    for action in actions:
        if action == 0:
            copy_piece.x -= 1
            if not valid_space(copy_piece, grid):
                copy_piece.x += 1
        elif action == 1:
            copy_piece.x += 1
            if not valid_space(copy_piece, grid):
                copy_piece.x -= 1
        elif action == 2:
            copy_piece.rotation = (copy_piece.rotation + 1) % len(copy_piece.shape)
            if not valid_space(copy_piece, grid):
                #print("Invalid rotation: ", copy_piece.rotation)
                copy_piece.rotation = (copy_piece.rotation - 1) % len(copy_piece.shape)
            
                
    reward = calculate_reward(next_grid, copy_piece)
        
    return copy_piece.x, copy_piece.rotation, reward, actions, int(bestX), int(bestRotation)

# Initialize the min and max rewards to None
min_reward = 1
max_reward = 0

def calculate_reward(next_grid, copy_piece):
    global min_reward, max_reward
    # Drop the piece to the lowest valid position
    while valid_space(copy_piece, next_grid):
        copy_piece.y += 1
    copy_piece.y -= 1  # Adjust for the last increment

    lock_positions(next_grid, copy_piece)

    height = get_aggregate_height(next_grid)
    complete_lines = get_complete_lines(next_grid)
    holes = get_holes(next_grid)
    bumpiness = get_bumpiness(next_grid)
    column_heights = get_column_heights(next_grid)
    height_dev = np.std(column_heights)
    row_scores = grid_completeness_score(next_grid)
    #print("Height: ", height, " Complete lines: ", complete_lines, " Holes: ", holes, " Bumpiness: ", bumpiness, " Height deviation: ", height_dev, " Row scores: ", row_scores)
    # Update the Q-network  
    # Calculate the raw reward
    reward = -0.51 * height + 0.76 * complete_lines + -0.36 * holes + -0.18 * bumpiness
    return reward
    
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
    
    return complete_lines * complete_lines

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

def row_completeness_score(row):
    score = 0
    for cell in row:
        if cell != (0, 0, 0):  # If the cell is filled
            score = score * 2 + 1  # Double the score and add 1
    return score

def grid_completeness_score(grid):
    total_score = 0
    for row in grid:
        total_score += row_completeness_score(row)
    return total_score

def lock_positions(grid, piece):
    formatted = convert_shape_format(piece)

    for pos in formatted:
        p = (pos[0], pos[1])
        if p[1] > -1:
            grid[p[1]][p[0]] = piece.color

    return grid

def main_menu(window):
    run = True
    while run:
        draw_text_middle('Press any key to begin', 50, (255, 255, 255), window)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                main(False, False)

    pygame.quit()


if __name__ == '__main__':
    win = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption('Tetris')
    draw_text_middle('Running workers', 50, (255, 255, 255), win)
    pygame.display.update()
    main(False, False)  # start game

