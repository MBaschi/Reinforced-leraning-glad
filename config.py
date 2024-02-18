"In this file are defined the main configurations of the game"
import random

random.seed(12321)

#SIMULATION VARIABLES
RADIAL_RESOLUTION = 10 #quantization of the direction in degrees
GLADIATOR_NAMES = ["A", "B"] #this list decide also the number of gladiators in the game
TIMER= 1000 #max number of turns

#ARENA VARIABLE
ARENA_SIZE = 100

#REWARDS AND PENALTIES
WIN_REWARD = 100
SUCCESFULL_ATTACK_REWARD = 2
BLOCKED_ATTACK_REWARD = 2
KILL_REWARD = 10

HITTED_PENALTY = -2
MISSED_ATTACK_PENALTY = -0.25
TIMEOUT_PENALTY = -0.1
DEATH_PENALTY = -10

#VISUALIZATION VARIABLES
SCREEN_HEIGHT = 1000
SCREEN_WIDTH = 1000
ARENA_VISUALIZATION_SIZE = int(SCREEN_HEIGHT*0.8)
GLADIATOR_SIZE = int(ARENA_VISUALIZATION_SIZE*0.01)
STATS_BARS_SIZE = int(GLADIATOR_SIZE/3)
LINE_SIZE = int(GLADIATOR_SIZE/3)

#FIXED VARIABLES: can't be changed, inserted just for readability
NUM_POSSIBLE_ACTIONS = 5 #len(Gladiator.possible_actions)