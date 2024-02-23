"""In this file are defined the object of the game with theri logic:
- Model: the q network of the gladiator
- Gladiator: the gladiator object 
- Enviroment: the enviroment of the game 
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from config import GLADIATOR_HEALTH,GLADIATOR_STAMINA,GLADIATOR_ATTACK_DAMAGE,GLADIATOR_ATTACK_RANGE,GLADIATOR_SPEED
from config import RADIAL_RESOLUTION, GLADIATOR_NAMES, NUM_POSSIBLE_ACTIONS, ARENA_SIZE, SUCCESFULL_ATTACK_REWARD, BLOCKED_ATTACK_REWARD, KILL_REWARD, HITTED_PENALTY, TIMEOUT_PENALTY, DEATH_PENALTY
from config import ARENA_VISUALIZATION_SIZE, GLADIATOR_SIZE, STATS_BARS_SIZE, LINE_SIZE
torch.autograd.set_detect_anomaly(True)

class Model():
    def __init__(self,gladiator_name):
        self.gladiator_name = gladiator_name
        self.output_size = (NUM_POSSIBLE_ACTIONS-1)*int(360/RADIAL_RESOLUTION)+1 #the -1 is because the action "rest" does not have a direction
        self.input_size = len(GLADIATOR_NAMES)*4+1 #5 is the number of features for each gladiator (position x, position y, health, stamina) and 1 is how much time is left
        self.q_network = nn.Sequential(
            nn.Linear(self.input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            #the last layer has as many neurons as the number of possible actions with also the direction
            nn.Linear(10, self.output_size) #the -1 is because the action "rest" does not have a direction
        )
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.discount_factor = 0.99

    def next_action(self,
                state: torch.Tensor
                )-> torch.Tensor:
        'Return the index of the action with the highest q value for the given state and the direction of the action'

        #choose the action with the highest q value
        self.q_values = self.q_network(state)
        self.output_neur_max = torch.argmax(self.q_values).item()

        if self.output_neur_max == self.output_size:
            return NUM_POSSIBLE_ACTIONS , 0 #action "rest" does not have a direction

        #transform the action index in the action and the direction
        action_index, direction_index = divmod(self.output_neur_max, int(360/RADIAL_RESOLUTION))
        #action in degrees
        direction = direction_index*RADIAL_RESOLUTION
        return action_index, direction

    def calculate_target_q_values(self,
                                  reward:torch.Tensor,
                                  next_state:torch.Tensor
                                  )-> torch.Tensor:
        with torch.no_grad():
            next_state_q_values = self.q_network(next_state)
            max_new_state_q_value = next_state_q_values.max().item()
            target_q_value = reward + self.discount_factor * max_new_state_q_value
            target_q_value = torch.tensor(target_q_value, dtype=torch.float32)
        return target_q_value

    def upgrade(self,
                reward:torch.Tensor,
                next_state:torch.Tensor
                )-> None:
        predicted_q_value = torch.max(self.q_values)
        target_q_value = self.calculate_target_q_values(reward, next_state)
        loss = self.loss_function(predicted_q_value, target_q_value)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

class Gladiator():

    possible_actions = ["attack", "block", "dash", "rest", "walk"]

    def __init__(self, name, gladiator_id, spawn_point ):
        self.name = name
 
        self.position={'x': spawn_point['x'], 'y': spawn_point['y']}
        self.health = GLADIATOR_HEALTH
        self.stamina = GLADIATOR_STAMINA

        self.damage = GLADIATOR_ATTACK_DAMAGE
        self.state = "stay" # stay, walk, attack, block, dodge, rest
        self.range = GLADIATOR_ATTACK_RANGE
        self.speed = GLADIATOR_SPEED
        self.field_view_angle = np.pi/3 # in grad

        self.gladiator_id = gladiator_id
        self.brain = Model(gladiator_name=self.name)

        self.target=None
        self.direction=0
        self.reward=0

    def convert_state_to_tensor(self, state):
        '''Transform the dictionary state to a torch tensor in wich each row is a gladitor state
           the first row is the state of the gladiator itself'''
        my_state = state[self.gladiator_id]
        state.pop(self.gladiator_id)
        state_tensor = [my_state] + [state[keys_left] for keys_left in state]
        state_tensor = torch.cat(state_tensor)
        return state_tensor

    def choose_action(self, arena_state):
        arena_state=self.convert_state_to_tensor(arena_state)
        action_index, self.direction=self.brain.next_action(arena_state)
        self.state = self.possible_actions[action_index]
        if self.stamina >100: self.stamina = 100
        if self.stamina <0:
            self.stamina = 0
            self.state = "rest"

    def perform_action(self, gladiators):
        if self.state == "attack":
            self.attack(self.direction, gladiators)
        elif self.state == "block":
            self.block()
        elif self.state == "dash":
            self.dash(self.direction)
        elif self.state == "rest":
            self.rest()
        elif self.state == "walk":
            self.walk(self.direction)

    def attack(self, direction, gladiators):
        '''
        Permorf an attack, reward and block are handeled here:
        - if the gladiator have stamina enter in the attack condition
        - create a list with all the target in the hit zone
        - for each target in the hit zone check if the attack is blocked or is succesfull and also if the target is dead
        '''
        if self.stamina > 0:
            self.stamina -= 10

            gladiator_in_hitzone = []
            for gladiator in gladiators:
                if gladiator != self and self.is_target_in_hitzone(gladiator, direction):
                    gladiator_in_hitzone.append(gladiator)

            for target in gladiator_in_hitzone:     
                self.reward += SUCCESFULL_ATTACK_REWARD #attack hitted 
                if target.state == "block" and target.is_target_in_hitzone(self, target.direction): #target blocked succesfully
                    target.stamina -= 5
                    target.reward += BLOCKED_ATTACK_REWARD #attack blocke
                else:
                    target.health -= self.damage
                    self.reward += SUCCESFULL_ATTACK_REWARD #attack succcessful
                    target.reward += HITTED_PENALTY #target penalized
                    if target.health <= 0:
                        #target is dead
                        self.reward += KILL_REWARD
                        target.reward += DEATH_PENALTY
                        #put dead body outside the arena
                        target.health = 0
                        target.position = {'x': -1, 'y': -1}

    def block(self):
        'Since the block lohic is simple is directly handeled in the attack function'

    def dash(self, direction):
        'Move the gladiator in the direction of the vector direction with a speed 3 times higher than the normal speed'
        if self.stamina >= 10:
            self.stamina -= 10

            direction_vector=convert_direction_to_vector(direction)
            new_position = {'x': self.position['x'] + 3*self.speed*direction_vector['x'], 'y': self.position['y'] + 3*self.speed*direction_vector['y']}

            #check if the new position is inside the arena
            for i in ['x', 'y']:
                if new_position[i] > 0 and new_position[i] < ARENA_SIZE:
                    self.position[i] = new_position[i]
                elif new_position[i] < 0:
                    self.position[i] = 0
                elif new_position[i] > ARENA_SIZE:
                    self.position[i] = ARENA_SIZE

    def rest(self):
        'Recover stamina when resting'
        self.stamina = min(100, self.stamina + 20)

    def walk(self, direction):
        'Move the gladiator in the direction of the vector direction'
        direction_vector=convert_direction_to_vector(direction)
        new_position = {'x': self.position['x'] + self.speed*direction_vector['x'], 'y': self.position['y'] + self.speed*direction_vector['y']}

        for i in ['x', 'y']:
            if new_position[i] > 0 and new_position[i] < ARENA_SIZE:
                self.position[i] = new_position[i]
            elif new_position[i] < 0:
                self.position[i] = 0
            elif new_position[i] > ARENA_SIZE:
                self.position[i] = ARENA_SIZE

    def is_target_in_hitzone(self, target, direction):
        '''Check if the target is in the field of view and in the range of the gladiator'''
        #transform the direction in a vector
        direction = convert_direction_to_vector(direction)

        # Calculate the direction to the target
        target_direction = {'x': target.position['x'] - self.position['x'], 'y': target.position['y'] - self.position['y']}

        # Normalize the directions to get unit vectors
        direction_magnitude = np.sqrt(direction['x']**2 + direction['y']**2)
        target_direction_magnitude = np.sqrt(target_direction['x']**2 + target_direction['y']**2)

        direction = {'x': direction['x'] / direction_magnitude, 'y': direction['y'] / direction_magnitude}
        target_direction = {'x': target_direction['x'] / target_direction_magnitude, 'y': target_direction['y'] / target_direction_magnitude}

        # Calculate the dot product of the two vectors
        dot_product = direction['x']*target_direction['x'] + direction['y']*target_direction['y']

        # Calculate the angle between the two vectors
        angle = np.arccos(dot_product)

        # Check if the target is within the field of view
        is_in_field_of_view = angle <= self.field_view_angle / 2

        # Check if the target is within range
        is_in_range = target_direction_magnitude <= self.range

        return is_in_field_of_view and is_in_range

    def respawn(self,position):
        self.position = position
        self.health = 100
        self.stamina = 100
        self.state = "stay"
        self.direction = 0
        self.reward = 0

def convert_direction_to_vector( direction):
    radian_direction = np.radians(direction)
    return {'x': np.cos(radian_direction), 'y': np.sin(radian_direction)}

class Enviroment():

    def __init__(self) :
        self.gladiators = []
        self.state = "start"

    def compile_arena_state(self,time):
        '''
        Rerurn a dictionary with keys the gladiator id and values a torche tensor with the following information:
        - position of each gladiator
        - health of each gladiator
        - stamina of each gladiator
        - orientation of each gladiator
        '''
        arena_state = {}
        for gladiator in self.gladiators:
            arena_state[gladiator.gladiator_id] = torch.tensor([gladiator.position['x'],
                                                                gladiator.position['y'],
                                                                gladiator.health,
                                                                gladiator.stamina], dtype=torch.float32)
        #add timer 
        arena_state['timer'] = torch.tensor([time], dtype=torch.float32)
        return arena_state

    def run_frame(self,time):
        '''
        Run the next frame of the game:
           - compile the dictionary with the state of the arena
           - each gladiator alive choose an action
           - the action is performed and effect (with the reward) are calculated
           - the reward is used to upgrade the q network of the gladiator
        '''
        arena_state = self.compile_arena_state(time)
        #check gladiator alive
        alive_gladiators = [gladiator for gladiator in self.gladiators if gladiator.health > 0]
        #each gladiator choose an action
        for gladiator in alive_gladiators:
            gladiator.choose_action(copy.deepcopy(arena_state))
        #each gladiator perform the action
        for gladiator in alive_gladiators:
            gladiator.perform_action(self.gladiators)
        #based on the action effect calculate the reward and upgrade the q network
        for gladiator in alive_gladiators:
            gladiator.reward += TIMEOUT_PENALTY #penalize the gladiator for not doing anything
            gladiator.brain.upgrade(gladiator.reward, gladiator.convert_state_to_tensor(copy.deepcopy(arena_state)))

    def add_gladiator(self, name, spawn_point):
        'Add a gladiator to the game'
        self.gladiators.append(Gladiator(name, len(self.gladiators), spawn_point))

    def draw(self, ax):
        # Draw background
        ax.set_facecolor((0, 0, 0))

        # Draw arena
        rect = plt.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE, linewidth=2, edgecolor=(1, 1, 1), facecolor='none')
        ax.add_patch(rect)

        # Draw gladiators
        for gladiator in self.gladiators:
            self.draw_gladiator(ax, gladiator.name, (gladiator.position['x'], gladiator.position['y']), gladiator.health, gladiator.stamina, gladiator.direction, gladiator.state)

        # Draw dashboard
        for i, gladiator in enumerate(self.gladiators):
            text = f"{gladiator.name} - health: {gladiator.health} - stamina: {gladiator.stamina} - action: {gladiator.state} - pos: {gladiator.position['x']},{gladiator.position['y']}"
            ax.text(0, ARENA_SIZE + 0.1*ARENA_SIZE + 0.3*ARENA_SIZE* i, text, color=(1, 1, 1), fontsize=10)
    
    def draw_gladiator(self, ax, name, position, health, stamina, direction, action):

        # Draw gladiator
        circle = plt.Circle(position, GLADIATOR_SIZE, color=(0, 0, 1))
        ax.add_artist(circle)

        # Draw health bar
        ax.add_patch(plt.Rectangle((position[0], position[1] - GLADIATOR_SIZE*1.2), 1.2*GLADIATOR_SIZE*(health / GLADIATOR_HEALTH), STATS_BARS_SIZE, color=(1, 0, 0)))
        
        # Draw stamina bar
        ax.add_patch(plt.Rectangle((position[0], position[1] - GLADIATOR_SIZE*1.2), 1.2*GLADIATOR_SIZE*(stamina / GLADIATOR_STAMINA), STATS_BARS_SIZE, color=(0, 1, 0)))

        # Draw direction
        direction_line_x = [position[0], position[0] + GLADIATOR_ATTACK_RANGE * np.cos(np.radians(direction))]
        direction_line_y = [position[1], position[1] + GLADIATOR_ATTACK_RANGE * np.sin(np.radians(direction))]
        ax.plot(direction_line_x, direction_line_y, color=(1, 1, 1))
   