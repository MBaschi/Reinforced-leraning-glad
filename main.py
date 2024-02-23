"""In this file is runned the game simulation and visualization"""
from random import randint
from basic_classes import Enviroment
from config import GLADIATOR_NAMES,TIMER,ARENA_SIZE,TRAINING_EPISODES,VIEW,VIEW_EVERY
from tqdm import tqdm
import pygame
from pygame.time import Clock
import matplotlib.pyplot as plt

pygame.init()
clock = Clock()  
FPS = 60  

#set up the drawing window
screen = pygame.display.set_mode([1000, 1000])
reward_record_1 = []
reward_record_2 = []

if __name__ == "__main__":
    #set up the arena
    arena = Enviroment()
    #add the gladiators
    for name in GLADIATOR_NAMES:
        spawn_point ={'x':randint(0,ARENA_SIZE), 'y':randint(0,ARENA_SIZE)}
        arena.add_gladiator(name, spawn_point)

    for epoch in tqdm(range(TRAINING_EPISODES)):
        for gladiator in arena.gladiators:
            spawn_point ={'x':randint(0,ARENA_SIZE), 'y':randint(0,ARENA_SIZE)}
            #spawn_point ={'x':randint(0,ARENA_SIZE*(epoch/TRAINING_EPISODES+0.1)), 'y':randint(0,ARENA_SIZE*(epoch/TRAINING_EPISODES+0.1))}
            spawn_point ={'x':randint(0,10), 'y':randint(0,10)}
            gladiator.respawn(spawn_point)

        for t in range(TIMER):
            #calculate arena next state
            arena.run_frame(time=t)
            #draw arena state
            
            if VIEW and epoch%VIEW_EVERY == 0:                
                arena.draw(screen)
                pygame.display.update()
                clock.tick(FPS)
            
            
            #check if the game ended
            if len(arena.gladiators) ==1:
                print(f"Gladiator {arena.gladiators[0].name} won the game")
                break
            elif len(arena.gladiators) == 0:
                print("The game ended in a draw")
                break
        print(f"Gladiator {arena.gladiators[0].name} reward {arena.gladiators[0].reward} ")
        print(f"Gladiator {arena.gladiators[1].name} reward {arena.gladiators[1].reward} ")
        reward_record_1.append(arena.gladiators[0].reward)
        reward_record_2.append(arena.gladiators[1].reward)

plt.plot(reward_record_1, label="Gladiator A")
plt.plot(reward_record_2, label="Gladiator B")
plt.legend()
plt.show()
pygame.quit()
