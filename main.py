"""In this file is runned the game simulation and visualization"""
from random import randint
from basic_classes import Enviroment
from config import GLADIATOR_NAMES,TIMER,ARENA_SIZE
from tqdm import tqdm
import pygame
from pygame.time import Clock

pygame.init()
clock = Clock()  
FPS = 30  

#set up the drawing window
screen = pygame.display.set_mode([1000, 1000])


if __name__ == "__main__":
    #set up the arena
    arena = Enviroment()
    #add the gladiators
    for name in GLADIATOR_NAMES:
        spawn_point ={'x':randint(0,ARENA_SIZE), 'y':randint(0,ARENA_SIZE)}
        arena.add_gladiator(name, spawn_point)

    for t in tqdm(range(TIMER)):
        #calculate arena next state
        arena.run_frame(time=t)
        #draw arena state
        arena.draw(screen)

        #update the screen
        pygame.display.update()
        clock.tick(FPS)

        if len(arena.gladiators) ==1:
            print(f"Gladiator {arena.gladiators[0].name} won the game")
            break
        elif len(arena.gladiators) == 0:
            print("The game ended in a draw")
            break
 