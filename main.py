"""In this file is runned the game simulation and visualization"""
import os
from random import randint
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from basic_classes import Enviroment
from config import GLADIATOR_NAMES,TIMER,ARENA_SIZE,TRAINING_EPISODES,VIEW,VIEW_EVERY,SCREEN_HEIGHT,SCREEN_WIDTH, SIMUALTION_RECORD_PATH
from tqdm import tqdm


fig = plt.figure(figsize=(SCREEN_WIDTH,SCREEN_HEIGHT))
plt.xlim(0,ARENA_SIZE)
plt.ylim(0,ARENA_SIZE)

ax =  fig.add_subplot(111)
metadata = dict(title=f"Simulation movie", artist="Baschi")
writer=PillowWriter(fps=30, metadata=metadata)
#writer=FFMpegWriter(fps=30, metadata=metadata)

def spawn_gladiators(arena):
    for gladiator in arena.gladiators:
        #spawn_point ={'x':randint(0,ARENA_SIZE), 'y':randint(0,ARENA_SIZE)}
        #spawn_point ={'x':randint(0,ARENA_SIZE*(epoch/TRAINING_EPISODES+0.1)), 'y':randint(0,ARENA_SIZE*(epoch/TRAINING_EPISODES+0.1))}
        spawn_point ={'x':randint(0,10), 'y':randint(0,10)}
        gladiator.respawn(spawn_point)

def run_episode(arena, draw=False):
    with tqdm(total=TIMER, desc="Episode progress", leave=False) as pbar:
        spawn_gladiators(arena)
        for t in range(TIMER):
            #calculate arena next state
            arena.run_frame(time=t)
    
            #draw arena state
            if draw:
                 arena.draw(ax,time=t)
                 writer.grab_frame()
                 ax.clear()
                 plt.xlim(0,ARENA_SIZE)
                 plt.ylim(0,ARENA_SIZE)
    
            #check if the game ended
            if len(arena.gladiators) ==1:
                print(f"Gladiator {arena.gladiators[0].name} won the game")
                break
            elif len(arena.gladiators) == 0:
                print("The game ended in a draw")
                break
            pbar.update()

if __name__ == "__main__":
    #set up the arena
    arena = Enviroment()
    #add the gladiators
    for name in GLADIATOR_NAMES:
        first_spawn_point ={'x':randint(0,ARENA_SIZE), 'y':randint(0,ARENA_SIZE)}
        arena.add_gladiator(name, first_spawn_point)
     
    with tqdm(total=TRAINING_EPISODES, desc="Training progress") as pbar: 
        for epoch in range(TRAINING_EPISODES):
            if VIEW and epoch%VIEW_EVERY == 0:
                with writer.saving(fig, os.path.join(SIMUALTION_RECORD_PATH,f"recording_epoch_{epoch}.gif"), 100):
                #with writer.saving(fig, os.path.join(SIMUALTION_RECORD_PATH,f"recording_epoch_{epoch}.mp4"), 100):
                    run_episode(arena,draw=True)
            else:
                run_episode(arena)   
            pbar.update() 