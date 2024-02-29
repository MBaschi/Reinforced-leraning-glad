"""In this file is runned the game simulation and visualization"""

import os
from random import randint
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from basic_classes import Enviroment
from config import (
    GLADIATOR_NAMES,
    TIMER,
    ARENA_SIZE,
    TRAINING_EPISODES,
    VIEW,
    VIEW_EVERY,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SIMUALTION_RECORD_PATH,
    FPS,
)
from tqdm import tqdm


fig = plt.figure(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT))
plt.xlim(0, ARENA_SIZE)
plt.ylim(0, ARENA_SIZE)

ax = fig.add_subplot(111)
metadata = dict(title=f"Simulation movie", artist="Baschi")
writer = PillowWriter(fps=FPS, metadata=metadata)
# writer=FFMpegWriter(fps=30, metadata=metadata)
#tensorboard_writer = SummaryWriter()

def spawn_gladiators(arena):
    i=0
    for gladiator in arena.gladiators:
        # spawn_point ={'x':randint(0,ARENA_SIZE), 'y':randint(0,ARENA_SIZE)}
        # spawn_point ={'x':randint(0,ARENA_SIZE*(epoch/TRAINING_EPISODES+0.1)), 'y':randint(0,ARENA_SIZE*(epoch/TRAINING_EPISODES+0.1))}
        spawn_point = {"x": randint(int(0.45*ARENA_SIZE), int(0.55*ARENA_SIZE)), "y": randint(int(0.45*ARENA_SIZE), int(0.55*ARENA_SIZE))}
        gladiator.respawn(spawn_point)


def run_episode(arena, draw=False, epsilon=0.1):
    with tqdm(total=TIMER, desc="Episode progress", leave=False) as pbar:
        spawn_gladiators(arena)
        for t in range(TIMER):
            # calculate arena next state
            game_ended = arena.run_frame(time=t, epsilon=epsilon)

            # draw arena state
            if draw:
                arena.draw(ax, time=t)
                writer.grab_frame()
                ax.clear()
                plt.xlim(0, ARENA_SIZE)
                plt.ylim(0, ARENA_SIZE)

            # check if the game ended
            if game_ended:
                break

            pbar.update()


if __name__ == "__main__":
    # set up the arena
    arena = Enviroment()
    # add the gladiators
    for name in GLADIATOR_NAMES:
        first_spawn_point = {"x": randint(0, ARENA_SIZE), "y": randint(0, ARENA_SIZE)}
        arena.add_gladiator(name, first_spawn_point)

    with tqdm(total=TRAINING_EPISODES, desc="Training progress") as pbar:
        for epoch in range(TRAINING_EPISODES):
            if VIEW and epoch % VIEW_EVERY == 0:
                with writer.saving(
                    fig,
                    os.path.join(
                        SIMUALTION_RECORD_PATH, f"recording_epoch_{epoch}.gif"
                    ),
                    100,
                ):
                    # with writer.saving(fig, os.path.join(SIMUALTION_RECORD_PATH,f"recording_epoch_{epoch}.mp4"), 100):
                    run_episode(arena, draw=True, epsilon=epoch/TRAINING_EPISODES)
            else:
                run_episode(arena,epsilon=epoch/TRAINING_EPISODES)
            pbar.update()
