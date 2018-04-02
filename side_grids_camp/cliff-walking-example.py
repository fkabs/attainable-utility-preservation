#
# Let's try to solve cliff-walking env with pycolab and Q-learning
#
import numpy as np
import datetime
from side_grids_camp.experiments.cliff_walking_env import make_game

# %% let's play
print("Start training.")
game = make_game()
start_time = datetime.datetime.now()
ret = 0

observation, reward, _ = game.its_showtime()
while not game.game_over:
    # replace with some agent here:
    action = np.random.choice(4)
    observation, reward, _ = game.play(action)
    # add some learning here
    ret += reward

elapsed = datetime.datetime.now() - start_time
print("Return: {}, elasped: {}".format(ret, elapsed))
print("Traning finished.")
