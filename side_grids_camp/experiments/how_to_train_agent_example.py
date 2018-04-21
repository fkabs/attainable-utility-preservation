#
# Let's try to solve cliff-walking env with pycolab and Q-learning
#
import numpy as np
import datetime
from side_grids_camp.experiments.cliff_walking_env import make_game as cliff_game
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game


# %% let's play with usual pycolab game
print("Start training cliff game.")
game = cliff_game()

start_time = datetime.datetime.now()
ret = 0

observation, reward, _ = game.its_showtime()
while not game.game_over:
    # action = supa_safe_agent.act(observation)  # implement this
    action = np.random.choice(4)
    observation, reward, _ = game.play(action)
    # supa_safe_agent.learn(observation, action, reward, game.game_over)  # implement this
    ret += reward

elapsed = datetime.datetime.now() - start_time
print("Return: {}, elasped: {}".format(ret, elapsed))
print("Traning finished.")


# %% let's try side effects sokoban
# it turns out it's a little different then before..
print("Start training side effects sokoban.")
env = sokoban_game(level=0)

start_time = datetime.datetime.now()
ret = 0

actions = env.action_spec().maximum + 1
time_step = env.reset()  # for the description of timestep see ai_safety_gridworlds.environments.shared.rl.environment
while not time_step.last():
    # action = supa_safe_agent.act(time_step.observation)  # implement this
    action = np.random.choice(actions)
    time_step = env.step(action)
    # supa_safe_agent.learn(time_step, action)  # implement this
    ret += time_step.reward

elapsed = datetime.datetime.now() - start_time
print("Return: {}, elasped: {}.".format(ret, elapsed))
print("Performance: {}.".format(env.get_last_performance()))
print("Traning finished.")
