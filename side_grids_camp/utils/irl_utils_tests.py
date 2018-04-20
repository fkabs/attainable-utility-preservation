#
# IRL UTILS TESTS
#
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np
from ai_safety_gridworlds.environments.shared.safety_game import Actions
from side_grids_camp.feature_extractors import Reshaper, ObjectDistances
from side_grids_camp.utils.irl_utils import *


# %% So dirty that you shouldn't read :P
class EmptyF():
    def process(self, state):
        return np.array([0])


# %% TESTS:
st_probs, _ = get_state_probs(state_board_map, board_state_map, [EmptyF()])
s = 34
a = 1
ss = st_probs[s, a, :].argmax()
# # Action codes:
# for i in range(len(Actions)): print("Action {} is ".format(i) + str(Actions(i)))
print("Chosen actions is "+ str(Actions(a)))
# %%
# pl_x, pl_y, box_x, box_y = sb_map[s]
env = get_game_at(*state_board_map[s])
env.reset().observation['board']
# %%
env = get_game_at(*state_board_map[ss])
env.reset().observation['board']
# %% check goal state probs
env = get_game_at(4, 4, 1, 2)
ts = env.reset()
ts.observation['board']

final_st = []
for px, py, bx, by in board_state_map:
    if px == 4 and py == 4:
        final_st.append(board_state_map[(px,py,bx,by)])
final_st

for st in final_st:
    print("State {}, 1 at {}, sum: {}".format(st, [st_probs[st, a, :].argmax() for a in range(4)],st_probs[st, :, :].sum()))
# Should be (zero probs of going anywhere):
# State 54, 1 at [0, 0, 0, 0], sum: 0.0
# State 56, 1 at [0, 0, 0, 0], sum: 0.0
# State 59, 1 at [0, 0, 0, 0], sum: 0.0
# State 55, 1 at [0, 0, 0, 0], sum: 0.0
# State 57, 1 at [0, 0, 0, 0], sum: 0.0
# State 58, 1 at [0, 0, 0, 0], sum: 0.0

# %% Tests to grayscale operations
env = get_game_at(3, 4, 1, 2)
world_shape = env.observation_spec()['board'].shape
sp = StateProcessor(world_shape[0], world_shape[1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    frame_ref = np.moveaxis(sokoban_game(level=0).reset().observation['RGB'], 0, -1)
    frame_ref = sp.process(sess, frame_ref)

    time_step = env.reset()
    frame1 = np.moveaxis(time_step.observation['RGB'], 0, -1)
    frame1 = sp.process(sess, frame1)

    time_step = env.step(Actions.UP)
    frame2 = np.moveaxis(time_step.observation['RGB'], 0, -1)
    frame2 = sp.process(sess, frame2)
    print(frame2)


# %% check getting states from grayscale
time_step.observation['board']
get_state_from_grayscale(frame2)
pl_box_coords(time_step.observation['board'])
board_state_map[(2,4,1,2)]


# %% check feature extractors
frame_ref.shape
reshaper = Reshaper(6, 6, ref=frame_ref)

reshaper.process([frame1]*2).shape

obj_dists = ObjectDistances([(78, 152)])
obj_dists.process([frame1]*2).shape

features = [reshaper, obj_dists]

st_pr1, f_m = get_state_probs(state_board_map, board_state_map, features)

# %%
def get_state_probs_old(sb_map, bs_map, actions=4):
    sts = len(sb_map)
    state_probs = np.zeros((sts, actions, sts))
    for state in range(sts):
        pl_x, pl_y, box_x, box_y = sb_map[state]
        env = get_game_at(pl_x, pl_y, box_x, box_y)
        for action in range(4):
            env.reset()
            time_step = env.step(action)
            state_probs[state, action, bs_map[pl_box_coords(time_step.observation['board'])]] = 1

    return state_probs
st_pr0 = get_state_probs_old(state_board_map, board_state_map)
np.array_equal(st_pr0, st_pr1)

f_m.shape


# %%

obj_dists.process([frame1, frame1])
obj_dists.process([frame1, frame2])
obj_dists.process([frame1, frame_ref]).shape


plt.imshow(frame_ref, cmap='gray')
plt.imshow(frame1, cmap="gray")
plt.imshow(frame2, cmap='gray')

# %% mechanics of computing object distances

img = frame_ref
output = []

# for c1, c2 in [(78, 152)]:
c1, c2 = (78, 152)
coords1 = np.argwhere(img == c1)
coords2 = np.argwhere(img == c2)

z1 = np.concatenate([coords1]*len(coords2))
z2 = np.concatenate([coords2]*len(coords1))

coord_diffs = np.abs(z1-z2)
dists = coord_diffs.sum(axis=1)
closest = np.argmin(dists) # NOTE: chooses first in case of tie

x_dist, y_dist = coord_diffs[closest]

output.append(x_dist)
output.append(y_dist)
output
