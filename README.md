# Attainable Utility Preservation

A test-bed for the approach outlined in [this paper], further augmenting [this expansion](https://github.com/side-grids/ai-safety-gridworlds) to DeepMind's [AI safety gridworlds](https://github.com/deepmind/ai-safety-gridworlds). For discussion of AUP's potential contribution to long-term AI safety, see [here](https://www.alignmentforum.org/posts/yEa7kwoMpsBgaBCgb/towards-a-new-impact-measure).

## Installation
1. Using Python 2.7 as the interpreter, acquire the libraries in `requirements.txt`.
2. Clone the repository using `--recursive` to snag the `pycolab` submodule:
`git clone --recursive https://github.com/alexander-turner/attainable-utility-preservation.git
`.
3. Run `charts.py` or `ablation.py`, tweaking the code to include the desired subset of environments. 

## Environments

>Our environments are Markov Decision Processes. All environments use a grid of
size at most 10x10. Each cell in the grid can be empty, or contain a wall or
other objects... The agent is located in one cell on
the grid and in every step the agent takes one of the actions from the action
set A = {`up`, `down`, `left`, `right`, `no-op`}. Each action modifies the agent's position to
the next cell in the corresponding direction unless that cell is a wall or
another impassable object, in which case the agent stays put.

>The agent interacts with the environment in an episodic setting: at the start of
each episode, the environment is reset to its starting configuration (which is
possibly randomized). The agent then interacts with the environment until the
episode ends, which is specific to each environment. We fix the maximal episode
length to 20 steps. Several environments contain a goal cell... If
the agent enters the goal cell, it receives a reward of +1 and the episode
ends.

>In the classical reinforcement learning framework, the agent's objective is to
maximize the cumulative (visible) reward signal. While this is an important part
of the agent's objective, in some problems this does not capture everything that
we care about. Instead of the reward function, we evaluate the agent on the
performance function *that is not observed by the agent*. The performance
function might or might not be identical to the reward function. In real-world
examples, the performance function would only be implicitly defined by the
desired behavior the human designer wishes to achieve, but is inaccessible to
the agent and the human designer.


### `Box`
![](https://i.imgur.com/UT4OvOi.png)
![](https://i.imgur.com/Cnplx2f.gif)

### `Dog`
![](https://i.imgur.com/cV6E2VQ.png)
![](https://i.imgur.com/1qdKHjX.gif)

### `Survival`
![](https://i.imgur.com/t2lvvsb.gif)

### `Conveyor`
![](https://i.imgur.com/yUu15Va.png)
![](https://i.imgur.com/eskrHjf.gif)

### `Sushi`
![](https://i.imgur.com/fRvHkTs.png)
![](https://i.imgur.com/tuBiErI.gif)

### `Vase`
![](https://i.imgur.com/AHwuHPK.png)
![](https://i.imgur.com/glGaytb.gif)

### `Burning`
![](https://i.imgur.com/gTmyyHM.png)