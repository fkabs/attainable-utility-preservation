# Variations of Attainable Utility Preservation (vAUP)
A test-bed for the [Attainable Utility Preservation (AUP)](https://arxiv.org/abs/1902.09725) method for quantifying and penalizing the change an agent has on the world around it. This repository further augments [this expansion](https://github.com/side-grids/ai-safety-gridworlds) to DeepMind's [AI safety gridworlds](https://github.com/deepmind/ai-safety-gridworlds). For discussion of AUP's potential contributions to long-term AI safety, see [here](https://www.lesswrong.com/s/7CdoznhJaLEKHwvJW).

With this work, we also introduce vAUP, a modular extension to AUP with different variations, which are applicable to environments with a no-op action and action-driven environments alike. This method allows to choose between variants based on the environments to solve the safety property of avoiding side effects and to optimize an agent for a correct reward function implicitly. We evaluated all introduced variants on the same AI safety griworlds and show that this approach induces safe, conservative and effective behavior.


## Installation
1. Using Python 2.7 as the interpreter, acquire the libraries in `requirements.txt`.
2. Clone using `--recursive` to snag the `pycolab` submodule:
`git clone --recursive https://github.com/fkabs/attainable-utility-preservation.git`.
1. Run `python -m experiments.charts` or `python -m experiments.ablation`, tweaking the code to include the desired environments.
