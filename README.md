# Standing Still Is Not An Option: Alternative Baselines for Attainable Utility Preservation

A test-bed for the [Attainable Utility Preservation (AUP)](https://arxiv.org/abs/1902.09725) method for quantifying and penalizing the change an agent has on the world around it. Current AUP approaches however assume the existence of a no-op action in the environment’s action space, which limits AUP to solve tasks where doing nothing for a single time-step is a valuable option. Depending on the environment, this cannot always be guaranteed.
We introduce four different baselines that do not build on such actions and therefore extend the concept of AUP to a broader class of environments. We evaluate all introduced variants on different AI safety gridworlds and show that this approach generalizes AUP to a broader range of tasks, with only little performance losses.

_This repository further augments [this expansion](https://github.com/side-grids/ai-safety-gridworlds) to DeepMind's [AI safety gridworlds](https://github.com/deepmind/ai-safety-gridworlds). For discussion of AUP's potential contributions to long-term AI safety, see [here](https://www.lesswrong.com/s/7CdoznhJaLEKHwvJW)._


## Installation
1. Using Python 2.7 as the interpreter, acquire the libraries in `requirements.txt`.
2. Clone using `--recursive` to snag the `pycolab` submodule:
`git clone --recursive https://github.com/fkabs/attainable-utility-preservation.git`.
1. Run `python -m experiments.charts` or `python -m experiments.ablation`, tweaking the code to include the desired environments.

---

_Work under review at International IFIP Cross Domain (CD) Conference for Machine Learning & Knowledge Extraction (MAKE) [CD-MAKE 2023](https://cd-make.net/)_
