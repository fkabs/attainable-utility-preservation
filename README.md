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

Work published at [Springer Nature Switzerland](https://link.springer.com/chapter/10.1007/978-3-031-40837-3_15).

Please use following reference to cite this work:

```{bibtex}
@inproceedings{eresheim2023,
  title = {Standing {{Still Is Not}} an~{{Option}}: {{Alternative Baselines}} for~{{Attainable Utility Preservation}}},
  shorttitle = {Standing {{Still Is Not}} an~{{Option}}},
  booktitle = {Machine {{Learning}} and {{Knowledge Extraction}}},
  author = {Eresheim, Sebastian and Kovac, Fabian and Adrowitzer, Alexander},
  editor = {Holzinger, Andreas and Kieseberg, Peter and Cabitza, Federico and Campagner, Andrea and Tjoa, A. Min and Weippl, Edgar},
  date = {2023},
  series = {Lecture {{Notes}} in {{Computer Science}}},
  pages = {239--257},
  publisher = {{Springer Nature Switzerland}},
  location = {{Cham}},
  doi = {10.1007/978-3-031-40837-3_15},
  abstract = {Specifying reward functions without causing side effects is still a challenge to be solved in Reinforcement Learning. Attainable Utility Preservation (AUP) seems promising to preserve the ability to optimize for a correct reward function in order to minimize negative side-effects. Current approaches however assume the existence of a no-op action in the environment's action space, which limits AUP to solve tasks where doing nothing for a single time-step is a valuable option. Depending on the environment, this cannot always be guaranteed. We introduce four different baselines that do not build on such actions and therefore extend the concept of AUP to a broader class of environments. We evaluate all introduced variants on different AI safety gridworlds and show that this approach generalizes AUP to a broader range of tasks, with only little performance losses.},
  isbn = {978-3-031-40837-3},
  langid = {english}
}
```
