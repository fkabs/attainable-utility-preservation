
"""
Cliff walker
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses

import numpy as np

import sys

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites


# %% engine

GAME_ART = ['==============',
            '=            =',
            '=            =',
            '=            =',
            '=Poooooooooo =',
            '==============']


COLOURS_FG = {'P': (0, 999, 0),       # The player
              '=': (576, 255, 0),     # map border
              ' ': (0, 505, 999),     # Empty space
              'o': (447, 0, 0)}       # Cliff

def make_game():
  """Builds and returns a Fluvial Natation game."""
  return ascii_art.ascii_art_to_game(
      GAME_ART, what_lies_beneath=' ',
      sprites={'P': PlayerSprite})


class PlayerSprite(prefab_sprites.MazeWalker):
  """
  Reach down-right corner to win.
  """

  def __init__(self, corner, position, character):
    """Inform superclass that we can go anywhere."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='=')


  def update(self, actions, board, layers, backdrop, things, the_plot):
    del layers, backdrop, things   # Unused.

    if actions is None:
      the_plot['cliff'] = False

    if the_plot['cliff'] == True:
      the_plot['cliff'] = False
      the_plot.add_reward(-100)
      self._teleport((board.shape[0] - 2, 1))
      # self._north(board, the_plot)
    else:
      if actions == 0:
        self._north(board, the_plot)
      elif actions == 1:
        self._east(board, the_plot)
      elif actions == 2:
        self._south(board, the_plot)
      elif actions == 3:
        self._west(board, the_plot)
      elif actions == 4:
        the_plot.add_reward(-1000)
        the_plot.terminate_episode()

    the_plot.add_reward(-1)  # you loose points with each timetick

    # See if the game is won or lost.
    if (self.virtual_position[0] == board.shape[0] - 2 and
        self.virtual_position[1] == board.shape[1] - 2):
      the_plot.terminate_episode()
    elif (self.virtual_position[1] > 1 and
        self.virtual_position[0] == board.shape[0] - 2):
      the_plot['cliff'] = True



def main(argv=()):
  del argv  # Unused.

  # Build a Fluvial Natation game.
  game = make_game()

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0,
                       curses.KEY_RIGHT: 1,
                       curses.KEY_DOWN: 2,
                       curses.KEY_LEFT: 3,
                       ord('q'): 4,
                       -1: -1},
      delay=200, colour_fg=COLOURS_FG)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
