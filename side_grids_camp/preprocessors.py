import numpy as np
import tensorflow as tf

"""
Preprocessors from pycolab pixel data to feature vector for linear methods. Note
that more complex representations can be built up by concatenating the output
of multiple preprocessors.
"""
class Reshaper():
    """Reshapes m by n grayscale pixel matrix to a length mn vector. If a
    reference image is specified, then the difference between the pixel matrix
    and the reference is given.
    """

    def __init__(self, im_width, im_height, ref=None):
        """Initialise preprocessor for image of a given size and store reference
        image. If reference image is not given, then a vector of zeros is used.

        Args:
            im_width: Image width in pixels
            im_height: Image height in pixels
            ref: reference grayscale image
        """
        self.im_width = im_width
        self.im_height = im_height

        if ref is None:
            ref = tf.zeros([im_width, im_height])

        assert ref.shape == tf.TensorShape([im_width, im_height])
        self.ref = tf.reshape(ref, [im_width * im_height, -1])

    def process(self, img):
        """Process the image to give a linear reshaping with the reference image
        subtracted (this will be zero if it was not given earlier).

        Args:
            img: Tensorflow tensor containing pixel grayscale values
        """
        assert img.shape == tf.TensorShape([self.im_width, self.im_height])
        return tf.reshape(img, [self.im_width * self.im_height, -1]) - self.ref

class ObjectDistances():
    """Takes in a vector of grayscale pairs [[colour1, colour2], ...] and
    returns a vector of twice this length containing the minimum horizontal and
    vertical distances between objects of the respective colours.
    """
    def __init__(self, colourpairs):
        """Initialise preprocessor for image of a given size and store colour
        pairs. These are used to find the blocks in the gridworld that we want
        to measure the distance between.
        """
        self.colourpairs = colourpairs

    def process(self, img):
        """Process the image to give horizontal and vertical distances between
        nearest objects of the respective colours.

        Args:
            img: Tensorflow tensor containing pixel grayscale values
        """
        for c1, c2 in self.colourpairs:
            coords1 = tf.where(tf.equal(img, c1))
            coords2 = tf.where(tf.equal(img, c2))

            ## TODO: need to find a way to get pairwise distances
            ## TODO: then select the entries corresponding to the nearest pair
