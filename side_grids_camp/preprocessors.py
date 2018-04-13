import numpy as np
import tensorflow as tf

"""
Preprocessors from pycolab pixel data to feature vector for linear methods.
"""
class Reshaper():
    """Reshapes m by n grayscale pixel matrix to a length mn vector. If a
    reference image is specified, then the difference between the pixel matrix
    and the reference is given.
    """

    def __init__(self, im_width, im_height, ref=None):
        """Initialise preprocessor for image of a given size and store reference
        image.

        Args:
            im_width: Image width in pixels
            im_height: Image height in pixels
            ref: ref grayscale image
        """
        self.im_width = im_width
        self.im_height = im_height

        if ref is None:
            ref = tf.zeros([im_width, im_height])

        assert ref.shape == tf.TensorShape([im_width, im_height])
        self.ref = tf.reshape(ref, [im_width * im_height, -1])

    def process(self, img):
        assert img.shape == tf.TensorShape([self.im_width, self.im_height])
        return tf.reshape(img, [self.im_width * self.im_height, -1]) - self.ref
