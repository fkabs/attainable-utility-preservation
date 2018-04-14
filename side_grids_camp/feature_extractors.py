import numpy as np

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
            ref = np.zeros([im_width, im_height])

        assert ref.shape == [im_width, im_height]
        self.ref = np.reshape(ref, [im_width * im_height, -1])

    def process(self, state):
        """Process the image to give a linear reshaping with the reference image
        subtracted (this will be zero if it was not given earlier).

        Args:
            state: list of 2 grayscale images [prev_img, img]
        """
        prev_img, img = state
        assert img.shape == [self.im_width, self.im_height]
        return np.reshape(img, [self.im_width * self.im_height, -1]) - self.ref
