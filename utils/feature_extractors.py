import numpy as np

"""
    All of these processors take input in the following format
        state1 = 2D array
        state2 = 2D array
        statePair = np.stack([state1, state2], axis=2)
"""


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

        assert ref.shape == (im_width, im_height)
        self.ref = np.reshape(ref, [im_width * im_height, -1])

    def process(self, statePair):
        """Process the image to give a linear reshaping with the reference image
        subtracted (this will be zero if it was not given earlier).

        Args:
            statePair: list of 2 grayscale images [prev_img, img]
        """
        #_, img = statePair
        img = statePair[:,:,1]
        assert img.shape == (self.im_width, self.im_height)
        return np.squeeze(np.reshape(img, [self.im_width * self.im_height, -1]) - self.ref)


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

    def process(self, statePair):
        """Process the image to give horizontal and vertical distances between
        nearest objects of the respective colours.

        Args:
            statePair: list of 2 grayscale images [prev_img, img]
        """
        img = statePair[:,:,1]

        output = []
        for c1, c2 in self.colourpairs:
            coords1 = np.argwhere(img == c1)
            coords2 = np.argwhere(img == c2)

            ## Assume that the distance is zero because one is on top
            if (len(coords1) == 0) or (len(coords2) == 0):
                output.append(0)
                output.append(0)

            ## Otherwise find the closest
            else:
                z1 = np.concatenate([coords1]*len(coords2))
                z2 = np.concatenate([coords2]*len(coords1))

                coord_diffs = np.abs(z1-z2)
                dists = coord_diffs.sum(axis=1)
                closest = np.argmin(dists) # NOTE: chooses first in case of tie

                x_dist, y_dist = coord_diffs[closest]

                output.append(x_dist)
                output.append(y_dist)

        return np.array(output)


# Remove elements corresponding to wall and floor tiles. Distorts shape.
def trim_walls_floors(statePair, floor, wall) :
    return statePair[ (statePair != floor) & (statePair != wall) ]


def diff_two_states(current, previous, object_type):

        currentWeirdLocations = np.where(current == object_type)
        currentIndices = np.stack(currentWeirdLocations, axis=1)
        prevWeirdLocations = np.where(previous == object_type)
        prevIndices = np.stack(prevWeirdLocations, axis=1)

        return np.flip(currentIndices - prevIndices, axis=1)



class CountAllObjects():
    def __init__(self, floor, wall, delta=False):
        self.delta = delta
        self.floorCode = floor
        self.wallCode = wall

    def process(self, statePair):
        current = statePair[:,:,1]
        previous = statePair[:,:,0]

        currentObjs = trim_walls_floors(current, self.floorCode, self.wallCode)
        prevObjs = trim_walls_floors(previous, self.floorCode, self.wallCode)

        if (self.delta == True):
            return len(currentObjs) - len(prevObjs)
        else:
            return len(currentObjs)



class CountOfTypes():
    def __init__(self, floor, wall, delta=False):
        self.delta = delta
        self.floorCode = floor
        self.wallCode = wall

    def process(self, statePair):
        current = statePair[:,:,1]
        previous = statePair[:,:,0]

        currentObjs = trim_walls_floors(current, self.floorCode, self.wallCode)
        prevObjs = trim_walls_floors(previous, self.floorCode, self.wallCode)

        if (self.delta == True):
            return len( set(np.unique(currentObjs)) \
                        - set(np.unique(prevObjs)) )
        else:
            return len(np.unique(currentObjs))



class CountObjectsOfType():
    """Takes object type (a greyscale value from 0 to 255) and returns
    the number of objects of that type present in the most recent statePair."""

    def __init__(self, object_type, delta=False):
        """Initialises feature extractor to count the number of objects of given
        type present in the most recent statePair (where type is represented as a
        greyscale value from 0 to 255). If delta is true, returns the change in
        the number of objects from previous statePair to current statePair."""
        self.object_type = object_type
        self.delta = delta

    def process(self, statePair):
        current = statePair[:,:,1]
        previous = statePair[:,:,0]

        if (self.delta == True):
            # return difference in num of objs between current statePair and previous statePair
            return (current == self.object_type).sum() - \
                    (previous == self.object_type).sum()
        else:
            # return number of objects of given type in current statePair
            return (current == self.object_type).sum()




# Assume: no self-destruct
# Assume: counts appearance/disappearance as motion
# Returns the
class DetectMotionInObjectType():
    def __init__(self, typ):
        self.object_type = typ

    def process(self, statePair):
        current = statePair[:,:,1]
        previous = statePair[:,:,0]

        return diff_two_states(currentIndices, prevIndices)


def count_movers(state_diff, counter):
    moved = np.abs(counter.process(state_diff))

    return len(moved > 0)


class CountTypesOfMovingObjects():
    """Returns the number different types of moving objects."""

    def __init__(self, floor_code=219, wall_code=152):
        self.floor_code = floor_code
        self.wall_code = wall_code

    def process(self, statePair):
        current = statePair[:,:,1]
        previous = statePair[:,:,0]

        actual_objects = current[(current != self.wall_code) & (current != self.floor_code)]
        obj_types = np.unique(actual_objects)
        print(obj_types)
        diffs = [diff_two_states(current, previous, typ) for typ in obj_types]
        return [np.count_nonzero(diff) for diff in diffs]

def count_all_movers(state):

    counter = CountTypesOfMovingObjects()
    type_counts = counter.process(state)
    return sum(type_counts)



class IsCornered() :
    def __init__(self, wall, objectCode):
        self.wallCode = wall
        self.objectCode = objectCode

    def process(self, statePair) :
        objectCode = self.objectCode
        state = statePair[:,:,1]
        coords = np.argwhere(state == objectCode)[0]

        leftRight = [ (coords[0], max(0, coords[1]+i)) for i in range(-1,2, 2) ]
        upDown = [ (max(0, coords[0]+i), coords[1]) for i in range(-1,2, 2) ]
        xCorners = [state[el] == self.wallCode for el in leftRight]
        yCorners = [state[el] == self.wallCode for el in upDown]

        return np.array([int(sum(xCorners) > 0 and sum(yCorners) > 0)])
