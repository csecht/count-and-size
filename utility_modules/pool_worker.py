"""
A companion module to parallel.MultiProc().pool_it for running parallel
processes on skimage images.
Derived from: https://docs.python.org/3/library/multiprocessing.html
              https://stackoverflow.com/questions/67847898/
                multiprocessing-in-python-hanging-the-system
"""
# Copyright (C) 2023 C. Echt under GNU General Public License'

# Third party imports.
import cv2
import numpy as np
# There is a bug(?) in PyCharm that does not recognize cv2 memberships,
#  so pylint and inspections flag every use of cv2.*.
# Be aware that this disables all checks of (E1101): *%s %r has no %r member%s*
# pylint: disable=no-member

def contour_area(img: np.ndarray, idx: int) -> np.ndarray:
    """
    Used as a worker function for multiprocess.Pool to generate contours
    of watershed basins or random_walker segments.
    This worker, in a separate py file, is necessary to avoid hanging on
    "Process ForkPoolWorker" iterations.
    Called from parallel.MultiProc().contour_the_segments().
    Args:
        img: the labeled skimage array being iterated over by
             multiprocessing.Pool.map().
        idx: the implicit integer element from the list of marker indexes
               of the labeled *image*.
    Returns:
        An indexed (labeled basin) ndarray element of a segment's largest
        contour area to be added to a Pool.map() list.
    """

    basin_mask = np.zeros(shape=img.shape, dtype="uint8")
    basin_mask[img == idx] = 255

    # Detect contours in the masked basins and return the largest one.
    contour, _ = cv2.findContours(image=basin_mask,
                                   mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    return max(contour, key=cv2.contourArea)
