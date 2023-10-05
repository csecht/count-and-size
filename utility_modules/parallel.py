"""A module to run parallel processes for contouring watershed basins."""
# Copyright (C) 2021-2023 C. Echt under GNU General Public License'

# Standard library imports.
from multiprocessing.pool import Pool

import cv2
import numpy as np

# Local application imports.
# pylint: disable=import-error
from utility_modules import constants as const

# There is a bug(?) in PyCharm that does not recognize cv2 memberships,
#  so pylint and inspections flag every use of cv2.*.
# Be aware that this disables all checks of (E1101): *%s %r has no %r member%s*
# pylint: disable=no-member

class MultiProc:
    """
    A Class that handles image multiprocessing outside the tk.TK app Classes
    so that pickling errors are avoided.
    Methods:
         contour_the_segments: the multiprocessing.Pool.map() func argument.
         pool_it: handles multiprocessing.Pool.map().
    """
    def __init__(self, image):
        self.image = image

    def contour_the_segments(self, label: int) -> np.ndarray:
        """
        Used as func in multiprocess.Pool to generate watershed contours
        for watershed basins or random_walker segments.
        Called from pool_it().
        Args:
            label: an integer element from the list of marker indexes
                   from a labeled image.
        Returns:
            The indexed (labeled basin) ndarray element of the largest
            contours to be added to a contour list for object sizing.
        """

        basin_mask = np.zeros(shape=self.image.shape, dtype="uint8")
        basin_mask[self.image == label] = 255

        # Detect contours in the masked basins and return the largest one.
        contours, _ = cv2.findContours(image=basin_mask,
                                       mode=cv2.RETR_EXTERNAL,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    @property
    def pool_it(self) -> list:
        """
        Parallel processing to find contours in the image specified by the
        Class attribute. Calls MultiProc.contour_the_segments().
        Called by ProcessImage.watershed_segmentation() or
        ProcessImage.randomwalk_segmentation().
        Usage: contours = parallel.MultiProc(img).pool_it

        Returns:
            List of watershed contours used for object sizing.
         """
        # Note: using multiprocessing to find ws contours takes less
        #  time than finding them serially with a for-loop.
        # DON'T use concurrent.futures or async here because we don't
        #  want the user doing concurrent things that may wreck the flow.
        # Note that the const.NCPU value is 1 less than the number of
        #  physical cores.
        # chunksize=40 was empirically optimized for speed using the
        #  sample images on a 6-core HP Pavilion Windows 11 laptop.
        with Pool(processes=const.NCPU) as mpool:
            contours: list = mpool.map(func=self.contour_the_segments,
                                       iterable=np.unique(ar=self.image),
                                       chunksize=40)

        return contours
