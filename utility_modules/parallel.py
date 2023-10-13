"""A module to run parallel processes for contouring skimage segments."""
# Copyright (C) 2023 C. Echt under GNU General Public License'

# Standard library imports
import multiprocessing
import sys
from multiprocessing.pool import Pool

# Third party imports.
import numpy as np

# Local application imports.
from . import constants as const
from . import pool_worker


def support_multiprocessing():
    """
    Use of MultiProc Class needs special support when used in a
    frozen app, e.g., a Pyinstaller executable.
    """
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        multiprocessing.freeze_support()


class MultiProc:
    """
    A Class to handle image multiprocessing outside tk.TK app Classes so
    that pickling errors can be avoided.
    Methods:
         contour_the_segments: the multiprocessing.Pool.map() func argument.
         pool_it: handles multiprocessing.Pool.map().
    """
    def __init__(self, image):
        self.image = image

    def contour_the_segments(self, label: int) -> np.ndarray:
        """
        Used as func in multiprocess.Pool to generate segment contours
        for watershed basins or random_walker segments.
        Called from pool_it(). Calls helper module pool_worker.py.
        Args:
            label: the implicit integer element from the list of marker
                   indices in a labeled image.
        Returns:
            From a worker module, he indexed (labeled basin) ndarray
            element of the largest contours to be added to a contour list
            for object sizing.
        """

        return pool_worker.contour_area(img=self.image, idx=label)

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
