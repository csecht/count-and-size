"""A module to run parallel processes for contouring skimage segments."""
# Copyright (C) 2023 C. Echt under GNU General Public License'

# Standard library imports
from multiprocessing import Lock
from multiprocessing.pool import Pool

# Third party imports.
import numpy as np

# Local application imports.
from . import constants as const
from . import pool_worker


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
        self.lock = None

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
        # Note: using multiprocessing to find segment contours takes
        #  less time than finding them serially with a for-loop, which
        #  helps with large files or many objects.

        # This implementation of Lock(), close(), and join() reduces
        #  or eliminates intermittent hanging of the Pool() function.
        # Idea and explanation for using Lock() from @dano at:
        #  https://stackoverflow.com/questions/25557686/
        #   python-sharing-a-lock-between-processes/25558333#25558333
        def init(lock):
            self.lock = lock

        lock_it = Lock()

        # chunksize=40 was empirically optimized for speed using the
        #  sample images on a 6-core HP Pavilion Windows 11 laptop.
        with Pool(processes=const.NCPU,
                  initializer=init,
                  initargs=(lock_it,)) as mpool:
            contours: list = mpool.map(func=self.contour_the_segments,
                                       iterable=np.unique(ar=self.image),
                                       chunksize=40)
            mpool.close()
            mpool.join()

        return contours
