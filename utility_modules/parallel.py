"""A module to run parallel processes for contouring skimage segments."""
# Copyright (C) 2023 C. Echt under GNU General Public License'

# Standard library imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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

        # This implementation of Lock(), close(), and join() reduces
        #   intermittent hanging (deadlock) of the Pool() function.
        # When used with method 'fork' in Linux, it is very fast, but
        #  'fork' is subject to deadlocks.
        # Idea and explanation for using Lock() from @dano at:
        #  https://stackoverflow.com/questions/25557686/
        #   python-sharing-a-lock-between-processes/25558333#25558333
        if const.MY_OS == 'lin':
            def init(lock):
                self.lock = lock

            lock_it = Lock()

            # chunksize=40 was empirically optimized for speed using the
            #  sample images on a 6-core HP Pavilion Windows 11 laptop.
            with Pool(processes=const.NCPU,
                      initializer=init,
                      initargs=(lock_it, )) as mpool:
                contours: list = mpool.map(func=self.contour_the_segments,
                                           iterable=np.unique(ar=self.image),
                                           chunksize=40)
                mpool.close()
                mpool.join()

            return contours

        # For Windows and macOS...
        with Pool(processes=const.NCPU) as mpool:
            contours: list = mpool.map(func=self.contour_the_segments,
                                       iterable=np.unique(ar=self.image),
                                       chunksize=40)

        # Stability is ensured with chunksize=1 even though it is slower.
        # The multiprocess 'spawn' method is needed to ensure stability
        #  on Linux, that is, it avoids deadlocks that happen with 'fork'.
        #  'fork' is about 2x faster than 'spawn', but, oh well.
        #  'spawn' is the default method on Windows and macOS.
        # get_context() is needed here because when mp.set_start_method('spawn')
        #  is used in if __name__ == "__main__" of size_it_rw.py, for unknown
        #  reasons, a tk error is raised for no attribute of randomwalk_segmentation().
        with ProcessPoolExecutor(max_workers=1, #const.NCPU,
                                 mp_context=mp.get_context('spawn')) as executor:
            contours = list(executor.map(self.contour_the_segments,
                                            np.unique(ar=self.image),
                                         chunksize=1))

        return contours
