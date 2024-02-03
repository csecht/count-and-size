#!/usr/bin/env python3
"""
A tkinter GUI for OpenCV processing of an image to obtain sizes, means,
and ranges of objects in a sample population. The distance transform,
watershed, and random walker algorithms are used interactively by setting
their parameter values with slide bars and pull-down menus. Related
image processing factors like contrast, brightness, noise reduction,
and filtering are also adjusted interactively, with live updating of the
resulting images.

A report is provided of parameter settings, object count, individual
object sizes, and sample size mean and range, along with an annotated
image file of labeled objects.

USAGE
For command line execution, from within the count-and-size-main folder:
python3 -m size_it --about
python3 -m size_it --help
python3 -m size_it
python3 -m size_it --terminal
Windows systems may need to substitute 'python3' with 'py' or 'python'.

Note that from the initial "Set starting parameters" window, the file,
scaling cannot be changed after the "Process now" button is clicked.
Once image processing begins, if the scale setting is not to your
liking, just quit, restart, and choose different values.

Image preprocessing functions do live updates as most settings are changed.
For some slider settings however, when prompted, click the
"Run..." button to initiate the final processing step.

Save settings report and the annotated image with the "Save" button.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
report window, or from command line with Ctrl-C.

Requires Python 3.7 or later and the packages opencv-python, numpy,
scikit-image, scipy, and psutil.
See this distribution's requirements.txt file for details.
Developed in Python 3.8 and 3.9, tested up to 3.11.
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys
from datetime import datetime
from json import loads
from pathlib import Path
from statistics import mean, median
from typing import List
from time import time

# Third party imports.
# tkinter(Tk/Tcl) is included with most Python3 distributions,
#  but may sometimes need to be regarded as third-party.
# There is a bug(?) in PyCharm that does not recognize cv2 memberships,
#  so pylint and inspections flag every use of cv2.*.
# Be aware that this disables all checks of (E1101): *%s %r has no %r member%s*
# pylint: disable=no-member
try:
    import cv2
    import numpy as np
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from skimage.segmentation import watershed, random_walker
    from skimage.feature import peak_local_max
    from scipy import ndimage

except (ImportError, ModuleNotFoundError) as import_err:
    sys.exit(
        '*** One or more required Python packages were not found'
        ' or need an update:\nOpenCV-Python, NumPy, scikit-image, SciPy, tkinter (Tk/Tcl).\n\n'
        'To install: from the current folder, run this command'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install -r requirements.txt\n\n'
        'Alternative command formats (system dependent):\n'
        '   py -m pip install -r requirements.txt (Windows)\n'
        '   pip install -r requirements.txt\n\n'
        'You may also install directly using, for example, this command,'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install opencv-python\n\n'
        'A package may already be installed, but needs an update;\n'
        '   this may be the case when the error message (below) is a bit cryptic\n'
        '   Example update command:\n'
        '   python3 -m pip install -U numpy\n\n'
        'On Linux, if tkinter is the problem, then you may need:\n'
        '   sudo apt-get install python3-tk\n\n'
        'See also: https://numpy.org/install/\n'
        '  https://tkdocs.com/tutorial/install.html\n'
        '  https://docs.opencv2.org/4.6.0/d5/de5/tutorial_py_setup_in_windows.html\n\n'
        'Consider running this app and installing missing packages in a virtual environment.\n'
        f'Error message:\n{import_err}')

# Local application imports.
# pylint: disable=import-error
# Need to place local imports after try...except to ensure exit messaging.
from utility_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             to_precision as to_p)

PROGRAM_NAME = utils.program_name()


class ProcessImage(tk.Tk):
    """
    A suite of OpenCV methods to apply various image processing
    functions involved in segmenting objects from an image file.

    Class methods:
    update_image
    adjust_contrast
    reduce_noise
    filter_image
    watershed_segmentation
    draw_ws_segments
    """

    def __init__(self):
        super().__init__()

        # Note: The matching selector widgets for the following
        #  control variables are in ContourViewer __init__.
        self.slider_val = {
            # For variables in config_sliders()...
            'alpha': tk.DoubleVar(),
            'beta': tk.IntVar(),
            'noise_k': tk.IntVar(),
            'noise_iter': tk.IntVar(),
            'filter_k': tk.IntVar(),
            'plm_mindist': tk.IntVar(),
            'plm_footprint': tk.IntVar(),
            'circle_r_min': tk.IntVar(),
            'circle_r_max': tk.IntVar(),
        }

        self.scale_factor = tk.DoubleVar()

        self.cbox_val = {
            # For textvariables in config_comboboxes()...
            'morphop': tk.StringVar(),
            'morphshape': tk.StringVar(),
            'filter_type': tk.StringVar(),
            'th_type': tk.StringVar(),
            'dt_type': tk.StringVar(),
            'dt_mask_size': tk.StringVar(),
            'ws_connectivity': tk.StringVar(),
            'size_std': tk.StringVar(),
            # For color_cbox textvariable in setup_start_window()...
            'color': tk.StringVar(),
        }

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  with Label images displayed in their respective tkimg_window Toplevel.
        # The cvimg images are numpy arrays.
        self.tkimg: dict = {}
        self.cvimg: dict = {}
        image_names = ('input',
                       'gray',
                       'contrasted',
                       'reduced_noise',
                       'filtered',
                       'thresholded',
                       'transformed',
                       'Watershed',
                       'Random Walker',
                       'segmented_objects',
                       'sized')

        for _name in image_names:
            self.tkimg[_name] = tk.PhotoImage()
            self.cvimg[_name] = const.STUB_ARRAY

        # img_label dictionary is set up in ViewImage.setup_image_windows(),
        #  but is used in all Class methods here.
        self.img_label: dict = {}

        # metrics dict is populated in ViewImage.open_input().
        self.metrics: dict = {}

        self.num_dt_segments: int = 0
        self.ws_basins: list = []
        self.rw_contours: list = []
        self.sorted_size_list: list = []
        self.unit_per_px = tk.DoubleVar()
        self.num_sigfig: int = 0
        self.time_start: float = 0
        self.elapsed: float = 0

    def update_image(self,
                     tkimg_name: str,
                     cvimg_array: np.ndarray) -> None:
        """
        Process a cv2 image array to use as a tk PhotoImage and update
        (configure) its window label for immediate display, at scale.
        Calls module manage.tk_image(). Called from all methods that
        display an image.

        Args:
            tkimg_name: The key name used in the tkimg and img_label
                        dictionaries.
            cvimg_array: The new cv2 processed numpy image array.

        Returns:
            None
        """

        self.tkimg[tkimg_name] = manage.tk_image(
            image=cvimg_array,
            scale_factor=self.scale_factor.get()
        )
        self.img_label[tkimg_name].configure(image=self.tkimg[tkimg_name])

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the input self.cvimg['gray'] image.
        Updates contrast and brightness via alpha and beta sliders.
        Displays contrasted and redux noise images.
        Called by process(). Calls update_image().

        Returns:
            None
        """
        # Source concepts:
        # https://docs.opencv2.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        # https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        self.cvimg['contrasted'] = (
            cv2.convertScaleAbs(
                src=self.cvimg['gray'],
                alpha=self.slider_val['alpha'].get(),
                beta=self.slider_val['beta'].get(),
            )
        )

        self.update_image(tkimg_name='contrasted',
                          cvimg_array=self.cvimg['contrasted'])

    def reduce_noise(self) -> None:
        """
        Reduce noise in the contrast adjust image erode and dilate actions
        of cv2.morphologyEx operations.
        Called by preprocess(). Calls update_image().

        Returns:
            None
        """

        # Need (sort of) kernel to be odd, to avoid an annoying shift of
        #   the displayed image.
        _k = self.slider_val['noise_k'].get()
        noise_k = _k + 1 if _k % 2 == 0 else _k
        iteration = self.slider_val['noise_iter'].get()

        # If redux iteration slider is set to 0, then proceed without,
        # noise reduction and use the contrast image from adjust_contrast().
        if iteration == 0:
            self.update_image(tkimg_name='reduced_noise',
                              cvimg_array=self.cvimg['contrasted'])
            return

        # Need integers for the cv function parameters.
        morph_shape = const.CV_MORPH_SHAPE[self.cbox_val['morphshape'].get()]
        morph_op = const.CV_MORPHOP[self.cbox_val['morphop'].get()]

        # See: https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/filtering.html
        #  on page, see: cv2.getStructuringElement(shape, ksize[, anchor])
        # see: https://docs.opencv2.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        element = cv2.getStructuringElement(
            shape=morph_shape,
            ksize=(noise_k, noise_k))

        # Use morphologyEx as a shortcut for erosion followed by dilation.
        # Read https://docs.opencv2.org/3.4/db/df6/tutorial_erosion_dilatation.html
        # https://theailearner.com/tag/cv-morphologyex/
        # The op argument from const.CV_MORPH_OP options:
        #   MORPH_OPEN is useful to remove noise and small features.
        #   MORPH_CLOSE is better for certain images, but generally is worse.
        #   MORPH_HITMISS helps to separate close objects by shrinking them.
        self.cvimg['reduced_noise'] = cv2.morphologyEx(
            src=self.cvimg['contrasted'],
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=cv2.BORDER_DEFAULT,
        )

        self.update_image(tkimg_name='reduced_noise',
                          cvimg_array=self.cvimg['reduced_noise'])

    def filter_image(self) -> None:
        """
        Applies a filter selection to blur the reduced noise image
        to prepare for threshold segmentation. Can also serve as a
        specialized noise reduction step.
        Called by preprocess().
        Calls update_image().

        Returns:
            None
        """

        filter_selected = self.cbox_val['filter_type'].get()
        border_type = cv2.BORDER_ISOLATED  # cv2.BORDER_REPLICATE #cv2.BORDER_DEFAULT
        noise_iter = self.slider_val['noise_iter'].get()

        _k = self.slider_val['filter_k'].get()

        # If filter kernel slider and noise iteration are both set to 0,
        # then proceed without filtering and use the contrasted image.
        if _k == 0 and noise_iter == 0:
            self.update_image(tkimg_name='filtered',
                              cvimg_array=self.cvimg['contrasted'])
            return

        # If filter kernel slider is set to 0, then proceed without
        # filtering and use the reduced noise image.
        if _k == 0:
            self.update_image(tkimg_name='filtered',
                              cvimg_array=self.cvimg['reduced_noise'])
            return

        # Need to filter the contrasted image when noise reduction is
        #  not applied.
        if noise_iter == 0:
            image2filter = self.cvimg['contrasted']
        else:
            image2filter = self.cvimg['reduced_noise']

        # cv2.GaussianBlur and cv2.medianBlur need to have odd kernels,
        #   but cv2.blur and cv2.bilateralFilter will shift image between
        #   even and odd kernels, so just make it odd for everything.
        # NOTE: don't allow a filter kernel value of 0 to be passed to
        #  cv2.bilateralFilter b/c it is too CPU intensive; a _k of zero
        #  results in a method return (above).
        filter_k = _k + 1 if _k % 2 == 0 else _k

        # Apply a filter to blur edges or image interior.
        # NOTE: filtered image dtype is uint8
        # Bilateral parameters:
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        # from doc: Sigma values: For simplicity, you can set the 2 sigma
        #  values to be the same. If they are small (< 10), the filter
        #  will not have much effect, whereas if they are large (> 150),
        #  they will have a very strong effect, making the image look "cartoonish".
        #  NOTE: The larger the sigma the greater the effect of kernel size d.
        #  NOTE: d=-1 or 0, is very CPU intensive.
        # Gaussian parameters:
        #  see: https://theailearner.com/2019/05/06/gaussian-blurring/
        #  see: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
        #  If only sigmaX is specified, sigmaY is taken as the same as sigmaX.
        #    If both are given as zeros, they are calculated from the kernel size.
        #    Gaussian blurring is highly effective in removing Gaussian noise
        #    from an image.
        if filter_selected == 'cv2.blur':
            self.cvimg['filtered'] = cv2.blur(
                src=image2filter,
                ksize=(filter_k, filter_k),
                borderType=border_type)
        elif filter_selected == 'cv2.bilateralFilter':
            self.cvimg['filtered'] = cv2.bilateralFilter(
                src=image2filter,
                d=filter_k,
                sigmaColor=100,
                sigmaSpace=100,
                borderType=border_type)
        elif filter_selected == 'cv2.GaussianBlur':
            self.cvimg['filtered'] = cv2.GaussianBlur(
                src=image2filter,
                ksize=(filter_k, filter_k),
                sigmaX=0,
                sigmaY=0,
                borderType=border_type)
        elif filter_selected == 'cv2.medianBlur':
            self.cvimg['filtered'] = cv2.medianBlur(
                src=image2filter,
                ksize=filter_k)

        self.update_image(tkimg_name='filtered',
                          cvimg_array=self.cvimg['filtered'])

    def th_and_dist_trans(self) -> None:
        """
        Produces a threshold image from the filtered image. This image
        is used for masking in watershed_segmentation(). It is separate
        here so that its display can be updated independently of running
        watershed_segmentation(). Called by preprocess().
        Returns:
            None
        """

        th_type: int = const.THRESH_TYPE[self.cbox_val['th_type'].get()]
        filter_k = self.slider_val['filter_k'].get()
        noise_iter = self.slider_val['noise_iter'].get()
        dt_type: int = const.DISTANCE_TRANS_TYPE[self.cbox_val['dt_type'].get()]
        mask_size = int(self.cbox_val['dt_mask_size'].get())

        # Note from doc: Currently, the Otsu's and Triangle methods
        #   are implemented only for 8-bit single-channel images.
        #   For other cv2.THRESH_*, thresh needs to be manually provided.
        # The thresh parameter is determined automatically (0 is placeholder).
        # Convert values above thresh to a maxval of 255, white.
        # Need to use type *_INVERSE for black-on-white images.

        if filter_k == 0 and noise_iter == 0:
            image2threshold = self.cvimg['contrasted']
        elif filter_k == 0:
            image2threshold = self.cvimg['reduced_noise']
        else:
            image2threshold = self.cvimg['filtered']

        _, self.cvimg['thresholded'] = cv2.threshold(
            src=image2threshold,
            thresh=0,
            maxval=255,
            type=th_type)

        # Calculate the distance transform of the objects' thresholds,
        #  by replacing each foreground (non-zero) element, with its
        #  shortest distance to the background (any zero-valued element).
        #  Returns a float64 ndarray.
        # Note that maskSize=0 calculates the precise mask size only for
        #   cv2.DIST_L2. cv2.DIST_L1 and cv2.DIST_C always use maskSize=3.
        self.cvimg['transformed']: np.ndarray = cv2.distanceTransform(
            src=self.cvimg['thresholded'],
            distanceType=dt_type,
            maskSize=mask_size)

        self.update_image(tkimg_name='thresholded',
                          cvimg_array=self.cvimg['thresholded'])
        # self.update_image(tkimg_name='transformed',
        #                   cvimg_array=np.uint8(self.cvimg['transformed']))

    def make_labeled_array(self) -> np.ndarray:
        """
        Finds peak local maximum as defined in skimage.feature. The
        array is used as the 'markers' or 'labels' arguments in
        segmentation methods.
        Called as process() argument for watershed_segmentation() or
        randomwalk_segmentation().

        Returns: A labeled ndarray to use in one of the segmentation
                 algorithms.
        """
        min_dist: int = self.slider_val['plm_mindist'].get()
        p_kernel: tuple = (self.slider_val['plm_footprint'].get(),
                           self.slider_val['plm_footprint'].get())
        plm_kernel = np.ones(shape=p_kernel, dtype=np.uint8)

        # Generate the markers as local maxima of the distance to the background.
        # see: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
        # Generate the markers as local maxima of the distance to the background.
        # Don't use exclude_border; objects touching image border will be excluded
        #   in ViewImage.select_and_size_objects().
        local_max: ndimage = peak_local_max(image=self.cvimg['transformed'],
                                            min_distance=min_dist,
                                            exclude_border=False,  # True is min_dist
                                            num_peaks=np.inf,
                                            footprint=plm_kernel,
                                            labels=self.cvimg['thresholded'],
                                            num_peaks_per_label=np.inf,
                                            p_norm=np.inf)  # Chebyshev distance
        # p_norm=2,  # Euclidean distance

        mask = np.zeros(shape=self.cvimg['transformed'].shape, dtype=bool)
        # Set background to True (not zero: True or 1)
        mask[tuple(local_max.T)] = True

        # Note that markers are single px, colored in grayscale by their label index.
        labeled_array, self.num_dt_segments = ndimage.label(input=mask)

        # Source: http://scipy-lectures.org/packages/scikit-image/index.html
        # From the doc: labels: array of ints, of same shape as data without channels dimension.
        #  Array of seed markers labeled with different positive integers for
        #  different phases. Zero-labeled pixels are unlabeled pixels.
        #  Negative labels correspond to inactive pixels that are not taken into
        #  account (they are removed from the graph).

        # Replace thresh_img background with -1 to ignore those pixels.
        labeled_array[labeled_array == self.cvimg['thresholded']] = -1

        return labeled_array

    def watershed_segmentation(self, labeled_array: np.ndarray) -> None:
        """
        Segment objects with skimage.segmentation.watershed().
        Argument *array* calls the make_labeled_array() method that
        returns a labeled array.
        Called from process().

        Args:
            labeled_array: A skimage.features.peak_local_max array,
                           e.g., from make_labeled_array().

        Returns: None
        """

        ws_connectivity = int(self.cbox_val['ws_connectivity'].get())  # 1, 4 or 8.

        # Note that the minus symbol with distances_img converts distance
        #  transform into a threshold. Watershed can work without the
        #  conversion, but does a better job identifying segments with it.
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_compact_watershed.html
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
        # Need watershed_line to show boundaries on displayed watershed contour_pointset.
        # compactness=1.0 based on: DOI:10.1109/ICPR.2014.181
        self.cvimg['segmented_objects']: np.ndarray = watershed(
            image=-self.cvimg['transformed'],
            markers=labeled_array,
            connectivity=ws_connectivity,
            mask=self.cvimg['thresholded'],
            compactness=1.0,
            watershed_line=True)

    def draw_ws_segments(self) -> None:
        """
        Find and draw contours for watershed basin segments.
        Called from process() with a watershed_segmentation() arg.
        Calls update_image().

        Returns: None
        """

        # self.ws_basins is used in select_and_size_objects() to draw enclosing circles.
        # Convert image array from int32 to uint8 data type to find contour_pointset.
        # Conversion with cv2.convertScaleAbs(watershed_img) also works.
        # NOTE: Use method=cv2.CHAIN_APPROX_NONE when masking individual segments
        #   in select_and_export_objects(). CHAIN_APPROX_NONE can work, but NONE is best?
        self.ws_basins, _ = cv2.findContours(image=np.uint8(self.cvimg['segmented_objects']),
                                             mode=cv2.RETR_EXTERNAL,
                                             method=cv2.CHAIN_APPROX_NONE)

        # Convert watershed array data from int32 to allow colored contours.
        self.cvimg['Watershed'] = cv2.cvtColor(src=np.uint8(self.cvimg['segmented_objects']),
                                               code=cv2.COLOR_GRAY2BGR)

        # Need to prevent a thickness value of 0, yet have it be a function
        #  of image size so that it looks good in scaled display. Because the
        #  watershed_img has a black background, the contour lines are
        #  easier to see and look better if they are thinner than in the
        #  annotated 'sized' image where metrics['line_thickness'] is used.
        #  When user changes line thickness with + & - keys, only the 'sized'
        #  image updates; the watershed image displays the original thickness.
        if self.metrics['line_thickness'] == 1:
            line_thickness = 1
        else:
            line_thickness = self.metrics['line_thickness'] // 2

        # Need to prevent black contours because they won't show on the
        #  black background of the watershed_img image.
        if self.cbox_val['color'].get() == 'black':
            line_color = const.COLORS_CV['blue']
        else:
            line_color = const.COLORS_CV[self.cbox_val['color'].get()]

        cv2.drawContours(image=self.cvimg['Watershed'],
                         contours=self.ws_basins,
                         contourIdx=-1,  # do all contours
                         color=line_color,
                         thickness=line_thickness,
                         lineType=cv2.LINE_AA)

        self.update_image(tkimg_name='segmented_objects',
                          cvimg_array=self.cvimg['Watershed'])

        # Now need to draw enclosing circles around watershed segments and
        #  annotate with object sizes in ViewImage.select_and_size_objects().

    def randomwalk_segmentation(self, labeled_array: np.ndarray) -> None:
        """
        Segment objects with skimage.segmentation.random_walker().
        Argument *array* calls the make_labeled_array() method that
        returns a labeled array.
        Called from process().

        Args:
            labeled_array: A skimage.features.peak_local_max array,
                           e.g., from make_labeled_array().

        Returns: None
        """

        # Note that cvimg['segmented_objects'] is used for both watershed
        #  and random walker images because both share the Label() grid for
        #  img_label['segmented_objects'] in the tkimg_window['transformed'] window.
        # NOTE: beta and tolerances were empirically determined during development
        #  for best performance with sample images run on an Intel i9600k @ 4.8 GHz.
        #  Default beta & tol values take ~8x longer to process for similar results.
        # Need pyamg installed with mode='cg_mg'. 'cg_j' gives bad results.
        self.cvimg['segmented_objects']: np.ndarray = random_walker(
            data=self.cvimg['thresholded'],
            labels=labeled_array,
            beta=5,  # default: 130,
            mode='cg_mg',  # default: 'cg_j'
            tol=0.1,  # default: 1.e-3
            copy=True,
            return_full_prob=False,
            spacing=None,
            prob_tol=0.1,  # default: 1.e-3
            channel_axis=None)

        # self.rw_contours is used in select_and_size_objects() to draw
        #   enclosing circles and calculate sizes of segmented objects.
        # Note: This for loop is much more stable, and in most cases faster,
        #  than using parallelization modules (parallel.py and pool-worker.py
        #  in utility_modules).
        self.rw_contours.clear()
        for label in np.unique(ar=self.cvimg['segmented_objects']):

            # If the label is zero, we are examining the 'background',
            #   so simply ignore it.
            if label == 0:
                continue

            # ...otherwise, allocate memory for the label region and draw
            #   it on the mask.
            mask = np.zeros(shape=self.cvimg['segmented_objects'].shape, dtype="uint8")
            mask[self.cvimg['segmented_objects'] == label] = 255

            # Detect contours in the mask, grab the largest, and add it
            #   to the list used to draw, size, export, etc. the RW ROIs.
            contours, _ = cv2.findContours(image=mask.copy(),
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            self.rw_contours.append(max(contours, key=cv2.contourArea))

    def draw_rw_segments(self) -> None:
        """
        Draw and display the random walker segments from
        random_walker_segmentation().
        Called from process_rw_and_sizes().
        Calls update_image().

        Returns: None
        """

        # Convert array data from int32 to allow colored contours.
        self.cvimg['Random Walker'] = cv2.cvtColor(src=np.uint8(self.cvimg['segmented_objects']),
                                                   code=cv2.COLOR_GRAY2BGR)

        # Need to prevent white or black contours because they
        #  won't show on the white background with black segments.
        if self.cbox_val['color'].get() in 'white, black':
            line_color: tuple = const.COLORS_CV['blue']
        else:
            line_color = const.COLORS_CV[self.cbox_val['color'].get()]

        # Note: this does not update until process_rw_and_sizes() is called.
        #  It shares a window with the distance transform image, which
        #  updates with slider or combobox preprocessing changes.
        cv2.drawContours(image=self.cvimg['Random Walker'],
                         contours=self.rw_contours,
                         contourIdx=-1,  # do all contours
                         color=line_color,
                         thickness=self.metrics['line_thickness'],
                         lineType=cv2.LINE_AA)

        self.update_image(tkimg_name='segmented_objects',
                          cvimg_array=self.cvimg['Random Walker'])

        # Now need to draw enclosing circles around RW segments and
        #  annotate with object sizes in ViewImage.select_and_size_objects().


class ViewImage(ProcessImage):
    """
    A suite of methods to display cv segments based on selected settings
    and parameters that are in ProcessImage() methods.
    Methods:
        open_input
        set_auto_scale_factor
        import_settings
        delay_size_std_info_msg
        show_info_message
        configure_circle_r_sliders
        _on_click_save_img
        widget_control
        validate_px_size_entry
        validate_custom_size_entry
        set_size_standard
        select_and_size_objects
        select_and_export_objects
        report_results
        preprocess
        process
        process_sizes
    """

    def __init__(self):
        super().__init__()

        self.first_run: bool = True

        self.report_frame = tk.Frame()
        self.selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        # Note: The matching control variable attributes for the
        #   following selector widgets are in ProcessImage __init__.
        self.slider = {
            'alpha': tk.Scale(master=self.selectors_frame),
            'alpha_lbl': tk.Label(master=self.selectors_frame),

            'beta': tk.Scale(master=self.selectors_frame),
            'beta_lbl': tk.Label(master=self.selectors_frame),

            'noise_k': tk.Scale(master=self.selectors_frame),
            'noise_k_lbl': tk.Label(master=self.selectors_frame),

            'noise_iter': tk.Scale(master=self.selectors_frame),
            'noise_iter_lbl': tk.Label(master=self.selectors_frame),

            'filter_k': tk.Scale(master=self.selectors_frame),
            'filter_k_lbl': tk.Label(master=self.selectors_frame),

            'plm_mindist': tk.Scale(master=self.selectors_frame),
            'plm_mindist_lbl': tk.Label(master=self.selectors_frame),

            'plm_footprint': tk.Scale(master=self.selectors_frame),
            'plm_footprint_lbl': tk.Label(master=self.selectors_frame),

            'circle_r_min': tk.Scale(master=self.selectors_frame),
            'circle_r_min_lbl': tk.Label(master=self.selectors_frame),

            'circle_r_max': tk.Scale(master=self.selectors_frame),
            'circle_r_max_lbl': tk.Label(master=self.selectors_frame),
        }

        self.cbox = {
            'morphop': ttk.Combobox(master=self.selectors_frame),
            'morphop_lbl': tk.Label(master=self.selectors_frame),

            'morphshape': ttk.Combobox(master=self.selectors_frame),
            'morphshape_lbl': tk.Label(master=self.selectors_frame),

            'filter_type': ttk.Combobox(master=self.selectors_frame),
            'filter_lbl': tk.Label(master=self.selectors_frame),

            'th_type': ttk.Combobox(master=self.selectors_frame),
            'th_type_lbl': tk.Label(master=self.selectors_frame),

            'dt_type': ttk.Combobox(master=self.selectors_frame),
            'dt_type_lbl': tk.Label(master=self.selectors_frame),

            'dt_mask_size': ttk.Combobox(master=self.selectors_frame),
            'dt_mask_size_lbl': tk.Label(master=self.selectors_frame),

            'ws_connectivity': ttk.Combobox(master=self.selectors_frame),
            'ws_connectivity_lbl': tk.Label(master=self.selectors_frame),

            'size_std_lbl': tk.Label(master=self.selectors_frame),
            'size_std': ttk.Combobox(master=self.selectors_frame),
        }

        # User-entered pixel diameters of selected size standards.
        self.size_std = {
            'px_entry': tk.Entry(master=self.selectors_frame),
            'px_val': tk.StringVar(master=self.selectors_frame),
            'px_lbl': tk.Label(master=self.selectors_frame),
            'custom_entry': tk.Entry(master=self.selectors_frame),
            'custom_val': tk.StringVar(master=self.selectors_frame),
            'custom_lbl': tk.Label(master=self.selectors_frame),
        }

        self.button = {
            'process_ws': ttk.Button(master=self),
            'process_rw': ttk.Button(master=self),
            'save_results': ttk.Button(master=self),
            'new_input': ttk.Button(master=self),
            'export_objects': ttk.Button(master=self),
            'export_settings': ttk.Button(master=self),
            'reset': ttk.Button(master=self),
        }

        # Screen pixel width is defined in set_auto_scale_factor().
        self.screen_width: int = 0

        # Defined in setup_start_window() Radiobuttons.
        self.do_inverse_th = tk.BooleanVar()

        self.input_file: str = ''

        # Flag for user's choice of segment export type.
        self.export_segment: bool = True
        self.export_hull: bool = False

        # Used to reset values that user may have tried to change during
        #  prolonged processing times.
        self.slider_values: list = []

        self.report_txt: str = ''
        self.imported_settings: dict = {}
        self.use_saved_settings: bool = False

        # Info label is gridded in configure_main_window().
        self.info_txt = tk.StringVar()
        self.info_label = tk.Label(master=self, textvariable=self.info_txt)

        self.seg_algorithm: str = ''

    def open_input(self, toplevel) -> bool:
        """
        Provides an open file dialog to select an initial or new input
        image file. Also sets a scale slider value for the displayed img.
        Called from setup_start_window() or "New input" button.
        Args:
            toplevel: The Toplevel window over which to place the dialog.

        Returns:
            True or False depending on whether input was selected.

        """
        self.input_file = filedialog.askopenfilename(
            parent=toplevel,
            title='Select input image',
            filetypes=[('JPG', '*.jpg'),
                       ('JPG', '*.jpeg'),
                       ('JPG', '*.JPG'),  # used for iPhone images
                       ('PNG', '*.png'),
                       ('TIFF', '*.tiff'),
                       ('TIFF', '*.tif'),
                       ('All', '*.*')],
        )

        # When user selects an input, open it, and proceed, but if
        #  user selects "Cancel", then quit when in start window, otherwise
        #  simply close the filedialog (default action) because it was
        #  called from the "New input" button in the mainloop (self) window.
        if self.input_file:
            self.cvimg['input'] = cv2.imread(self.input_file)
            self.cvimg['gray'] = cv2.cvtColor(src=self.cvimg['input'],
                                              code=cv2.COLOR_RGBA2GRAY)
        elif toplevel != self:
            # Was called from start window and user clicked "Cancel".
            utils.quit_gui(mainloop=self)
        else:
            # Was called from report window and user clicked "Cancel".
            return False

        # Auto-set images' scale factor based on input image size.
        #  Can be later reset with keybindings in bind_scale_adjustment().
        #  circle_r_slider ranges are a function of input image size.
        self.metrics = manage.input_metrics(img=self.cvimg['input'])
        self.set_auto_scale_factor()
        self.configure_circle_r_sliders()

        # Handle condition for when user selects to use saved settings.
        input_path = Path(self.input_file).parent
        settings_path = Path(input_path / const.SETTINGS_FILE_NAME)
        if settings_path.exists():
            if self.first_run:
                msg = (f'Yes, use settings file in the folder: {input_path.parts[-1]}.\n'
                       'No, use default settings.')
            else:
                msg = (f'Yes, use settings file in folder: {input_path.parts[-1]}.\n'
                       'No, use current settings.')

            self.use_saved_settings = messagebox.askyesno(
                title=f"Use saved settings with {Path(self.input_file).name}?",
                detail=msg)

        if self.use_saved_settings:
            self.import_settings()

        return True

    def set_auto_scale_factor(self) -> None:
        """
        As a convenience for user, set a default scale factor to that
        needed for images to fit easily on the screen, either 1/3
        screen px width or 2/3 screen px height, depending
        on input image orientation.

        Returns: None
        """

        # Note that the scale factor is not included in saved_settings.json.
        _y, _x = self.metrics['gray_img'].shape
        if _x >= _y:
            estimated_scale = round((self.screen_width * 0.33) / _x, 2)
        else:
            estimated_scale = round((self.winfo_screenheight() * 0.66) / _y, 2)

        self.scale_factor.set(estimated_scale)

    def import_settings(self) -> None:
        """
        The dictionary of saved settings, imported via json.loads(),
        that are to be applied to a new image. Includes all settings
        except the scale_factor for window image size.
        """

        input_folder = Path(self.input_file).parent
        settings_file = Path(input_folder / const.SETTINGS_FILE_NAME)

        try:
            with open(settings_file, mode='rt', encoding='utf-8') as _fp:
                settings_json = _fp.read()
                self.imported_settings: dict = loads(settings_json)
        except FileNotFoundError as fnf:
            print('The settings JSON file could not be found.\n'
                  f'{fnf}')
        except OSError as oserr:
            print('There was a problem reading the settings JSON file.\n'
                  f'{oserr}')

        # Set/Reset Scale widgets.
        for _k in self.slider_val:
            self.slider_val[_k].set(self.imported_settings[_k])

        # Set/Reset Combobox widgets.
        for _k in self.cbox_val:
            self.cbox_val[_k].set(self.imported_settings[_k])

        # Set the threshold selection Radiobutton in setup_start_window().
        self.do_inverse_th.set(self.imported_settings['do_inverse_th'])

        self.metrics['font_scale'] = self.imported_settings['font_scale']
        self.metrics['line_thickness'] = self.imported_settings['line_thickness']

        self.size_std['px_val'].set(self.imported_settings['px_val'])
        self.size_std['custom_val'].set(self.imported_settings['custom_val'])

        self.seg_algorithm = self.imported_settings['seg_algorithm']

    def delay_size_std_info_msg(self) -> None:
        """
        When no size standard values ar entered, after a few seconds,
        display the size standard instructions in the mainloop (app)
        window. Internal function calls show_info_message().
        Called from process(), process_sizes(), and
        configure_buttons._new_input().

        Returns: None
        """

        def _show_msg() -> None:
            _info = ('\nWhen the entered pixel size is 1 AND selected size standard\n'
                     'is "None", then the size units displayed size are pixels.\n'
                     'Size units are millimeters for any pre-set size standard.\n'
                     f'(Processing time elapsed: {self.elapsed})\n')

            self.show_info_message(info=_info, color='black')

        if (self.size_std['px_val'].get() == '1' and
                self.cbox_val['size_std'].get() == 'None'):
            app.after(ms=5555, func=_show_msg)

    def show_info_message(self, info: str, color: str) -> None:
        """
        Configure for display and update the informational message in
        the report and settings window.
        Args:
            info: The text string of the message to display.
            color: The font color string, either as a key in the
                   const.COLORS_TK dictionary or as a Tk compatible fg
                   color string code, i.e. hex code.

        Returns:
            None
        """
        self.info_txt.set(info)

        # Need to handle case when color is defined as a dictionary key
        #  or as a hex code.
        try:
            tk_color = const.COLORS_TK[color]
        except KeyError:
            tk_color = color

        self.info_label.config(fg=tk_color)

    def configure_circle_r_sliders(self) -> None:
        """
        Called from config_sliders() and open_input().
        Returns:

        """

        # Note: this widget configuration method is here, instead of in
        #  SetupApp() b/c it is called from open_input() as well as from
        #  config_sliders().

        # Note: may need to adjust circle_r_min scaling with image size b/c
        #  large contours cannot be selected if circle_r_max is too small.
        min_circle_r = self.metrics['max_circle_r'] // 6
        max_circle_r = self.metrics['max_circle_r']

        self.slider['circle_r_min'].configure(
            from_=1, to=min_circle_r,
            tickinterval=min_circle_r / 10,
            variable=self.slider_val['circle_r_min'],
            **const.SCALE_PARAMETERS)
        self.slider['circle_r_max'].configure(
            from_=1, to=max_circle_r,
            tickinterval=max_circle_r / 10,
            variable=self.slider_val['circle_r_max'],
            **const.SCALE_PARAMETERS)

    def _on_click_save_img(self, image_name: str) -> None:
        """
        Save a window image (Label) that was rt-clicked.
        Called only from display_windows() for keybindings.

        Args:
            image_name: The key name (string) used in the img_label
                        dictionary.
        Returns:
            None
        """
        tkimg = self.tkimg[image_name]

        click_info = (f'The displayed {image_name} image was saved at'
                      f' {self.scale_factor.get()} scale.')

        utils.save_report_and_img(input_path=self.input_file,
                                  img2save=tkimg,
                                  txt2save=click_info,
                                  caller=image_name)

        # Provide user with a notice that a file was created and
        #  give user time to read the message before resetting it.
        input_folder = Path(self.input_file).parent
        _info = (f'\nThe result image, "{image_name}", was saved to:\n'
                 f'{input_folder}\n'
                 f'with a timestamp, at a scale of {self.scale_factor.get()}.\n\n')
        self.show_info_message(info=_info, color='black')

    def widget_control(self, action: str) -> None:
        """
        Used to disable settings widgets when segmentation is running.
        Provides a watch cursor while widgets are disabled.
        Gets Scale() values at time of disabling and resets them upon
        enabling, thus preventing user click events retained in memory
        during processing from changing slider position post-processing.

        Args:
            action: Either 'off' to disable widgets, or 'on' to enable.
        """
        if action == 'off':
            for _name, _w in self.slider.items():
                _w.configure(state=tk.DISABLED)

                # Grab the current slider values, in case user tries to change.
                if isinstance(_w, tk.Scale):
                    self.slider_values.append(self.slider_val[_name].get())
            for _, _w in self.cbox.items():
                _w.configure(state=tk.DISABLED)
            for _, _w in self.button.items():
                _w.grid_remove()
            for _, _w in self.size_std.items():
                if not isinstance(_w, tk.StringVar):
                    _w.configure(state=tk.DISABLED)

            self.config(cursor='watch')
            self.update()
        else:  # is 'on'
            idx = 0
            for _name, _w in self.slider.items():
                _w.configure(state=tk.NORMAL)

                # Restore the slider values to overwrite any changes.
                if self.slider_values and isinstance(_w, tk.Scale):
                    self.slider_val[_name].set(self.slider_values[idx])
                    idx += 1
            for _, _w in self.cbox.items():
                if isinstance(_w, tk.Label):
                    _w.configure(state=tk.NORMAL)
                else:  # is tk.Combobox
                    _w.configure(state='readonly')
            for _, _w in self.button.items():
                _w.grid()
            for _, _w in self.size_std.items():
                if not isinstance(_w, tk.StringVar):
                    _w.configure(state=tk.NORMAL)

            self.config(cursor='')
            self.update()
            self.slider_values.clear()

    def validate_px_size_entry(self) -> None:
        """
        Check whether pixel size Entry() value is a positive integer.
        Post a message if the entry is not valid.
        Calls widget_control().
        """

        size_std_px: str = self.size_std['px_val'].get()
        try:
            int(size_std_px)
            if int(size_std_px) <= 0:
                raise ValueError
        except ValueError:

            # Need widget_control to prevent runaway sliders, if clicked.
            self.widget_control('off')
            _post = ('Enter only integers > 0 for the pixel diameter.\n'
                     f'{size_std_px} was entered. Defaulting to 1.')
            messagebox.showerror(title='Invalid entry',
                                 detail=_post)
            self.size_std['px_val'].set('1')
            self.widget_control('on')

    def validate_custom_size_entry(self) -> None:
        """
        Check whether custom size Entry() value is a real number.
        Post a message if the entry is not valid.
        Calls widget_control().
        """
        custom_size: str = self.size_std['custom_val'].get()
        size_std_px: str = self.size_std['px_val'].get()

        # Verify that entries are positive numbers. Define self.num_sigfig.
        # Custom sizes can be entered as integer, float, or power operator.
        # Number of significant figures is the lowest of that for the
        #  standard's size value or pixel diameter. Therefore, lo-res input
        #  are more likely to have size std diameters of <100 px, thus
        #  limiting calculated sizes to 2 sigfig.
        try:
            float(custom_size)  # will raise ValueError if not a number.
            if float(custom_size) <= 0:
                raise ValueError

            self.unit_per_px.set(float(custom_size) / int(size_std_px))

            if size_std_px == '1':
                self.num_sigfig = utils.count_sig_fig(custom_size)
            else:
                self.num_sigfig = min(utils.count_sig_fig(custom_size),
                                      utils.count_sig_fig(size_std_px))
        except ValueError:

            # Need widget_control to prevent runaway sliders, if clicked.
            self.widget_control('off')
            messagebox.showinfo(
                title='Custom size',
                detail='Enter a number > 0.\n'
                       'Accepted types:\n'
                       '  integer: 26, 2651, 2_651\n'
                       '  decimal: 26.5, 0.265, .2\n'
                       '  exponent: 2.6e10, 2.6e-2')
            self.size_std['custom_val'].set('0.0')
            self.widget_control('on')

    def set_size_standard(self) -> None:
        """
        Assign a unit conversion factor to the observed pixel diameter
        of the chosen size standard and calculate the number of
        significant figures for preset or custom size entries.
        Called from process(), process_sizes(), __init__.

        Returns: None
        """

        self.validate_px_size_entry()
        size_std_px: str = self.size_std['px_val'].get()
        size_std: str = self.cbox_val['size_std'].get()
        preset_std_size: float = const.SIZE_STANDARDS[size_std]

        # For clarity, need to show the custom size Entry widget only
        #  when 'Custom' is selected.
        # Verify that entries are numbers and define self.num_sigfig.
        #  Custom sizes can be entered as integer, float, or power operator.
        # Number of significant figures is the lowest of that for the
        #  standard's size value or pixel diameter.
        if size_std == 'Custom':
            self.size_std['custom_entry'].grid()
            self.size_std['custom_lbl'].grid()
            self.validate_custom_size_entry()

        else:  # is one of the preset size standards
            self.size_std['custom_entry'].grid_remove()
            self.size_std['custom_lbl'].grid_remove()
            self.size_std['custom_val'].set('0.0')

            self.unit_per_px.set(preset_std_size / int(size_std_px))

            if size_std_px == '1':
                self.num_sigfig = utils.count_sig_fig(preset_std_size)
            else:
                self.num_sigfig = min(utils.count_sig_fig(preset_std_size),
                                      utils.count_sig_fig(size_std_px))

    def select_and_size_objects(self, contour_pointset: list) -> None:
        """
        Select object contour ROI based on area size and position,
        draw an enclosing circle around contours, then display them
        on the input image. Objects are expected to be oblong so that
        circle diameter can represent the object's length.
        Called by process(), process_sizes(), bind_annotation_styles().
        Calls update_image().

        Args:
            contour_pointset: A list of contour coordinates generated by
                              one of the segmentation algorithms. e.g.,
                              self.ws_basins or self.rw_contours.
        Returns:
            None
        """

        self.cvimg['sized'] = self.cvimg['input'].copy()

        selected_sizes: List[float] = []
        preferred_color: tuple = const.COLORS_CV[self.cbox_val['color'].get()]
        font_scale: float = self.metrics['font_scale']
        line_thickness: int = self.metrics['line_thickness']

        # The size range slider values are radii pixels. This is done b/c:
        #  1) Displayed values have fewer digits, so a cleaner slide bar.
        #  2) Sizes are diameters, so radii are conceptually easier than areas.
        #  So, need to convert to area for the cv2.contourArea function.
        c_area_min = self.slider_val['circle_r_min'].get() ** 2 * np.pi
        c_area_max = self.slider_val['circle_r_max'].get() ** 2 * np.pi

        # Set limits for coordinate points to identify contours that
        # are within 1 px of an image file border (edge).
        bottom_edge = self.cvimg['gray'].shape[0] - 1
        right_edge = self.cvimg['gray'].shape[1] - 1

        if not contour_pointset:
            utils.no_objects_found_msg()
            return

        flag = False
        for _c in contour_pointset:

            # Exclude None elements.
            # Exclude contours not in the specified size range.
            # Exclude contours that have a coordinate point intersecting an img edge.
            #  ... those that touch top or left edge or are background.
            #  ... those that touch bottom or right edge.
            if _c is None:
                continue
            if not c_area_max > cv2.contourArea(_c) >= c_area_min:
                continue
            if {0, 1}.intersection(set(_c.ravel())):
                continue
            # Break from inner loop when either edge touch is found.
            for _p in _c:
                for coord in _p:
                    _x, _y = tuple(coord)
                    if _x == right_edge or _y == bottom_edge:
                        flag = True
                if flag:
                    break
            if flag:
                flag = False
                continue

            # Draw a circle enclosing the contour, measure its diameter,
            #  and save each object_size measurement to the selected_sizes
            #  list for reporting.
            ((_x, _y), _r) = cv2.minEnclosingCircle(_c)

            # Note: sizes are full-length floats.
            object_size: float = _r * 2 * self.unit_per_px.get()

            # Need to set sig. fig. to display sizes in annotated image.
            #  num_sigfig value is determined in set_size_standard().
            size2display: str = to_p.to_precision(value=object_size,
                                                  precision=self.num_sigfig)

            # Need to have pixel diameters as integers. Because...
            #  When num_sigfig is 4, as is case for None:'1.001' in
            #  const.SIZE_STANDARDS, then for px_val==1, with lower
            #  sig.fig., objects <1000 px diameter would display as
            #  decimal fractions. So, round()...
            if (self.size_std['px_val'].get() == '1' and
                    self.cbox_val['size_std'].get() == 'None'):
                size2display = str(round(float(size2display)))

            # Convert size strings to float, assuming that individual
            #  sizes listed in the report may be used in a spreadsheet
            #  or for other statistical analysis.
            selected_sizes.append(float(size2display))

            # Need to properly center text in the circled object.
            ((txt_width, _), baseline) = cv2.getTextSize(
                text=size2display,
                fontFace=const.FONT_TYPE,
                fontScale=font_scale,
                thickness=line_thickness)
            offset_x = txt_width / 2

            cv2.circle(img=self.cvimg['sized'],
                       center=(round(_x), round(_y)),
                       radius=round(_r),
                       color=preferred_color,
                       thickness=line_thickness,
                       lineType=cv2.LINE_AA,
                       )
            cv2.putText(img=self.cvimg['sized'],
                        text=size2display,
                        org=(round(_x - offset_x), round(_y + baseline)),
                        fontFace=const.FONT_TYPE,
                        fontScale=font_scale,
                        color=preferred_color,
                        thickness=line_thickness,
                        lineType=cv2.LINE_AA,
                        )

        # The sorted size list is used for reporting individual sizes
        #   and size summary metrics.
        if selected_sizes:
            self.sorted_size_list = sorted(selected_sizes)
        else:
            utils.no_objects_found_msg()

        self.update_image(tkimg_name='sized',
                          cvimg_array=self.cvimg['sized'])

    def select_and_export_objects(self) -> int:
        """
        Takes a list of contour segments, selects, masks and extracts
        each, to a bounding rectangle, for export of ROI to file.
        Calls utility_modules/utils.export_segments().
        Called from Button command in configure_buttons().

        Returns: Integer count of exported segments.
        """

        # Evaluate user's messagebox askyesnocancel answer, from configure_buttons().
        if self.export_segment:
            # Export masked selected object segments.
            export_this = 'result'
        elif self.export_segment is False:
            # Export enlarged bounding rectangles around segments.
            export_this = 'roi'
        else:  # user selected 'Cancel', which returns None, the default.
            return 0

        # Grab current time to pass to utils.export_segments() module.
        #  This is done here, outside the for loop, to avoid having the
        #  export timestamp change (by one or two seconds) during processing.
        # The index count is also passed as a export_segments() argument.
        time_now = datetime.now().strftime('%Y%m%d%I%M%S')
        roi_idx = 0

        # Use the identical selection criteria as in select_and_size_objects().
        c_area_min = self.slider_val['circle_r_min'].get() ** 2 * np.pi
        c_area_max = self.slider_val['circle_r_max'].get() ** 2 * np.pi
        bottom_edge = self.cvimg['gray'].shape[0] - 1
        right_edge = self.cvimg['gray'].shape[1] - 1
        flag = False

        if self.seg_algorithm == 'Watershed':
            contour_pointset = self.ws_basins
        else:  # is 'Random Walker'
            contour_pointset = self.rw_contours

        for _c in contour_pointset:

            # As in select_and_size_objects():
            #  Exclude None elements.
            #  Exclude contours not in the specified size range.
            #  Exclude contours that have a coordinate point intersecting
            #   the img edge, that is...
            #   ...those that touch top or left edge or are background.
            #   ...those that touch bottom or right edge.
            if _c is None:
                return 0
            if not c_area_max > cv2.contourArea(_c) >= c_area_min:
                continue
            if {0, 1}.intersection(set(_c.ravel())):
                continue
            # Break from inner loop when either edge touch is found.
            for _p in _c:
                for coord in _p:
                    _x, _y = tuple(coord)
                    if _x == right_edge or _y == bottom_edge:
                        flag = True
                if flag:
                    break
            if flag:
                flag = False
                continue

            # Idea for segment extraction from:
            #  https://stackoverflow.com/questions/21104664/
            #   extract-all-bounding-boxes-using-opencv-python
            # The ROI slice encompasses the selected segment contour.
            _x, _y, _w, _h = cv2.boundingRect(_c)

            # Slightly expand the _c segment's ROI bounding box on the input image.
            y_slice = slice(_y - 4, (_y + _h + 3))
            x_slice = slice(_x - 4, (_x + _w + 3))
            roi = self.cvimg['input'][y_slice, x_slice]
            roi_idx += 1

            # Idea for masking from: https://stackoverflow.com/questions/70209433/
            #   opencv-creating-a-binary-mask-from-the-image
            # Need to use full input img b/c that is what _c pointset refers to.
            # Steps: Make a binary mask of segment on full input image.
            #        Crop the mask to a slim border around the segment.
            mask = np.zeros_like(cv2.cvtColor(src=self.cvimg['input'],
                                              code=cv2.COLOR_BGR2GRAY))
            roi_mask = mask[y_slice, x_slice]

            if self.export_hull:
                hull = cv2.convexHull(_c)
                chosen_contours = [hull]
            else:  # is False, user selected "No".
                chosen_contours = [_c]

            cv2.drawContours(image=mask,
                             contours=chosen_contours,
                             contourIdx=-1,
                             color=(255, 255, 255),
                             thickness=cv2.FILLED)

            # Note: this contour step provides a cleaner border around the segment.
            cv2.drawContours(image=mask,
                             contours=chosen_contours,
                             contourIdx=-1,
                             color=(0, 0, 0),
                             thickness=4)

            # Idea for extraction from: https://stackoverflow.com/questions/59432324/
            #  how-to-mask-image-with-binary-mask
            # Extract the segment from input to a black background, then if it's
            #  valid, convert black background to white and export it.
            result = cv2.bitwise_and(src1=roi, src2=roi, mask=roi_mask)

            if result is not None:
                result[roi_mask == 0] = 255

                if export_this == 'result':
                    # Export just the object segment.
                    export_chosen = result
                else:  # is 'roi', so export segment's enlarged bounding box.
                    export_chosen = roi

                utils.export_segments(input_path=self.input_file,
                                      img2exp=export_chosen,
                                      index=roi_idx,
                                      timestamp=time_now)
            else:
                print(f'There was a problem with segment # {roi_idx},'
                      ' so it was not exported.')

        return roi_idx

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from process(), process_sizes(), __init__.
        Returns:
            None
        """

        # Note: recall that *_val dictionaries are inherited from ProcessImage().
        px_h, px_w = self.cvimg['gray'].shape
        alpha: float = self.slider_val['alpha'].get()
        beta: int = self.slider_val['beta'].get()
        noise_iter: int = self.slider_val['noise_iter'].get()
        morph_op: str = self.cbox_val['morphop'].get()
        morph_shape: str = self.cbox_val['morphshape'].get()
        filter_type: str = self.cbox_val['filter_type'].get()
        th_type: str = self.cbox_val['th_type'].get()
        circle_r_min: int = self.slider_val['circle_r_min'].get()
        circle_r_max: int = self.slider_val['circle_r_max'].get()
        plm_mindist: int = self.slider_val['plm_mindist'].get()
        dt_type: str = self.cbox_val['dt_type'].get()
        mask_size: int = int(self.cbox_val['dt_mask_size'].get())
        plm_footprint: int = self.slider_val['plm_footprint'].get()
        p_kernel: tuple = (plm_footprint, plm_footprint)

        if self.seg_algorithm == 'Watershed':
            ws_connectivity: str = self.cbox_val['ws_connectivity'].get()
            compact_val = '1.0'  # NOTE: update if change val in watershed method.
        else:  # is 'Random Walker'
            ws_connectivity: str = 'n/a'
            compact_val = 'n/a'

        # Only odd kernel integers are used for processing.
        _nk: int = self.slider_val['noise_k'].get()
        if _nk == 0:
            noise_k = 'noise reduction not applied'
        else:
            noise_k = _nk + 1 if _nk % 2 == 0 else _nk

        _fk: int = self.slider_val['filter_k'].get()
        if _fk == 0:
            filter_k = 'kernel = 0; filter not applied'
        else:
            filter_k = _fk + 1 if _fk % 2 == 0 else _fk
            filter_k = f'({filter_k}, {filter_k})'

        size_std: str = self.cbox_val['size_std'].get()
        if size_std == 'Custom':
            size_std_size: str = self.size_std['custom_entry'].get()
        else:
            size_std_size: str = const.SIZE_STANDARDS[size_std]

        # Size units are millimeters for the preset size standards.
        unit = 'unknown' if size_std in 'None, Custom' else 'mm'

        # Work up some summary metrics with correct number of sig. fig.
        if self.sorted_size_list:
            num_selected: int = len(self.sorted_size_list)
            unit_per_px: str = to_p.to_precision(value=self.unit_per_px.get(),
                                                 precision=self.num_sigfig)
            mean_unit_dia: str = to_p.to_precision(value=mean(self.sorted_size_list),
                                                   precision=self.num_sigfig)
            median_unit_dia: str = to_p.to_precision(value=median(self.sorted_size_list),
                                                     precision=self.num_sigfig)
            smallest = to_p.to_precision(value=min(self.sorted_size_list),
                                         precision=self.num_sigfig)
            biggest = to_p.to_precision(value=max(self.sorted_size_list),
                                        precision=self.num_sigfig)
            size_range: str = f'{smallest}--{biggest}'
        else:
            num_selected = 0
            unit_per_px = 'n/a'
            mean_unit_dia = 'n/a'
            median_unit_dia = 'n/a'
            size_range = 'n/a'

        # Text is formatted for clarity in window, terminal, and saved file.
        # Divider symbol is Box Drawings Double Horizontal from https://coolsymbol.com/
        space = 23
        tab = " " * space
        divider = "═" * 20  # divider's unicode_escape: u'\u2550\'

        self.report_txt = (
            f'\nImage: {self.input_file}\n'
            f'Image size, pixels (w x h): {px_w}x{px_h}\n'
            f'{divider}\n'
            f'{"Contrast:".ljust(space)}convertScaleAbs alpha={alpha}, beta={beta}\n'
            f'{"Noise reduction:".ljust(space)}cv2.getStructuringElement ksize={noise_k},\n'
            f'{tab}cv2.getStructuringElement shape={morph_shape}\n'
            f'{tab}cv2.morphologyEx iterations={noise_iter}\n'
            f'{tab}cv2.morphologyEx op={morph_op},\n'
            f'{"Filter:".ljust(space)}{filter_type} ksize={filter_k}\n'
            f'{"cv2.threshold:".ljust(space)}type={th_type}\n'
            f'{"cv2.distanceTransform:".ljust(space)}'
            f'distanceType={dt_type}, maskSize={mask_size}\n'
            f'{"scimage functions:".ljust(space)}segmentation algorithm: {self.seg_algorithm}\n'
            f'{"   peak_local_max:".ljust(space)}'
            f'min_distance={plm_mindist}, footprint=np.ones({p_kernel})\n'
            f'{"   watershed params:".ljust(space)}'
            f'connectivity={ws_connectivity}, compactness={compact_val}\n'
            f'{divider}\n'
            f'{"# Selected objects:".ljust(space)}{num_selected},'
            f' out of {self.num_dt_segments} total segments\n'
            f'{"Selected size range:".ljust(space)}{circle_r_min}--{circle_r_max} pixels\n'
            f'{"Selected size std.:".ljust(space)}{size_std}, with a diameter of'
            f' {size_std_size} {unit} units.\n'
            f'{tab}Pixel diameter entered: {self.size_std["px_val"].get()},'
            f' unit/px factor: {unit_per_px}\n'
            f'{"Object size metrics,".ljust(space)}mean: {mean_unit_dia}, median:'
            f' {median_unit_dia}, range: {size_range}'
        )

        utils.display_report(frame=self.report_frame,
                             report=self.report_txt)

    def preprocess(self, event=None) -> None:
        """
        Run processing functions prior to watershed_segmentation() to
        allow calling them and updating their images independently of the
        lengthy processing time of watershed_segmentation().

        Args:
            event: Implicit widget event.

        Returns:
            *event* as a formality; functionally None.
        """

        self.widget_control('on')
        self.adjust_contrast()
        self.reduce_noise()
        self.filter_image()
        self.th_and_dist_trans()
        self.set_size_standard()
        self.report_results()

        # Var first_run is reset to False in process() at startup.
        if not self.first_run:
            _info = ('\nPreprocessing completed.\n'
                     'Select a "Run" button to process segments and object sizes\n\n\n')
            self.show_info_message(info=_info, color='blue')

            # This update is needed to re-scale the input when
            #  bind_scale_adjustment() keybinds are used. Other images
            #  are rescaled in their respective processing methods.
            self.update_image(tkimg_name='input',
                              cvimg_array=self.cvimg['input'])

        return event

    def process(self) -> None:
        """
        Runs image segmentation processing methods from ProcessImage(),
        plus methods for annotation style, sizing, and reporting.

        Returns: None
        """

        # Need to first check that entered size values are okay.
        if not self.first_run:
            self.validate_px_size_entry()
            if self.cbox_val['size_std'].get() == 'Custom':
                self.validate_custom_size_entry()

        self.info_label.config(fg=const.COLORS_TK['blue'])

        _info = '\n\nRunning segmentation algorithm...\n\n\n'
        self.show_info_message(info=_info, color='blue')

        self.widget_control('off')
        self.time_start: float = time()

        # Note that at startup, seg_algorithm is defined as 'Watershed':
        #  in set_defaults(); after that, it is defined by the "Run..." buttons.
        if self.seg_algorithm == 'Watershed':
            self.watershed_segmentation(self.make_labeled_array())
            self.draw_ws_segments()
            self.select_and_size_objects(contour_pointset=self.ws_basins)
        else:  # is 'Random Walker'
            self.randomwalk_segmentation(self.make_labeled_array())
            self.draw_rw_segments()
            self.select_and_size_objects(contour_pointset=self.rw_contours)

        # Record processing time; preprocessing time is negligible.
        self.elapsed = round(time() - self.time_start, 3)
        self.report_results()
        self.widget_control('on')

        # Here, at the end of the processing pipeline, is where the
        #  first_run flag is set to False.
        setting_type = 'Saved' if self.use_saved_settings else 'Default'
        if self.first_run:
            self.first_run = False
            self.use_saved_settings = False

            _info = (f'\nInitial processing time elapsed: {self.elapsed}\n'
                     f'{setting_type} settings were used.\n\n\n')
            self.show_info_message(info=_info, color='black')

        else:
            _info = (f'\n{self.seg_algorithm} segmentation completed and sizes calculated.\n'
                     f'{self.elapsed} processing seconds elapsed.\n\n\n')
            self.show_info_message(info=_info, color='blue')

        self.delay_size_std_info_msg()

    def process_sizes(self, caller: str) -> None:
        """
        Call only sizing and reporting methods to improve performance.
        Called from the circle_r_min and circle_r_max sliders.
        Calls set_size_standard(), select_and_size_objects(), report_results().

        Args:
            caller: An identifier for the event-bound widget, either
                    "circle_r" for the radius range sliders, or
                    "size_std" for size standard combobox, or,
                    "px_entry" and "custom_entry" for size std entries.

        Returns: None
        """
        self.set_size_standard()

        size_std: str = self.cbox_val['size_std'].get()
        size_std_px: str = self.size_std['px_val'].get()

        if size_std == 'None' and int(size_std_px) > 1:
            self.size_std['custom_entry'].grid_remove()
            self.size_std['custom_lbl'].grid_remove()
            self.size_std['custom_val'].set('0.0')
            _info = ('\nUsing a pixel size greater than 1 AND "None" for a'
                     ' size standard\n'
                     'will give wrong object sizes.\n\n\n')
            self.show_info_message(info=_info, color='vermilion')
        elif caller == 'circle_r':
            _info = ('\nNew size range for selected objects.\n'
                     'Counts may have changed.\n\n\n')
            self.show_info_message(info=_info, color='blue')
        elif caller in 'size_std, px_entry, custom_entry':
            _info = '\nNew object sizes calculated.\n\n\n\n'
            self.show_info_message(info=_info, color='blue')
        else:
            print('Oops, an unrecognized process_sizes() caller argument\n'
                  f'was used: caller={caller}')

        if self.seg_algorithm == 'Watershed':
            self.select_and_size_objects(contour_pointset=self.ws_basins)
        else:  # is 'Random Walker'
            self.select_and_size_objects(contour_pointset=self.rw_contours)

        self.report_results()

        # Display the size standard instructions only when no size
        #  standard values are entered.
        self.delay_size_std_info_msg()


class SetupApp(ViewImage):
    """
    The mainloop Class that configures windows and widgets.
    Methods:
        setup_main_window
        setup_start_window
        start_now
        _delete_window_message
        setup_image_windows
        configure_main_window
        _settings_dict
        configure_buttons
        _need_to_click
        config_sliders
        config_comboboxes
        config_entries
        _current_contours
        bind_annotation_styles
        bind_scale_adjustment
        set_defaults
        grid_widgets
        grid_img_labels
        display_windows
    """

    def __init__(self):
        super().__init__()

        # Dictionary items are populated in setup_image_windows(), with
        #   tk.Toplevel as values; don't want tk windows created here.
        self.tkimg_window: dict = {}
        self.window_title: dict = {}

        # This order of events, when coordinated with the calls in
        #  start_now(), allow macOS implementation to flow well.
        self.setup_main_window()
        self.setup_start_window()

    def setup_main_window(self):
        """
        For clarity, remove from view the Tk mainloop window created
        by the inherited ProcessImage() Class. But, to make window
        transitions smoother, first position it where it needs to go.
        Put this window toward the top right corner of the screen
        so that it doesn't cover up the img windows; also so that
        the bottom of the window is, hopefully, not below the bottom
        of the screen.

        Returns:
            None
        """

        # Deiconify in display_windows(), but hide for now.
        self.wm_withdraw()

        self.screen_width = self.winfo_screenwidth()

        # Make geometry offset a function of the screen width.
        #  This is needed b/c of the way different platforms' window
        #  managers position windows.
        w_offset = int(self.screen_width * 0.55)
        self.geometry(f'+{w_offset}+0')
        self.wm_minsize(width=450, height=450)

        # Need to provide exit info msg to Terminal.
        self.protocol(name='WM_DELETE_WINDOW',
                      func=lambda: utils.quit_gui(mainloop=self))

        self.bind('<Escape>', func=lambda _: utils.quit_gui(mainloop=self))
        self.bind('<Control-q>', func=lambda _: utils.quit_gui(mainloop=self))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

    def setup_start_window(self) -> None:
        """
        Set up a basic Toplevel, then prompt for an input file, then
        proceed with configuring the window's widgets to set initial
        parameters of display scale, font color, and threshold type
        (inverse vs. not). A button will then trigger image processing
        steps to begin. Window is destroyed once button is used.

        Returns:
            None
        """

        # Need style of the ttk.Button to match main window button style.
        manage.ttk_styles(mainloop=self)

        process_btn_txt = tk.StringVar(value='Process now')

        def _call_start(event=None) -> None:
            """Remove this start window, then call the suite of methods
            to get things going. For larger images, show new toplevel
            info msg and change process_now_button text to a wait msg.
            Called from process_now_button and Return/Enter keys.
            Args:
                event: The implicit key action event, when used.
            Returns:
                 *event* as a formality; is functionally None.
            """
            # Use a spinning cursor to indicate that something is happening
            #  because larger images may take a while to process and show.
            process_btn_txt.set('Processing started, wait...')
            start_win.config(cursor='watch')
            self.start_now()
            start_win.destroy()

            return event

        # Window basics:
        # Open with a temporary, instructional title.
        start_win = tk.Toplevel()
        start_win.title('First, select an input image file')
        start_win.wm_resizable(width=False, height=False)
        start_win.config(relief='raised',
                         bg=const.DARK_BG,
                         # bg=const.COLORS_TK['sky blue'],  # for development
                         highlightthickness=3,
                         highlightcolor=const.COLORS_TK['yellow'],
                         highlightbackground=const.DRAG_GRAY)
        start_win.columnconfigure(index=0, weight=1)
        start_win.columnconfigure(index=1, weight=1)

        # Need to allow complete tk mainloop shutdown from the system's
        #   window manager 'close' icon in the start window bar.
        start_win.protocol(name='WM_DELETE_WINDOW',
                           func=lambda: utils.quit_gui(mainloop=self))

        start_win.bind('<Escape>', lambda _: utils.quit_gui(mainloop=self))
        start_win.bind('<Control-q>', lambda _: utils.quit_gui(mainloop=self))
        start_win.bind('<Return>', func=_call_start)
        start_win.bind('<KP_Enter>', func=_call_start)

        # Window widgets:
        # Provide a placeholder window header for input file info.
        window_header = tk.Label(
            master=start_win,
            text='Image: waiting to be selected...\nSize: TBD',
            **const.LABEL_PARAMETERS)

        # Unicode arrow symbols: up \u2101, down \u2193
        if const.MY_OS == 'dar':
            color_help = '← Can change later with shift-control-↑ & ↓  '
            color_tip = 'shift-control-↑ & shift-control-↓'
        else:
            color_help = '← Can change later with Ctrl-↑ & Ctrl-↓  '
            color_tip = ' Ctrl-↑ & Ctrl-↓'

        color_label = tk.Label(master=start_win,
                               text='Annotation font color:',
                               **const.LABEL_PARAMETERS)
        color_msg_lbl = tk.Label(master=start_win,
                                 text=color_help,
                                 **const.LABEL_PARAMETERS)
        color_cbox = ttk.Combobox(master=start_win,
                                  values=list(const.COLORS_CV.keys()),
                                  textvariable=self.cbox_val['color'],
                                  width=11,
                                  height=14,
                                  **const.COMBO_PARAMETERS)
        color_cbox.current(0)  # blue

        inverse_label = tk.Label(master=start_win,
                                 text='Use INVERSE threshold type?',
                                 **const.LABEL_PARAMETERS)
        inverse_yes = tk.Radiobutton(master=start_win,
                                     text='Yes',
                                     value=True,
                                     variable=self.do_inverse_th,
                                     **const.RADIO_PARAMETERS)
        inverse_no = tk.Radiobutton(master=start_win,
                                    text='No',
                                    value=False,
                                    variable=self.do_inverse_th,
                                    **const.RADIO_PARAMETERS)
        inverse_no.select()

        process_now_button = ttk.Button(master=start_win,
                                        textvariable=process_btn_txt,
                                        style='My.TButton',
                                        width=0,
                                        command=_call_start)

        # Create menu instance and add pull-down menus.
        menubar = tk.Menu(master=start_win, )
        start_win.config(menu=menubar)

        os_accelerator = 'Command' if const.MY_OS == 'dar' else 'Ctrl'
        file = tk.Menu(master=self.master, tearoff=0)
        menubar.add_cascade(label=utils.program_name(), menu=file)
        file.add_command(label='Process now',
                         command=_call_start,
                         accelerator='Return')  # macOS doesn't recognize 'Enter'
        file.add_command(label='Quit',
                         command=lambda: utils.quit_gui(self),
                         # macOS doesn't recognize 'Command+Q' as an accelerator
                         #   b/c cannot override that system's native Command-Q,
                         accelerator=f'{os_accelerator}+Q')

        help_menu = tk.Menu(master=start_win, tearoff=0)
        tips = tk.Menu(master=start_win, tearoff=0)
        menubar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_cascade(label='Tips...', menu=tips)

        # Bullet symbol from https://coolsymbol.com/, unicode_escape: u'\u2022'
        # Unicode arrow symbols: left \u2190, right \u2192
        if const.MY_OS == 'dar':
            tip_scaling_text = 'with shift-control-← & shift-control-→.'
        else:
            tip_scaling_text = 'with Ctrl-← & Ctrl-→.'

        tips.add_command(label='• Images are automatically scaled to fit on')
        tips.add_command(label='     the screen. Scaling can be changed later')
        tips.add_command(label=f'     {tip_scaling_text}')
        tips.add_command(label='• Use a lighter font color with darker objects.')
        tips.add_command(label='• Font and line color can be changed with')
        tips.add_command(label=f'     {color_tip}.')
        tips.add_command(label='• Font size can be changed with')
        tips.add_command(label='     Ctrl-+ & Ctrl--(minus).')
        tips.add_command(label='• Font and line thickness can be changed with')
        tips.add_command(label='     Shift-Ctrl-+ & Shift-Ctrl--(minus).')
        tips.add_command(label='• Use the INVERSE threshold type for dark')
        tips.add_command(label='     objects on a light background.')
        tips.add_command(label='• Enter or Return key also starts processing.')
        tips.add_command(label="• More Tips are in the repository's README file.")
        tips.add_command(label='• Esc or Ctrl-Q from any window exits the program.')

        help_menu.add_command(label='About',
                              command=lambda: utils.about_win(parent=start_win))

        # Grid start win widgets; sorted by row.
        padding = dict(padx=5, pady=5)

        window_header.grid(row=0, column=0, columnspan=2, **padding, sticky=tk.EW)

        color_label.grid(row=2, column=0, **padding, sticky=tk.E)
        color_cbox.grid(row=2, column=1, **padding, sticky=tk.W)

        # Best to use cross-platform relative padding of color msg label,
        #  which is placed to the right of the color combobox.
        start_win.update()
        color_padx = (color_cbox.winfo_reqwidth() + 10, 0)
        color_msg_lbl.grid(row=2, column=1,
                           padx=color_padx, pady=5, sticky=tk.W)

        inverse_label.grid(row=3, column=0, **padding, sticky=tk.E)
        inverse_no.grid(row=3, column=1, **padding, sticky=tk.W)
        inverse_yes.grid(row=3, column=1, padx=(50, 0), sticky=tk.W)
        process_now_button.grid(row=3, column=1, **padding, sticky=tk.E)

        # Gray-out widget labels until an input file is selected.
        # The settings widgets themselves will be inactive while the
        #  filedialog window is open.
        color_label.config(state=tk.DISABLED)
        color_msg_lbl.config(state=tk.DISABLED)
        inverse_label.config(state=tk.DISABLED)

        # Take a break in configuring the window to grab the input.
        # For macOS: Need to have the filedialog be a child of
        #   start_win and need update() here.
        self.update()
        self.open_input(toplevel=start_win)

        # Finally, give start window its active title,...
        start_win.title('Set start parameters')

        # ...fill in window header with input path and pixel dimensions,...
        _h, _w = self.cvimg['gray'].shape
        window_header.config(text=f'Image: {self.input_file}\n'
                                  f'Image size, pixels (w x h): {_w}x{_h}')

        # ...and make all widgets active.
        color_label.config(state=tk.NORMAL)
        color_msg_lbl.config(state=tk.NORMAL)
        inverse_label.config(state=tk.NORMAL)

    def start_now(self) -> None:
        """
        Initiate the processing pipeline by setting up and configuring
        all settings widgets.
        Called from setup_start_window() with the "Process now...".
        Returns:
            None
        """

        # This calling sequence produces a slight delay (longer for larger files)
        #  before anything is displayed, but assures that everything displays
        #  nearly simultaneously for a visually cleaner start.
        self.setup_image_windows()
        self.configure_main_window()
        self.configure_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_entries()
        self.bind_annotation_styles()
        self.bind_scale_adjustment()
        self.set_defaults()
        self.grid_widgets()
        self.grid_img_labels()

        # Run processing for the starting image prior to displaying images.
        # Call preprocess(), process(), and display_windows(), in this
        #  sequence, for best performance. These methods are inherited
        #  from  ViewImage(), while the other methods are members of
        #  this SetupApp() class.
        self.preprocess()
        self.process()
        self.display_windows()

    def _delete_window_message(self) -> None:
        """
        Provide a notice in report and settings (mainloop, self)
        window.
        Called only as a .protocol() func in setup_image_windows().

        Returns: None
        """

        prev_txt = self.info_txt.get()
        prev_fg = self.info_label.cget('fg')

        _info = ('\nThat window cannot be closed from its window bar.\n'
                 'Minimize it if it is in the way.\n'
                 'Esc or Ctrl-Q keys will quit the program.\n\n')
        self.show_info_message(info=_info, color='vermilion')

        self.update()

        # Give user time to read the _info before resetting it to
        #  the previous info text.
        app.after(ms=5555)
        self.show_info_message(info=prev_txt, color=prev_fg)

    def setup_image_windows(self) -> None:
        """
        Create and configure all Toplevel windows and their Labels that
        are used to display and update processed images.

        Returns:
            None
        """

        # Dictionary item order determines stack order of windows.
        # Toplevel() is assigned here, not in __init__, to control timing
        #  and smoothness of window appearance at startup.
        # Note that keys in tkimg_window and window_title must match
        self.tkimg_window = {
            'input_contrasted': tk.Toplevel(),
            'reduced_filtered': tk.Toplevel(),
            'thresholded_segmented': tk.Toplevel(),
            'sized': tk.Toplevel(),
        }

        self.window_title = {
            'input_contrasted': 'Input image <- | -> Adjusted grayscale contrast',
            'reduced_filtered': 'Reduced noise <- | -> Filtered',
            'thresholded_segmented': f'Thresholded <- | -> {self.seg_algorithm} segments',
            'sized': 'Selected & Sized objects',
        }

        # Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods
        #  via ProcessImage.update_image().
        #  Labels are gridded in their respective tkimg_window in grid_img_labels().
        self.img_label = {
            'input': tk.Label(self.tkimg_window['input_contrasted']),
            'contrasted': tk.Label(self.tkimg_window['input_contrasted']),

            'reduced_noise': tk.Label(self.tkimg_window['reduced_filtered']),
            'filtered': tk.Label(self.tkimg_window['reduced_filtered']),

            'thresholded': tk.Label(self.tkimg_window['thresholded_segmented']),
            'segmented_objects': tk.Label(self.tkimg_window['thresholded_segmented']),

            'sized': tk.Label(self.tkimg_window['sized'])
        }

        # Need an image to replace blank tk desktop icon for each img window.
        #  Set correct path to the local 'images' directory and icon file.
        icon_path = None
        try:
            #  If the icon file is not present, a Terminal notice will be
            #   printed from <if __name__ == "__main__"> at startup.
            icon_path = tk.PhotoImage(file=utils.valid_path_to('image/sizeit_icon_512.png'))
            self.iconphoto(True, icon_path)
        except tk.TclError as _msg:
            pass

        # Withdraw all windows here for clean transition; all are deiconified
        #  in display_windows().
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        # Allow image label panels in image windows to resize with window.
        #  Note that images don't proportionally resize, just their boundaries;
        #  images will remain anchored at their top left corners.
        for _name, _toplevel in self.tkimg_window.items():
            _toplevel.wm_withdraw()
            if icon_path:
                _toplevel.iconphoto(True, icon_path)
            _toplevel.wm_minsize(width=200, height=100)
            _toplevel.protocol(name='WM_DELETE_WINDOW', func=self._delete_window_message)
            _toplevel.columnconfigure(index=0, weight=1)
            _toplevel.columnconfigure(index=1, weight=1)
            _toplevel.rowconfigure(index=0, weight=1)
            _toplevel.title(self.window_title[_name])
            _toplevel.config(**const.WINDOW_PARAMETERS)
            _toplevel.bind('<Escape>', func=lambda _: utils.quit_gui(self))
            _toplevel.bind('<Control-q>', func=lambda _: utils.quit_gui(self))

    def configure_main_window(self) -> None:
        """
        Settings and report window (mainloop, self) keybindings,
        configurations, and grids for contour settings and reporting frames.

        Returns:
            None
        """

        # Color in the main (self) window and give it a yellow border;
        #   border highlightcolor changes to grey with loss of focus.
        self.config(**const.WINDOW_PARAMETERS)

        # Default Frame() arguments work fine to display report text.
        # bg won't show when grid sticky EW for tk.Text; see utils.display_report().
        self.selectors_frame.configure(relief='raised',
                                       bg=const.DARK_BG,
                                       # bg=const.COLORS_TK['sky blue'],  # for development
                                       borderwidth=5)

        # Allow Frames and widgets to resize with main window.
        #  Row 1 is the report, row2 selectors, rows 2,3,4 are for Buttons().
        self.rowconfigure(index=0, weight=1)
        self.rowconfigure(index=1, weight=1)

        # Keep the report scrollbar active in the resized frame.
        self.report_frame.rowconfigure(index=0, weight=1)

        # Expect there to be 20 rows in the selectors Frame.
        for i in range(21):
            self.selectors_frame.rowconfigure(index=i, weight=1)

        self.columnconfigure(index=0, weight=1)
        self.columnconfigure(index=1, weight=1)
        self.report_frame.columnconfigure(index=0, weight=1)

        # Allow only sliders, not labels, to expand with window.
        self.selectors_frame.columnconfigure(index=1, weight=1)

        self.report_frame.grid(column=0, row=0,
                               columnspan=2,
                               padx=(5, 5), pady=(5, 5),
                               sticky=tk.EW)
        self.selectors_frame.grid(column=0, row=1,
                                  columnspan=2,
                                  padx=5, pady=(0, 5),
                                  ipadx=4, ipady=4,
                                  sticky=tk.EW)

        self.info_label.config(font=const.WIDGET_FONT,
                               bg=const.MASTER_BG,
                               fg='black')

        # Note: with rowspan=5, there must be 5 return characters in
        #  each info string to prevent shifts of frame row spacing.
        #  5 because that seems to be needed to cover the combined
        #  height of the last three rows (2, 3, 4) with buttons.
        #  Sticky is 'east' to prevent horizontal shifting when, during
        #  segmentation processing, all buttons in col 0 are removed.
        self.info_label.grid(column=1, row=2, rowspan=5,
                             padx=(0, 20), sticky=tk.E)

        # Note: the main window (mainloop, self, app) is deiconified in
        #  display_windows() after all image windows so that, at startup,
        #  it stacks on top.

    def _settings_dict(self) -> dict:
        """
        Creates a dictionary is used for the 'Export settings' button
        cmd to save all current settings to a json file. The dict keys
        must match those in slider_val, cbox, cbox_val, and size_std
        dictionaries.
        Called only from configure_buttons._export_settings().

        Returns: A dictionary of all settings values.
        """

        if self.seg_algorithm == 'Watershed':
            ws_connectivity = int(self.cbox_val['ws_connectivity'].get())
        else:  # is 'Random Walker'
            # Set to 1 b/c if Run Watershed following a settings import,
            #  then anything than integer 1, 4, 8 raises an error. A 'n/a'
            #  will be reported when running Random Walker, however.
            ws_connectivity = 1

        settings_dict = {
            'alpha': self.slider_val['alpha'].get(),
            'beta': self.slider_val['beta'].get(),
            'noise_iter': self.slider_val['noise_iter'].get(),
            'morphop': self.cbox_val['morphop'].get(),
            'morphshape': self.cbox_val['morphshape'].get(),
            'filter_type': self.cbox_val['filter_type'].get(),
            'th_type': self.cbox_val['th_type'].get(),
            'do_inverse_th': self.do_inverse_th.get(),
            'circle_r_min': self.slider_val['circle_r_min'].get(),
            'circle_r_max': self.slider_val['circle_r_max'].get(),
            'plm_mindist': self.slider_val['plm_mindist'].get(),
            'plm_footprint': self.slider_val['plm_footprint'].get(),
            'dt_type': self.cbox_val['dt_type'].get(),
            'dt_mask_size': int(self.cbox_val['dt_mask_size'].get()),
            'ws_connectivity': ws_connectivity,
            'noise_k': self.slider_val['noise_k'].get(),
            'filter_k': self.slider_val['filter_k'].get(),
            'size_std': self.cbox_val['size_std'].get(),
            # 'scale': self.scale_factor.get(),
            'px_val': self.size_std['px_val'].get(),
            'custom_val': self.size_std['custom_val'].get(),
            'color': self.cbox_val['color'].get(),
            'font_scale': self.metrics['font_scale'],
            'line_thickness': self.metrics['line_thickness'],
            'seg_algorithm': self.seg_algorithm,
        }

        return settings_dict

    def configure_buttons(self) -> None:
        """
        Assign and grid Buttons in the settings (mainloop, self) window.
        Called from __init__.

        Returns:
            None
        """
        manage.ttk_styles(mainloop=self)

        input_folder = Path(self.input_file).parent

        # These inner functions are used for Button commands.
        def _run_watershed():
            self.seg_algorithm = 'Watershed'
            self.process()
            self.tkimg_window['thresholded_segmented'].title(
                'Thresholded <- | -> Watershed segments')

        def _run_random_walker():
            self.seg_algorithm = 'Random Walker'
            self.process()
            self.tkimg_window['thresholded_segmented'].title(
                'Thresholded <- | -> Random Walker segments')

        def _save_results():
            """
            Save annotated sized image and its Report text with
            individual object sizes appended.
            """
            _sizes = ', '.join(str(i) for i in self.sorted_size_list)
            utils.save_report_and_img(
                input_path=self.input_file,
                img2save=self.cvimg['sized'],
                txt2save=self.report_txt + f'\n{_sizes}',
                caller=utils.program_name(),
            )

            _info = ('\n\nSettings report and result image have been saved to:\n'
                     f'{input_folder}\n\n')
            self.show_info_message(info=_info, color='blue')

        def _export_settings():
            """
            Save only the settings dictionary, as a json file. It is
            handled as a special case in utils.save_report_and_img().
            """

            settings_path = f'{input_folder}/{const.SETTINGS_FILE_NAME}'
            settings_dict = self._settings_dict()
            utils.save_report_and_img(
                input_path=settings_path,
                img2save=const.STUB_ARRAY,
                txt2save='',
                caller='',
                settings2save=settings_dict,
            )

            _info = ('\nSettings values have been exported to:\n'
                     f'{settings_path}\n'
                     'and are available to use with "New input" or\n'
                     'at startup. Previous settings file is overwritten.\n')
            self.show_info_message(info=_info, color='blue')

        def _export_objects():
            self.export_segment = messagebox.askyesnocancel(
                title="Export only objects' segmented areas?",
                detail='Yes, ...with a white background.\n'
                       'No, include area around object\n'
                       '     ...with image\'s background.\n'
                       'Cancel: Export nothing and return.')

            if self.export_segment:
                self.export_hull = messagebox.askyesno(
                    title='Fill in partially segmented objects?',
                    detail='Yes, try to include more object area;\n'
                           '     may include some image background.\n'
                           'No, export just segments, on white.\n')

            _num = self.select_and_export_objects()
            _info = (f'\n\n{_num} selected objects were individually exported to:\n'
                     f'{input_folder}\n\n')
            self.show_info_message(info=_info, color='blue')

        def _new_input():
            """
            Reads a new image file and applies current settings for
            preprocessing.
            Returns: None
            """
            if self.open_input(toplevel=self):
                self.update_image(tkimg_name='input',
                                  cvimg_array=self.cvimg['input'])
                _info = '\n\nA new input image has been loaded. Processing...\n\n\n'
                self.show_info_message(info=_info, color='blue')

            else:  # User canceled input selection or closed messagebox window.
                _info = '\n\nNo new input file was selected.\n\n\n'
                self.show_info_message(info=_info, color='vermilion')

                self.delay_size_std_info_msg()

                return

            if self.use_saved_settings:
                self.import_settings()

            self.preprocess()
            self.process()
            self.report_results()

            _info = ('\n\nProcessing completed for the new input image.\n'
                     f'{self.elapsed} processing seconds elapsed.\n\n')
            self.show_info_message(info=_info, color='blue')

        def _reset_to_default_settings():
            """Order of calls is important here."""
            self.slider_values.clear()
            self.metrics = manage.input_metrics(img=self.cvimg['input'])
            self.set_auto_scale_factor()
            self.configure_circle_r_sliders()
            self.set_defaults()
            self.widget_control('off')  # is turned 'on' in preprocess()
            self.preprocess()

            _info = ('\nSettings have been reset to their defaults.\n'
                     'Check and adjust them if needed, then...\n'
                     'Select a "Run" button to finalize updating the\n'
                     'report and image results.\n')
            self.show_info_message(info=_info, color='blue')

        # Configure all items in the dictionary of ttk buttons.
        button_params = dict(
            width=0,
            style='My.TButton',
        )

        self.button['process_ws'].config(
            text='Run Watershed',
            command=_run_watershed,
            **button_params)

        self.button['process_rw'].config(
            text='Run Random Walker',
            command=_run_random_walker,
            **button_params)

        self.button['save_results'].config(
            text='Save results',
            command=_save_results,
            **button_params)

        self.button['export_settings'].config(
            text='Export settings',
            command=_export_settings,
            **button_params)

        self.button['new_input'].config(
            text='New input',
            command=_new_input,
            **button_params)

        self.button['export_objects'].config(
            text='Export objects',
            command=_export_objects,
            **button_params)

        self.button['reset'].config(
            text='Reset',
            command=_reset_to_default_settings,
            **button_params)

    def _need_to_click(self, event=None):
        """
        Post notice when selecting peak_local_max, because plm slider
        values are used in segmentation methods, which are called
        only from button commands.
        Called only from mouse click binding in config_sliders().
        """

        # Update report with current plm_* slider values; don't wait
        #  for the segmentation algorithm to run before updating.
        self.report_results()

        if self.slider_val['plm_footprint'].get() == 1:
            _info = ('\nA peak_local_max footprint of 1 may run a long time.\n'
                     'If that is not a problem, then...\n'
                     'Click a "Run" button to update the "Size-selected.." \n'
                     'and "Segmented objects" images.\n')
            self.show_info_message(info=_info, color='vermilion')
        else:
            _info = ('\nClick "Run" to update the report and the\n'
                     '"Size-selected.." and "Segmented objects" images.\n\n\n')
            self.show_info_message(info=_info, color='blue')

        return event

    def config_sliders(self) -> None:
        """
        Configure arguments and mouse button bindings for all Scale
        widgets in the settings (mainloop, self) window.
        Called from __init__.

        Returns:
            None
        """
        # Set minimum width for the enclosing Toplevel by setting a length
        #  for a single Scale widget that is sufficient to fit everything
        #  in the Frame given current padding parameters. Need to use only
        #  for one Scale() in each Toplevel().
        scale_len = int(self.screen_width * 0.25)

        # Scale widgets that are pre-random_walker (contrast, noise,
        # filter and threshold) and size max/min are called with mouse
        # button release.
        # Peak-local-max params and select_and_size_objects() are called with Button.
        self.slider['alpha_lbl'].configure(text='Contrast/gain/alpha:',
                                           **const.LABEL_PARAMETERS)
        self.slider['alpha'].configure(from_=0.0, to=4.0,
                                       length=scale_len,
                                       resolution=0.1,
                                       tickinterval=0.5,
                                       variable=self.slider_val['alpha'],
                                       **const.SCALE_PARAMETERS)

        self.slider['beta_lbl'].configure(text='Brightness/bias/beta:',
                                          **const.LABEL_PARAMETERS)
        self.slider['beta'].configure(from_=-127, to=127,
                                      tickinterval=25,
                                      variable=self.slider_val['beta'],
                                      **const.SCALE_PARAMETERS)

        self.slider['noise_k_lbl'].configure(text='Reduce noise, kernel size\n'
                                                  '(only odd integers used):',
                                             **const.LABEL_PARAMETERS)
        self.slider['noise_k'].configure(from_=1, to=51,
                                         tickinterval=5,
                                         variable=self.slider_val['noise_k'],
                                         **const.SCALE_PARAMETERS)

        self.slider['noise_iter_lbl'].configure(text='Reduce noise, iterations\n'
                                                     '(0 may extend processing time):',
                                                **const.LABEL_PARAMETERS)

        self.slider['noise_iter'].configure(from_=0, to=5,
                                            tickinterval=1,
                                            variable=self.slider_val['noise_iter'],
                                            **const.SCALE_PARAMETERS)

        self.slider['filter_k_lbl'].configure(text='Filter kernel size\n'
                                                   '(only odd integers, 0 is no filter):',
                                              **const.LABEL_PARAMETERS)
        self.slider['filter_k'].configure(from_=0, to=111,
                                          tickinterval=9,
                                          variable=self.slider_val['filter_k'],
                                          **const.SCALE_PARAMETERS)

        self.slider['plm_mindist_lbl'].configure(text='peak_local_max min_distance:',
                                                 **const.LABEL_PARAMETERS)
        self.slider['plm_mindist'].configure(from_=1, to=500,
                                             length=scale_len,
                                             tickinterval=40,
                                             variable=self.slider_val['plm_mindist'],
                                             **const.SCALE_PARAMETERS)

        self.slider['plm_footprint_lbl'].configure(text='peak_local_max footprint:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['plm_footprint'].configure(from_=1, to=50,
                                               length=scale_len,
                                               tickinterval=5,
                                               variable=self.slider_val['plm_footprint'],
                                               **const.SCALE_PARAMETERS)

        self.slider['circle_r_min_lbl'].configure(text='Circled radius size\n'
                                                       'minimum pixels:',
                                                  **const.LABEL_PARAMETERS)
        self.slider['circle_r_max_lbl'].configure(text='Circled radius size\n'
                                                       'maximum pixels:',
                                                  **const.LABEL_PARAMETERS)

        self.configure_circle_r_sliders()

        # To avoid processing all the intermediate values between normal
        #  slider movements, bind sliders to call functions only on
        #  left button release.
        # Most are bound to preprocess(); process() is initiated
        #  only with a Button(). To speed program responsiveness when
        #  changing the size range, only the sizing and reporting methods
        #  are called on mouse button release.
        # Note that the isinstance() condition doesn't improve performance,
        #  but is there for clarity's sake.
        for _name, _w in self.slider.items():
            if isinstance(_w, tk.Label):
                continue
            if 'circle_r' in _name:
                _w.bind('<ButtonRelease-1>',
                        func=lambda _: self.process_sizes(caller='circle_r'))
            elif 'plm_' in _name:
                _w.bind('<ButtonRelease-1>', self._need_to_click)
            else:  # is alpha, beta, noise_k, noise_iter, filter_k.
                _w.bind('<ButtonRelease-1>', self.preprocess)

    def config_comboboxes(self) -> None:
        """
        Configure arguments and mouse button bindings for all Comboboxes
        in the settings (mainloop, self) window.
        Called from __init__.

        Returns:
             None
        """

        # Different Combobox widths are needed to account for font widths
        #  and padding in different systems.
        width_correction = 2 if const.MY_OS == 'win' else 0  # is Linux or macOS

        # Combobox styles are set in manage.ttk_styles(), called in configure_buttons().
        self.cbox['morphop_lbl'].config(text='Reduce noise, morphology operator:',
                                        **const.LABEL_PARAMETERS)
        self.cbox['morphop'].config(textvariable=self.cbox_val['morphop'],
                                    width=18 + width_correction,
                                    values=list(const.CV_MORPHOP.keys()),
                                    **const.COMBO_PARAMETERS)

        self.cbox['morphshape_lbl'].config(text='... shape:',
                                           **const.LABEL_PARAMETERS)
        self.cbox['morphshape'].config(textvariable=self.cbox_val['morphshape'],
                                       width=16 + width_correction,
                                       values=list(const.CV_MORPH_SHAPE.keys()),
                                       **const.COMBO_PARAMETERS)

        self.cbox['filter_lbl'].config(text='Filter type:',
                                       **const.LABEL_PARAMETERS)
        self.cbox['filter_type'].config(textvariable=self.cbox_val['filter_type'],
                                        width=14 + width_correction,
                                        values=list(const.CV_FILTER.keys()),
                                        **const.COMBO_PARAMETERS)

        self.cbox['th_type_lbl'].config(text='Threshold type:',
                                        **const.LABEL_PARAMETERS)
        self.cbox['th_type'].config(textvariable=self.cbox_val['th_type'],
                                    width=26 + width_correction,
                                    values=list(const.THRESH_TYPE.keys()),
                                    **const.COMBO_PARAMETERS)

        self.cbox['dt_type_lbl'].configure(text='cv2.distanceTransform, distanceType:',
                                           **const.LABEL_PARAMETERS)
        self.cbox['dt_type'].configure(textvariable=self.cbox_val['dt_type'],
                                       width=12,
                                       values=list(const.DISTANCE_TRANS_TYPE.keys()),
                                       **const.COMBO_PARAMETERS)

        self.cbox['dt_mask_size_lbl'].configure(text='... maskSize:',
                                                **const.LABEL_PARAMETERS)
        self.cbox['dt_mask_size'].configure(textvariable=self.cbox_val['dt_mask_size'],
                                            width=2,
                                            values=('0', '3', '5'),
                                            **const.COMBO_PARAMETERS)
        # mask size constants are: cv2.DIST MASK PRECISE, cv2.DIST MASK 3, cv2.DIST MASK 5.

        self.cbox['ws_connectivity_lbl'].config(text='Watershed connectivity:',
                                                **const.LABEL_PARAMETERS)
        self.cbox['ws_connectivity'].config(textvariable=self.cbox_val['ws_connectivity'],
                                            width=2,
                                            values=('1', '4', '8'),
                                            **const.COMBO_PARAMETERS)

        self.cbox['size_std_lbl'].config(text='Select the standard used in image:',
                                         **const.LABEL_PARAMETERS)
        self.cbox['size_std'].config(textvariable=self.cbox_val['size_std'],
                                     width=12 + width_correction,
                                     values=list(const.SIZE_STANDARDS.keys()),
                                     **const.COMBO_PARAMETERS)

        # Now bind functions to all Comboboxes.
        # Note that the isinstance() Label condition isn't needed for
        # performance, it just clarifies the bind intention.
        for _name, _w in self.cbox.items():
            if isinstance(_w, tk.Label):
                continue
            if _name == 'size_std':
                _w.bind('<<ComboboxSelected>>',
                        func=lambda _: self.process_sizes(caller='size_std'))
            else:  # is morphop, morphshape, filter, th_type, dt_type, dt_mask_size.
                _w.bind('<<ComboboxSelected>>', func=self.preprocess)

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for Entry widgets
        in the settings (mainloop, self) window.
        Called from __init__.

        Returns: None
        """

        self.size_std['px_entry'].config(textvariable=self.size_std['px_val'],
                                         # font=const.WIDGET_FONT,
                                         width=6)
        self.size_std['px_lbl'].config(text='Enter px diameter of size standard:',
                                       **const.LABEL_PARAMETERS)

        self.size_std['custom_entry'].config(textvariable=self.size_std['custom_val'],
                                             width=8)
        self.size_std['custom_lbl'].config(text="Enter custom standard's size:",
                                           **const.LABEL_PARAMETERS)

        for _name, _w in self.size_std.items():
            if isinstance(_w, tk.Entry):
                _w.bind('<Return>', lambda _, n=_name: self.process_sizes(caller=n))
                _w.bind('<KP_Enter>', lambda _, n=_name: self.process_sizes(caller=n))

    def _current_contours(self) -> list:
        """
        Determines current segment algorithm is use.
        Called only from internal functions in bind_annotation_styles().
        Returns:
            Current algorithm's contour pointset, as a list.

        """
        if self.seg_algorithm == 'Watershed':
            contours: list = self.ws_basins
        else:  # is 'Random Walker'
            contours: list = self.rw_contours
        return contours

    def bind_annotation_styles(self) -> None:
        """
        Set key bindings to change font size, color, and line thickness
        of annotations in the 'sized' cv2 image.
        Called after at startup and after any segmentation algorithm call
        from process().

        Returns: None
        """

        def _increase_font_size() -> None:
            """Limit upper font size scale to a 5x increase."""
            font_scale: float = self.metrics['font_scale']
            font_scale *= 1.1
            font_scale = round(min(font_scale, 5), 2)
            self.metrics['font_scale'] = font_scale
            self.select_and_size_objects(contour_pointset=self._current_contours())

        def _decrease_font_size() -> None:
            """Limit lower font size scale to a 1/5 decrease."""
            font_scale: float = self.metrics['font_scale']
            font_scale *= 0.9
            font_scale = round(max(font_scale, 0.20), 2)
            self.metrics['font_scale'] = font_scale
            self.select_and_size_objects(contour_pointset=self._current_contours())

        def _increase_line_thickness() -> None:
            """Limit upper thickness to 10."""
            line_thickness: int = self.metrics['line_thickness']
            line_thickness += 1
            self.metrics['line_thickness'] = min(line_thickness, 10)
            self.select_and_size_objects(contour_pointset=self._current_contours())

        def _decrease_line_thickness() -> None:
            """Limit lower thickness to 1."""
            line_thickness: int = self.metrics['line_thickness']
            line_thickness -= 1
            self.metrics['line_thickness'] = max(line_thickness, 1)
            self.select_and_size_objects(contour_pointset=self._current_contours())

        colors = list(const.COLORS_CV.keys())

        def _next_font_color() -> None:
            current_color: str = self.cbox_val['color'].get()
            current_index = colors.index(current_color)
            # Need to stop increasing idx at the end of colors list.
            if current_index == len(colors) - 1:
                next_color = colors[len(colors) - 1]
            else:
                next_color = colors[current_index + 1]
            self.cbox_val['color'].set(next_color)
            print('Annotation font is now:', next_color)
            self.select_and_size_objects(contour_pointset=self._current_contours())

        def _preceding_font_color() -> None:
            current_color: str = self.cbox_val['color'].get()
            current_index = colors.index(current_color)
            # Need to stop decreasing idx at the beginning of colors list.
            if current_index == 0:
                current_index = 1
            preceding_color = colors[current_index - 1]
            self.cbox_val['color'].set(preceding_color)
            print('Annotation font is now :', preceding_color)
            self.select_and_size_objects(contour_pointset=self._current_contours())

        # Bindings are needed only for the settings and sized img windows,
        #  but is simpler to use bind_all() which does not depend on widget focus.
        # NOTE: On Windows, KP_* is not a recognized keysym string; works on Linux.
        #  Windows keysyms 'plus' & 'minus' are for both keyboard and keypad.
        self.bind_all('<Control-equal>', lambda _: _increase_font_size())
        self.bind_all('<Control-minus>', lambda _: _decrease_font_size())
        self.bind_all('<Control-KP_Subtract>', lambda _: _decrease_font_size())

        self.bind_all('<Shift-Control-plus>', lambda _: _increase_line_thickness())
        self.bind_all('<Shift-Control-KP_Add>', lambda _: _increase_line_thickness())
        self.bind_all('<Shift-Control-underscore>', lambda _: _decrease_line_thickness())

        self.bind_all('<Control-Up>', lambda _: _next_font_color())
        self.bind_all('<Control-Down>', lambda _: _preceding_font_color())

        # Need platform-specific keypad keysym.
        if const.MY_OS == 'win':
            self.bind_all('<Control-plus>', lambda _: _increase_font_size())
            self.bind_all('<Shift-Control-minus>', lambda _: _decrease_line_thickness())
        else:
            self.bind_all('<Control-KP_Add>', lambda _: _increase_font_size())
            self.bind_all('<Shift-Control-KP_Subtract>', lambda _: _decrease_line_thickness())

    def bind_scale_adjustment(self) -> None:
        """
        The displayed image scale is set when an image is imported, but
        can be adjusted in-real-time with these keybindings.

        Returns: None
        """

        # These names need to be a subset of those in ProcessImage.__init__
        #  tuple of image_names used to create keys for the cvimg dictionary.
        tkimg_names = ('input',
                       'contrasted',
                       'reduced_noise',
                       'filtered',
                       'thresholded',
                       'sized',
                       )

        # Note that these string names must match the respective keys in
        #  draw_ws_segments() and draw_rw_segments() methods.
        seg_name = 'Watershed' if self.seg_algorithm == 'Watershed' else 'Random Walker'

        def _apply_new_scale():
            """
            The scale_factor is applied in ProcessImage.update_image()
            """
            _sf = round(self.scale_factor.get(), 2)
            _info = f'\n\nA new scale factor of {_sf} has been applied.\n\n\n'
            self.show_info_message(info=_info, color='black')

            # Note: conversion with np.uint8() for the cvimg_array argument
            #  is required to display the 'transformed' image, if used.
            for _n in tkimg_names:
                self.update_image(tkimg_name=_n,
                                  cvimg_array=self.cvimg[_n])
            self.update_image(tkimg_name='segmented_objects',
                              cvimg_array=self.cvimg[seg_name])

        def _increase_scale_factor() -> None:
            """
            Limit upper factor to a 4x increase to maintain performance.
            """
            scale_factor: float = self.scale_factor.get()
            scale_factor *= 1.1
            scale_factor = round(min(scale_factor, 4), 2)
            self.scale_factor.set(scale_factor)
            _apply_new_scale()

        def _decrease_scale_factor() -> None:
            """
            Limit lower factor to a 1/4 decrease to maintain readability.
            """
            scale_factor: float = self.scale_factor.get()
            scale_factor *= 0.9
            scale_factor = round(max(scale_factor, 0.25), 2)
            self.scale_factor.set(scale_factor)
            _apply_new_scale()

        self.bind_all('<Control-Right>', lambda _: _increase_scale_factor())
        self.bind_all('<Control-Left>', lambda _: _decrease_scale_factor())

    def set_defaults(self) -> None:
        """
        Sets and resets selector widgets and keybind scaling functions
        for image sizes and annotations. Evaluates selection of using
        saved or default settings.

        Called from __init__ and "Reset" button.
        Returns:
            None
        """

        # Default settings are optimized for sample1.jpg input.

        # This condition is needed to evaluate user's choices at startup.
        if self.first_run and self.use_saved_settings:
            self.import_settings()
            return

        if self.first_run:
            if self.do_inverse_th.get():
                self.cbox['th_type'].current(1)
                self.cbox_val['th_type'].set('cv2.THRESH_OTSU_INVERSE')
            else:  # user selected 'No' for INVERSE threshold type.
                self.cbox['th_type'].current(0)
                self.cbox_val['th_type'].set('cv2.THRESH_OTSU')
        else:
            self.cbox['th_type'].current(0)  # 'cv2.THRESH_OTSU'

        # Only at startup, does this seg_algorithm value have effect;
        #   after that, when the 'Reset' button is used,
        #   its value is defined by the "Run..." buttons.
        self.seg_algorithm = 'Watershed'

        # Set/Reset Scale widgets.
        self.slider_val['alpha'].set(1.0)
        self.slider_val['beta'].set(0)
        self.slider_val['noise_k'].set(7)
        self.slider_val['noise_iter'].set(3)
        self.slider_val['filter_k'].set(5)
        self.slider_val['circle_r_min'].set(8)
        self.slider_val['circle_r_max'].set(300)

        # Increase PLM values for larger files to reduce the number
        #  of contours, thus decreasing startup/reset processing time.
        if self.metrics['img_area'] > 6 * 10e5:
            self.slider_val['plm_mindist'].set(200)
            self.slider_val['plm_footprint'].set(9)
        else:
            self.slider_val['plm_mindist'].set(40)
            self.slider_val['plm_footprint'].set(3)

        # Set/Reset Combobox widgets.
        self.cbox['morphop'].current(0)  # 'cv2.MORPH_OPEN' == 2
        self.cbox['morphshape'].current(2)  # 'cv2.MORPH_ELLIPSE' == 2
        self.cbox['filter_type'].current(1)  # 'cv2.bilateralFilter'
        self.cbox['dt_type'].current(1)  # 'cv2.DIST_L2' == 2
        self.cbox['dt_mask_size'].current(1)  # '3' == cv2.DIST_MASK_3
        self.cbox['ws_connectivity'].current(1)  # '4'
        self.cbox['size_std'].current(0)  # 'None'
        self.cbox_val['color'].set('blue')

        # Set/Reset Entry widgets.
        self.size_std['px_val'].set('1')
        self.size_std['custom_val'].set('0.0')

    def grid_widgets(self) -> None:
        """
        Developer: Grid all widgets as a method to clarify spatial
        relationships.
        Called from __init__.

        Returns:
            None
        """

        # Use the dict() function with keyword arguments to mimic the
        #  keyword parameter structure of the grid() function.
        east_grid_params = dict(
            padx=5,
            pady=(4, 0),
            sticky=tk.E)
        east_params_relative = dict(
            pady=(4, 0),
            sticky=tk.E)
        slider_grid_params = dict(
            padx=5,
            pady=(4, 0),
            sticky=tk.EW)
        west_grid_params = dict(
            padx=5,
            pady=(4, 0),
            sticky=tk.W)
        button_grid_params = dict(
            padx=10,
            pady=(0, 2),
            sticky=tk.W)
        relative_b_grid_params = dict(
            pady=(0, 2),
            sticky=tk.W)

        # Widgets gridded in the self.selectors_frame Frame.
        # Sorted by row number:
        self.slider['alpha_lbl'].grid(column=0, row=0, **east_grid_params)
        self.slider['alpha'].grid(column=1, row=0, **slider_grid_params)

        self.slider['beta_lbl'].grid(column=0, row=1, **east_grid_params)
        self.slider['beta'].grid(column=1, row=1, **slider_grid_params)

        self.cbox['morphop_lbl'].grid(column=0, row=2, **east_grid_params)
        self.cbox['morphop'].grid(column=1, row=2, **west_grid_params)

        # Note: Put morph shape on same row as morph op.
        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['morphshape'].grid(column=1, row=2, **east_grid_params)

        self.slider['noise_k_lbl'].grid(column=0, row=4, **east_grid_params)
        self.slider['noise_k'].grid(column=1, row=4, **slider_grid_params)

        self.slider['noise_iter_lbl'].grid(column=0, row=5, **east_grid_params)
        self.slider['noise_iter'].grid(column=1, row=5, **slider_grid_params)

        self.cbox['filter_lbl'].grid(column=0, row=6, **east_grid_params)
        self.cbox['filter_type'].grid(column=1, row=6, **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['th_type'].grid(column=1, row=6, **east_grid_params)

        self.slider['filter_k_lbl'].grid(column=0, row=9, **east_grid_params)
        self.slider['filter_k'].grid(column=1, row=9, **slider_grid_params)

        self.cbox['dt_type_lbl'].grid(column=0, row=10, **east_grid_params)
        self.cbox['dt_type'].grid(column=1, row=10, **west_grid_params)

        # May not be optimized placement for non-Linux platforms.
        # If change the padx here, match it below for the mask_lbl_padx offset.
        self.cbox['dt_mask_size_lbl'].grid(column=1, row=10,
                                           padx=(120, 0),
                                           pady=(4, 0),
                                           sticky=tk.W)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['ws_connectivity'].grid(column=1, row=10, **east_grid_params)

        self.slider['plm_mindist_lbl'].grid(column=0, row=12, **east_grid_params)
        self.slider['plm_mindist'].grid(column=1, row=12, **slider_grid_params)

        self.slider['plm_footprint_lbl'].grid(column=0, row=13, **east_grid_params)
        self.slider['plm_footprint'].grid(column=1, row=13, **slider_grid_params)

        self.slider['circle_r_min_lbl'].grid(column=0, row=17, **east_grid_params)
        self.slider['circle_r_min'].grid(column=1, row=17, **slider_grid_params)

        self.slider['circle_r_max_lbl'].grid(column=0, row=18, **east_grid_params)
        self.slider['circle_r_max'].grid(column=1, row=18, **slider_grid_params)

        self.size_std['px_lbl'].grid(column=0, row=19, **east_grid_params)
        self.size_std['px_entry'].grid(column=1, row=19, **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['size_std'].grid(column=1, row=19, **east_grid_params)

        self.size_std['custom_entry'].grid(column=1, row=20, **east_grid_params)

        # Buttons are in the mainloop window, not in a Frame().
        self.button['process_ws'].grid(column=0, row=2, **button_grid_params)
        self.button['save_results'].grid(column=0, row=3, **button_grid_params)
        self.button['export_objects'].grid(column=0, row=4, **button_grid_params)

        # Use update() because update_idletasks() doesn't always work to
        #  get the gridded widgets' correct winfo_reqwidth.
        self.update()

        # Now grid widgets with relative padx values based on widths of
        #  their corresponding partner widgets. Needed across platforms.
        morphshape_padx = (0, self.cbox['morphshape'].winfo_reqwidth() + 10)
        thtype_padx = (0, self.cbox['th_type'].winfo_reqwidth() + 10)
        mask_lbl_padx = (self.cbox['dt_mask_size_lbl'].winfo_reqwidth() + 120, 0)
        ws_connectivity_padx = (0, self.cbox['ws_connectivity'].winfo_reqwidth() + 10)
        size_std_padx = (0, self.cbox['size_std'].winfo_reqwidth() + 10)
        custom_std_padx = (0, self.size_std['custom_entry'].winfo_reqwidth() + 10)
        export_obj_w: int = self.button['export_objects'].winfo_reqwidth()
        save_results_w: int = self.button['save_results'].winfo_reqwidth()
        process_rw_padx = (self.button['process_ws'].winfo_reqwidth() + 15, 0)

        self.cbox['morphshape_lbl'].grid(column=1, row=2,
                                         padx=morphshape_padx,
                                         **east_params_relative)

        self.cbox['th_type_lbl'].grid(column=1, row=6,
                                      padx=thtype_padx,
                                      **east_params_relative)

        self.cbox['dt_mask_size'].grid(column=1, row=10,
                                       padx=mask_lbl_padx,
                                       pady=(4, 0),
                                       sticky=tk.W)

        self.cbox['ws_connectivity_lbl'].grid(column=1, row=10,
                                              padx=ws_connectivity_padx,
                                              **east_params_relative)

        self.cbox['size_std_lbl'].grid(column=1, row=19,
                                       padx=size_std_padx,
                                       **east_params_relative)

        self.size_std['custom_lbl'].grid(column=1, row=20,
                                         padx=custom_std_padx,
                                         **east_params_relative)

        # Remove initially; show only when Custom size is needed.
        self.size_std['custom_lbl'].grid_remove()

        # Buttons' grids in the mainloop (self) window.
        self.button['process_rw'].grid(
            column=0, row=2,
            padx=process_rw_padx,
            **relative_b_grid_params)

        self.button['export_settings'].grid(
            column=0, row=3,
            padx=(save_results_w + 15, 0),
            **relative_b_grid_params)

        self.button['new_input'].grid(
            column=0, row=4,
            padx=(export_obj_w + 15, 0),
            **relative_b_grid_params)

        self.button['reset'].grid(
            column=0, row=4,
            padx=(export_obj_w * 2, 0),
            **relative_b_grid_params)

    def grid_img_labels(self) -> None:
        """
        Grid all image Labels inherited from ProcessImage().
        Labels' 'master' argument for the img window is defined in
        ProcessImage.setup_image_windows(). Label 'image' param is
        updated with .configure() in each PI processing method.
        Called from __init__.

        Returns:
            None
        """

        self.img_label['input'].grid(**const.PANEL_LEFT)
        self.img_label['contrasted'].grid(**const.PANEL_RIGHT)

        self.img_label['reduced_noise'].grid(**const.PANEL_LEFT)
        self.img_label['filtered'].grid(**const.PANEL_RIGHT)

        self.img_label['thresholded'].grid(**const.PANEL_LEFT)
        self.img_label['segmented_objects'].grid(**const.PANEL_RIGHT)

        self.img_label['sized'].grid(**const.PANEL_RIGHT)

    def display_windows(self) -> None:
        """
        Ready all image window for display. Show the input image in its window.
        Bind rt-click to save any displayed image.
        Called from __init__.
        Calls update_image(), utils.save_report_and_img().

        Returns:
            None
        """

        # All image windows were withdrawn upon their creation in
        #  setup_image_windows() to keep things tidy.
        #  Now is the time to show them.
        for _, toplevel in self.tkimg_window.items():
            toplevel.wm_deiconify()

        # Display the input image. It is static, so does not need
        #  updating, but for consistency's sake the
        #  statement structure used to display and update processed
        #  images is used here.
        # Note: here and throughout, use 'self' to scope the
        #  ImageTk.PhotoImage image in the Class, otherwise it will/may
        #  not display because of garbage collection.
        self.update_image(tkimg_name='input',
                          cvimg_array=self.cvimg['input'])

        # macOS right mouse button has a different ID.
        rt_click = '<Button-3>' if const.MY_OS in 'lin, win' else '<Button-2>'

        # Do not specify the image array in this binding, but instead
        #  specify in _on_click_save_img() method so that the current image
        #  is saved. Bind to a lambda function, not to a direct call.
        for name, label in self.img_label.items():
            label.bind(rt_click,
                       lambda _, n=name: self._on_click_save_img(image_name=n))

        # Update to ensure that the label images are current.
        self.update()

        # Now is time to show the mainloop (self) report window that
        #   was hidden in setup_main_window().
        #   Deiconifying here stacks it on top of all windows at startup.
        self.wm_deiconify()


def run_checks() -> None:
    """
    Check system, versions, and command line arguments.
    Program exits if any critical check fails or if the argument
    --about is used, which prints 'about' info, then exits.
    Module check_platform() also enables display scaling on Windows.

    Returns:
            None
    """
    utils.check_platform()

    vcheck.minversion('3.7')
    vcheck.maxversion('3.11')
    manage.arguments()


if __name__ == "__main__":

    run_checks()

    try:
        print(f'{PROGRAM_NAME} has launched...')
        app = SetupApp()
        app.title(f'{PROGRAM_NAME} Report & Settings')

        # The custom app icon is expected to be in the repository images folder.
        try:
            icon = tk.PhotoImage(file=utils.valid_path_to('images/sizeit_icon_512.png'))
            app.wm_iconphoto(True, icon)
        except tk.TclError as err:
            print('Cannot display program icon, so it will be blank or the tk default.\n'
                  f'tk error message: {err}')

        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal command line. ***\n')
