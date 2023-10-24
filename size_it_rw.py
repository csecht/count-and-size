#!/usr/bin/env python3
"""
A tkinter GUI for OpenCV processing of an image to obtain sizes, means,
and ranges of objects in a sample population. The distance transform and
random walker algorithms are used interactively by setting their parameter
values with slide bars and pull-down menus. Related image processing
factors like contrast, brightness, noise reduction, and filtering are
also adjusted interactively, with live updating of the resulting images.

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
scaling, and annotation color cannot be changed after the "Process now"
button is clicked. Once image processing begins, if the run settings are
not to your liking, just quit, restart, and choose different values.

After changing slider or pull-down settings, click the "Run settings"
button to initiate a processing cycle for the new settings.

Save settings report and the annotated image with the "Save" button.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
settings windows, or from command line with Ctrl-C.

Requires Python 3.7 or later and the packages opencv-python, numpy,
scikit-image, scipy, and psutil.
See this distribution's requirements.txt file for details.
Developed in Python 3.8 and 3.9, tested up to 3.11.
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys
from pathlib import Path
from statistics import mean, median
from typing import List
from time import time

# Local application imports.
# pylint: disable=import-error
from utility_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             to_precision as to_p)

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
    from skimage.segmentation import random_walker
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
        f'Error message:\n{import_err}')


class ProcessImage(tk.Tk):
    """
    A suite of OpenCV methods for applying various image processing
    functions involved in segmenting objects from an image file.

    Class methods:
    update_image
    adjust_contrast
    reduce_noise
    filter_image
    randomwalk_segmentation
    contour_rw_segments
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
            # For scale_slider variable in setup_start_window()...
            'scale': tk.DoubleVar(),
        }

        self.cbox_val = {
            # For textvariables in config_comboboxes()...
            'morphop': tk.StringVar(),
            'morphshape': tk.StringVar(),
            'filter': tk.StringVar(),
            'th_type': tk.StringVar(),
            'dt_type': tk.StringVar(),
            'dt_mask_size': tk.StringVar(),
            'ws_connect': tk.StringVar(),
            'size_std': tk.StringVar(),
            # For color_cbox textvariable in setup_start_window()...
            'color': tk.StringVar(),
        }

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  with Label images displayed in their respective img_window Toplevel.
        # The cvimg images are numpy arrays.
        self.tkimg = {
            'input': tk.PhotoImage(),
            'gray': tk.PhotoImage(),
            'contrast': tk.PhotoImage(),
            'redux': tk.PhotoImage(),
            'filter': tk.PhotoImage(),
            'dist_trans': tk.PhotoImage(),
            'thresh': tk.PhotoImage(),
            'sized': tk.PhotoImage(),
        }

        self.cvimg = {
            'input': const.STUB_ARRAY,
            'gray': const.STUB_ARRAY,
            'contrast': const.STUB_ARRAY,
            'redux': const.STUB_ARRAY,
            'filter': const.STUB_ARRAY,
            'dist_trans': const.STUB_ARRAY,
            'thresh': const.STUB_ARRAY,
            'sized': const.STUB_ARRAY,
        }

        # img_label dictionary is set up in ImageViewer.setup_image_windows(),
        #  but is used in all Class methods here.
        self.img_label: dict = {}

        # metrics dict is populated in ImageViewer.setup_start_window()
        self.metrics: dict = {}

        self.num_dt_segments: int = 0
        self.randomwalk_contours: list = []
        self.sorted_size_list: list = []
        self.unit_per_px = tk.DoubleVar()
        self.num_sigfig: int = 0
        self.info_label = tk.Label(self)
        self.time_start: float = 0
        self.elapsed: float = 0
        self.first_run = True

    def update_image(self,
                     img_name: str,
                     img_array: np.ndarray) -> None:
        """
        Process a cv2 image array to use as a tk PhotoImage and update
        (configure) its window label for immediate display.
        Calls module manage.tk_image().

        Args:
            img_name: The key name used in the tkimg and img_label
                      dictionaries.
            img_array: The new cv2 processed numpy image array.

        Returns:
            None
        """

        # Use .configure to update images.
        self.tkimg[img_name] = manage.tk_image(
            image=img_array,
            scale_coef=self.slider_val['scale'].get()
        )
        self.img_label[img_name].configure(image=self.tkimg[img_name])

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the input self.cvimg['gray'] image.
        Updates contrast and brightness via alpha and beta sliders.
        Displays contrasted and redux noise images.
        Called by process_rw_and_sizes(). Calls update_image().

        Returns:
            None
        """
        # Source concepts:
        # https://docs.opencv2.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        # https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        self.cvimg['contrast'] = (
            cv2.convertScaleAbs(
                src=self.cvimg['gray'],
                alpha=self.slider_val['alpha'].get(),
                beta=self.slider_val['beta'].get(),
            )
        )

        self.update_image(img_name='contrast',
                          img_array=self.cvimg['contrast'])

    def reduce_noise(self) -> None:
        """
        Reduce noise in the contrast adjust image erode and dilate actions
        of cv2.morphologyEx operations.
        Called by process_rw_and_sizes(). Calls update_image().

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
            self.update_image(img_name='redux',
                              img_array=self.cvimg['contrast'])
            return

        # Need integers for the cv function parameters.
        morph_shape = const.CV_MORPH_SHAPE[self.cbox_val['morphshape'].get()]
        morph_op = const.CV_MORPHOP[self.cbox_val['morphop'].get()]
        border_type = cv2.BORDER_DEFAULT  # const.CV_BORDER[self.cbox_val['border'].get()]

        # See: https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/filtering.html
        #  on page, see: cv2.getStructuringElement(shape, ksize[, anchor])
        # see: https://docs.opencv2.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        element = cv2.getStructuringElement(
            shape=morph_shape,
            ksize=(noise_k, noise_k))

        # Use morphologyEx as a shortcut for erosion followed by dilation.
        #   MORPH_OPEN is useful to remove noise and small features.
        #   MORPH_HITMISS helps to separate close objects by shrinking them.
        # Read https://docs.opencv2.org/3.4/db/df6/tutorial_erosion_dilatation.html
        # https://theailearner.com/tag/cv-morphologyex/
        # Note that the self attribution here is to prevent garbage collection.
        self.cvimg['redux'] = cv2.morphologyEx(
            src=self.cvimg['contrast'],
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=border_type,
        )

        self.update_image(img_name='redux',
                          img_array=self.cvimg['redux'])

    def filter_image(self) -> None:
        """
        Applies a filter selection to blur the reduced noise image
        to prepare for threshold segmentation. Can also serve as a
        specialized noise reduction step.
        Called from randomwalk_segmentation() and process_rw_and_sizes().
        Calls update_image().

        Returns:
            None
        """

        filter_selected = self.cbox_val['filter'].get()
        border_type = cv2.BORDER_DEFAULT
        noise_iter = self.slider_val['noise_iter'].get()

        _k = self.slider_val['filter_k'].get()

        # If filter kernel slider and noise iteration are both set to 0,
        # then proceed without filtering and use the contrasted image.
        if _k == 0 and noise_iter == 0:
            self.update_image(img_name='filter',
                              img_array=self.cvimg['contrast'])
            return

        # If filter kernel slider is set to 0, then proceed without,
        # filtering and use the reduced noise image.
        if _k == 0:
            self.update_image(img_name='filter',
                              img_array=self.cvimg['redux'])
            return

        # Need to filter the contrasted image when noise reduction is
        #  not applied.
        if noise_iter == 0:
            image2filter = self.cvimg['contrast']
        else:
            image2filter = self.cvimg['redux']

        # cv2.GaussianBlur and cv2.medianBlur need to have odd kernels,
        #   but cv2.blur and cv2.bilateralFilter will shift image between
        #   even and odd kernels, so just make it odd for everything.
        filter_k = _k + 1 if _k % 2 == 0 else _k

        # Apply a filter to blur edges:
        # Bilateral parameters:
        # https://docs.opencv2.org/3.4/d4/d86/group__imgproc__filter.html
        # from doc: Sigma values: For simplicity, you can set the 2 sigma
        #  values to be the same. If they are small (< 10), the filter
        #  will not have much effect, whereas if they are large (> 150),
        #  they will have a very strong effect, making the image look "cartoonish".
        # NOTE: The larger the sigma the greater the effect of kernel size d.
        # NOTE: filtered image dtype is uint8

        if filter_selected == 'cv2.bilateralFilter':
            self.cvimg['filter'] = cv2.bilateralFilter(
                src=image2filter,
                # d=-1 or 0, is very CPU intensive.
                d=filter_k,
                sigmaColor=19,
                sigmaSpace=19,
                borderType=border_type)

        # Gaussian parameters:
        # see: https://theailearner.com/tag/cv-gaussianblur/
        # see: https://dsp.stackexchange.com/questions/32273/
        #  how-to-get-rid-of-ripples-from-a-gradient-image-of-a-smoothed-image
        elif filter_selected == 'cv2.GaussianBlur':
            self.cvimg['filter'] = cv2.GaussianBlur(
                src=image2filter,
                ksize=(filter_k, filter_k),
                sigmaX=0,
                sigmaY=0,
                borderType=border_type)
        elif filter_selected == 'cv2.medianBlur':
            self.cvimg['filter'] = cv2.medianBlur(
                src=image2filter,
                ksize=filter_k)
        elif filter_selected == 'cv2.blur':
            self.cvimg['filter'] = cv2.blur(
                src=image2filter,
                ksize=(filter_k, filter_k),
                borderType=border_type)

        self.update_image(img_name='filter',
                          img_array=self.cvimg['filter'])

    def th_and_dist_trans(self) -> None:
        """
        Produces a threshold image from the filtered image. This image
        is used for masking in randomwalk_segmentation(). It is separate
        here so that its display can be updated independently of running
        randomwalk_segmentation().
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
        # Convert values above thresh to a maxval of 255, white.
        # The thresh parameter is determined automatically (0 is placeholder).
        # Need to use type *_INVERSE for black on white images.
        if filter_k == 0 and noise_iter == 0:
            image2threshold = self.cvimg['contrast']
        elif filter_k == 0:
            image2threshold = self.cvimg['redux']
        else:
            image2threshold = self.cvimg['filter']

        _, self.cvimg['thresh'] = cv2.threshold(src=image2threshold,
                                                thresh=0,
                                                maxval=255,
                                                type=th_type)

        # Calculate the distance transform of the input, by replacing each
        #   foreground (non-zero) element, with its shortest distance to
        #   the background (any zero-valued element).
        #   Returns a float64 ndarray.
        # Note that maskSize=0 calculates the precise mask size only for
        #   cv2.DIST_L2. cv2.DIST_L1 and cv2.DIST_C always use maskSize=3.
        self.cvimg['dist_trans']: np.ndarray = cv2.distanceTransform(
            src=self.cvimg['thresh'],
            distanceType=dt_type,
            maskSize=mask_size)

        self.update_image(img_name='thresh',
                          img_array=self.cvimg['thresh'])
        self.update_image(img_name='dist_trans',
                          img_array=np.uint8(self.cvimg['dist_trans']))

    @property
    def randomwalk_segmentation(self) -> list:
        """
        Segment objects with skimage.feature.peak_local_max() and
        skimage.segmentation.random_walker().
        Called as arg for select_and_size() from process_rw_and_sizes().
        Calls update_image().

        Returns:
            The contour pointset list from parallel.MultiProc(rw_img).pool_it
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
        #   in ImageViewer.select_and_size().
        local_max: ndimage = peak_local_max(image=self.cvimg['dist_trans'],
                                            min_distance=min_dist,
                                            exclude_border=False, # True is min_dist
                                            num_peaks=np.inf,
                                            footprint=plm_kernel,
                                            labels=self.cvimg['thresh'],
                                            num_peaks_per_label=np.inf,
                                            p_norm=np.inf)  # Chebyshev distance
                                            # p_norm=2,  # Euclidean distance

        mask = np.zeros(shape=self.cvimg['dist_trans'].shape, dtype=bool)
        # Set background to True (not zero: True or 1)
        mask[tuple(local_max.T)] = True

        # Note that markers are single px, colored in grayscale by their label index?
        labeled_array, self.num_dt_segments = ndimage.label(input=mask)

        # Source: http://scipy-lectures.org/packages/scikit-image/index.html
        # From the doc: labels: array of ints, of same shape as data without channels dimension.
        #  Array of seed markers labeled with different positive integers for
        #  different phases. Zero-labeled pixels are unlabeled pixels.
        #  Negative labels correspond to inactive pixels that are not taken into
        #  account (they are removed from the graph).

        # Replace thresh_img background with -1 to ignore those pixels.
        labeled_array[labeled_array == self.cvimg['thresh']] = -1

        if not self.first_run:
            _info = ('\nFound peaks from distance transform.\n'
                     'Running random walker algorithm, please wait...\n\n')
            self.info_label.config(fg=const.COLORS_TK['blue'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)

        # NOTE: beta and tolerances were empirically determined for best
        #  performance with the sample images running an Intel i9600k @ 4.8 GHz.
        #  Default beta & tol take ~8x longer to process for similar results.
        rw_img = random_walker(data=self.cvimg['thresh'],
                               labels=labeled_array,
                               beta=5, # default: 130,
                               mode='cg_mg',  # Need pyamg installed. Default: 'cg_j'.
                               tol=0.1, # default: 1.e-3
                               copy=True,
                               return_full_prob=False,
                               spacing=None,
                               prob_tol=0.1,  # default: 1.e-3
                               channel_axis=None)

        if not self.first_run:
            _info = '\nRandom walker completed. Finding contours for sizing...\n\n\n'
            self.info_label.config(fg=const.COLORS_TK['blue'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)

        # self.randomwalk_contours is used in select_and_size() to draw
        #   enclosing circles and calculate sizes of ws objects.
        # Note: This for loop is much more stable, and in most cases faster,
        #  than using parallelization methods.
        self.randomwalk_contours.clear()
        for label in np.unique(ar=rw_img):

            # If the label is zero, we are examining the 'background',
            #   so simply ignore it.
            if label == 0:
                continue

            # ...otherwise, allocate memory for the label region and draw
            #   it on the mask.
            mask = np.zeros(shape=rw_img.shape, dtype="uint8")
            mask[rw_img == label] = 255

            # Detect contours in the mask and grab the largest one.
            contours, _ = cv2.findContours(image=mask.copy(),
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            # Add to the list used to draw circles around WS contours.
            self.randomwalk_contours.append(max(contours, key=cv2.contourArea))

        return self.randomwalk_contours


class ImageViewer(ProcessImage):
    """
    A suite of methods to display cv contours based on chosen settings
    and parameters as applied in ProcessImage().
    Methods:
    manage_main_win
    setup_start_window
    start_now
    setup_image_windows
    configure_main_window
    show_info_messages
    setup_buttons
    config_sliders
    config_comboboxes
    config_entries
    config_annotations
    grid_widgets
    grid_img_labels
    display_input_and_others
    set_defaults
    set_size_std
    select_and_size
    report_results
    process_rw_and_sizes
    process_sizes
    """

    def __init__(self):
        super().__init__()

        self.contour_report_frame = tk.Frame()
        self.contour_selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        self.do_inverse_th = tk.BooleanVar()

        # Note: The matching control variable attributes for the
        #   following selector widgets are in ProcessImage __init__.
        self.slider = {
            'alpha': tk.Scale(master=self.contour_selectors_frame),
            'alpha_lbl': tk.Label(master=self.contour_selectors_frame),

            'beta': tk.Scale(master=self.contour_selectors_frame),
            'beta_lbl': tk.Label(master=self.contour_selectors_frame),

            'noise_k': tk.Scale(master=self.contour_selectors_frame),
            'noise_k_lbl': tk.Label(master=self.contour_selectors_frame),

            'noise_iter': tk.Scale(master=self.contour_selectors_frame),
            'noise_iter_lbl': tk.Label(master=self.contour_selectors_frame),

            'filter_k': tk.Scale(master=self.contour_selectors_frame),
            'filter_k_lbl': tk.Label(master=self.contour_selectors_frame),

            'plm_mindist': tk.Scale(master=self.contour_selectors_frame),
            'plm_mindist_lbl': tk.Label(master=self.contour_selectors_frame),

            'plm_footprint': tk.Scale(master=self.contour_selectors_frame),
            'plm_footprint_lbl': tk.Label(master=self.contour_selectors_frame),

            'circle_r_min': tk.Scale(master=self.contour_selectors_frame),
            'circle_r_min_lbl': tk.Label(master=self.contour_selectors_frame),

            'circle_r_max': tk.Scale(master=self.contour_selectors_frame),
            'circle_r_max_lbl': tk.Label(master=self.contour_selectors_frame),
        }

        self.cbox = {
            'morphop': ttk.Combobox(master=self.contour_selectors_frame),
            'morphop_lbl': tk.Label(master=self.contour_selectors_frame),

            'morphshape': ttk.Combobox(master=self.contour_selectors_frame),
            'morphshape_lbl': tk.Label(master=self.contour_selectors_frame),

            'filter': ttk.Combobox(master=self.contour_selectors_frame),
            'filter_lbl': tk.Label(master=self.contour_selectors_frame),

            'th_type': ttk.Combobox(master=self.contour_selectors_frame),
            'th_type_lbl': tk.Label(master=self.contour_selectors_frame),

            'dt_type': ttk.Combobox(master=self.contour_selectors_frame),
            'dt_type_lbl': tk.Label(master=self.contour_selectors_frame),

            'dt_mask_size': ttk.Combobox(master=self.contour_selectors_frame),
            'dt_mask_size_lbl': tk.Label(master=self.contour_selectors_frame),

            # for size standards
            'size_std_lbl': tk.Label(master=self.contour_selectors_frame),
            'size_std': ttk.Combobox(master=self.contour_selectors_frame),
        }

        # User-entered pixel diameters of selected size standards.
        self.size_std = {
            'px_entry': tk.Entry(self.contour_selectors_frame),
            'px_val': tk.StringVar(),
            'px_lbl': tk.Label(self.contour_selectors_frame),

            'custom_entry': tk.Entry(self.contour_selectors_frame),
            'custom_val': tk.StringVar(),
            'custom_lbl': tk.Label(self.contour_selectors_frame),
        }

        self.button = {
            'reset': ttk.Button(),
            'save': ttk.Button(),
            'process': ttk.Button(),
        }

        # Dictionary items are populated in setup_image_windows(), with
        #   tk.Toplevel as values; don't want tk windows created here.
        self.img_window: dict = {}

        # Used to reset values that user may have tried to change during
        #  prolonged processing times.
        self.slider_values: list = []

        # Is an instance attribute here only because it is used in call
        #  to utils.save_settings_and_img() from the Save button.
        self.report_txt: str = ''

        # Manage the starting windows, grab the input and run settings,
        #  then proceed with image processing and sizing.
        # This order of events allows macOS implementation to flow well.
        self.manage_main_window()
        self.input_file: str = ''
        self.setup_start_window()

    def manage_main_window(self):
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

        # Make geometry offset a function of the screen width.
        #  This is needed b/c of the way different platforms' window
        #  managers position windows.
        w_offset = int(self.winfo_screenwidth() * 0.55)
        self.geometry(f'+{w_offset}+0')
        self.resizable(width=True, height=False)

        # Need to provide exit info msg to Terminal.
        self.protocol(name='WM_DELETE_WINDOW',
                      func=lambda: utils.quit_gui(app))

        self.bind('<Escape>', func=lambda _: utils.quit_gui(app))
        self.bind('<Control-q>', func=lambda _: utils.quit_gui(app))
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
            # Provide some info to user for why the start screen appears
            #  frozen when processing larger images.
            process_btn_txt.set('Processing started, wait...')
            start_win.config(cursor='watch')
            self.start_now()
            start_win.destroy()
            return event

        # Window basics:
        # Open with a temporary, instructional title.
        start_win = tk.Toplevel()
        start_win.title('First, select an image file')
        start_win.minsize(width=500, height=165)
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
                           func=lambda: utils.quit_gui(app))

        start_win.bind('<Escape>', lambda _: utils.quit_gui(app))
        start_win.bind('<Control-q>', lambda _: utils.quit_gui(app))
        start_win.bind('<Return>', func=_call_start)
        start_win.bind('<KP_Enter>', func=_call_start)

        # Take a break in configuring the window to grab the input.
        # For macOS: Need to have the filedialog be a child of
        #   start_win and need update() here.
        self.update()
        self.input_file = filedialog.askopenfilename(
            parent=start_win,
            title='Select input image',
            filetypes=[('JPG', '*.jpg'),
                       ('JPG', '*.jpeg'),
                       ('JPG', '*.JPG'),  # used for iPhone images
                       ('PNG', '*.png'),
                       ('TIFF', '*.tiff'),
                       ('TIFF', '*.tif'),
                       ('All', '*.*')],
        )

        if self.input_file:
            self.cvimg['input'] = cv2.imread(self.input_file)
            self.cvimg['gray'] = cv2.cvtColor(src=self.cvimg['input'],
                                              code=cv2.COLOR_RGBA2GRAY)
            self.metrics = manage.input_metrics(self.cvimg['input'])
        else:  # User has closed the filedialog window instead of selecting a file.
            utils.quit_gui(self)

        # Once a file is selected, the file dialog is removed, and the
        #  start window setup can proceed, now with its active title.
        start_win.title('Set start parameters')
        start_win.resizable(width=False, height=False)
        self.update_idletasks()

        # Window widgets:
        # Provide a header with file path and pixel dimensions.
        file_label = tk.Label(
            master=start_win,
            text=f'Image: {self.input_file}\n'
                 f'size:{self.cvimg["gray"].shape[0]}x{self.cvimg["gray"].shape[1]}',
            **const.LABEL_PARAMETERS)

        scale_label = tk.Label(master=start_win,
                               text='Scale image display to:',
                               **const.LABEL_PARAMETERS)
        scale_slider = tk.Scale(master=start_win,
                                from_=0.05, to=2,
                                resolution=0.05,
                                tickinterval=0.2,
                                variable=self.slider_val['scale'],
                                length=int(self.winfo_screenwidth() * 0.2),
                                **const.SCALE_PARAMETERS)
        self.slider_val['scale'].set(0.5)

        color_label = tk.Label(master=start_win,
                               text='Annotation font color:',
                               **const.LABEL_PARAMETERS)
        color_cbox = ttk.Combobox(master=start_win,
                                  values=list(const.COLORS_CV.keys()),
                                  textvariable=self.cbox_val['color'],
                                  width=12,
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
        inverse_no = tk.Radiobutton(start_win,
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

        # Window grid settings; sorted by row.
        padding = dict(padx=6, pady=6)

        file_label.grid(row=0, column=0,
                        **padding, columnspan=2, sticky=tk.EW)

        scale_label.grid(row=1, column=0, **padding, sticky=tk.E)
        scale_slider.grid(row=1, column=1, **padding, sticky=tk.W)

        color_label.grid(row=2, column=0, **padding, sticky=tk.E)
        color_cbox.grid(row=2, column=1, **padding, sticky=tk.W)

        inverse_label.grid(row=3, column=0, **padding, sticky=tk.E)
        inverse_no.grid(row=3, column=1, **padding, sticky=tk.W)
        inverse_yes.grid(row=3, column=1, padx=(50, 0), sticky=tk.W)
        process_now_button.grid(row=3, column=1, **padding, sticky=tk.E)

        # Create menu instance and add pull-down menus.
        menubar = tk.Menu(master=start_win, )
        start_win.config(menu=menubar)

        os_accelerator = 'Command' if const.MY_OS == 'dar' else 'Ctrl'
        file = tk.Menu(self.master, tearoff=0)
        menubar.add_cascade(label=utils.program_name(), menu=file)
        file.add_command(label='Process now',
                         command=_call_start,
                         accelerator='Return') # macOS doesn't recognize 'Enter'
        file.add_command(label='Quit',
                         command=lambda: utils.quit_gui(app),
                         # macOS doesn't recognize 'Command+Q' as an accelerator
                         #   b/c cannot override that system's native Command-Q,
                         accelerator=f'{os_accelerator}+Q')

        help_menu = tk.Menu(master=start_win, tearoff=0)
        tips = tk.Menu(master=start_win, tearoff=0)
        menubar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_cascade(label='Tips...', menu=tips)
        # Bullet symbol from https://coolsymbol.com/, unicode_escape: u'\u2022'
        tips.add_command(label='• Larger image files need a smaller scale factor')
        tips.add_command(label='     to fit image windows on the screen.')
        tips.add_command(label='• Use a lighter font color with darker objects.')
        tips.add_command(label='• Use the INVERSE threshold type for dark')
        tips.add_command(label='     objects on a light background.')
        tips.add_command(label='• Enter or Return key also starts processing.')
        tips.add_command(label='• More Tips are in the README file.')
        tips.add_command(label='• Esc or Ctrl-Q from any window exits the program.')
        help_menu.add_command(label='About',
                              command=lambda: utils.about_win(parent=start_win))

    def start_now(self) -> None:
        """
        Initiate the processing pipeline by setting up and configuring
        all settings widgets.
        Called from setup_start_window().
        Returns:
            None
        """

        # This calling sequence produces a slight delay (longer for larger files)
        #  before anything is displayed, but assures that everything displays
        #  simultaneously for a visually cleaner start.
        self.setup_image_windows()
        self.configure_main_window()
        self.show_info_messages()
        self.setup_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_entries()
        self.config_annotations()
        self.set_defaults()
        self.grid_widgets()
        self.grid_img_labels()
        # Place preprocess(), process_rw_and_sizes() and display_windows(),
        # in this sequence, to be called last for best performance.
        self.preprocess()
        self.process_rw_and_sizes()
        self.display_windows()

    def setup_image_windows(self) -> None:
        """
        Create and configure all Toplevel windows and their Labels that
        are used to display and update processed images.

        Returns:
            None
        """

        def _window_info():
            """
            Provide a notice in report (mainloop, app) window.
            Called locally from .protocol().
            """
            _info = ('\nThat window cannot be closed from its window bar.\n'
                    'Minimize it if it is in the way.\n'
                    'Esc or Ctrl-Q keys will Quit the program.\n')
            self.info_label.config(fg=const.COLORS_TK['vermilion'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)
            # Give user time to read the message before resetting it.
            app.after(ms=4444, func=self.show_info_messages)

        # NOTE: keys here must match corresponding keys in const.WIN_NAME.
        # Dictionary item order determines stack order of windows.
        self.img_window = {
            'input': tk.Toplevel(),
            'contrast': tk.Toplevel(),
            'filter': tk.Toplevel(),
            'dist_trans': tk.Toplevel(),
            'sized': tk.Toplevel(),
        }

        # Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods
        #  via ProcessImage.update_image().
        #  Labels are gridded in their respective img_window in grid_img_labels().
        self.img_label = {
            'input': tk.Label(self.img_window['input']),
            'gray': tk.Label(self.img_window['input']),

            'contrast': tk.Label(self.img_window['contrast']),
            'redux': tk.Label(self.img_window['contrast']),

            'filter': tk.Label(self.img_window['filter']),
            'thresh': tk.Label(self.img_window['filter']),

            'dist_trans': tk.Label(self.img_window['dist_trans']),

            'sized': tk.Label(self.img_window['sized']),
        }

        # Need an image to replace blank tk desktop icon for each img window.
        #  Set correct path to the local 'images' directory and icon file.
        # Withdraw all windows here for clean transition; all are deiconified
        #  in display_windows().
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        # Allow image label panels in image windows to resize with window.
        #  Note that images don't proportionally resize, just their boundaries;
        #  images will remain anchored at their top left corners.
        # Configure windows the same as the settings window, to give a yellow
        #  border when it has focus and light grey when being dragged.
        icon_path = None
        try:
            #  If the icon file is not present, a Terminal msg will display from
            #  <if __name__ == "__main__"> at startup.
            icon_path = tk.PhotoImage(file=utils.valid_path_to('image/sizeit_icon_512.png'))
            app.iconphoto(True, icon_path)
        except tk.TclError as _msg:
            pass

        # Because sharing the constants.py module with size_it.py, need to
        #  rename the window that does not display watershed segments
        #  used in size_it.py.
        # Random_walker segments display as an inverse threshold img, so
        #   there is no need to show them alongside, as done in size_it.
        const.WIN_NAME['dist_trans'] = 'Distances transformed'

        for _name, toplevel in self.img_window.items():
            toplevel.wm_withdraw()
            if icon_path:
                toplevel.iconphoto(True, icon_path)
            toplevel.minsize(width=200, height=100)
            toplevel.protocol(name='WM_DELETE_WINDOW', func=_window_info)
            toplevel.columnconfigure(index=0, weight=1)
            toplevel.columnconfigure(index=1, weight=1)
            toplevel.rowconfigure(index=0, weight=1)
            toplevel.title(const.WIN_NAME[_name])
            toplevel.config(bg=const.MASTER_BG,
                            highlightthickness=5,
                            highlightcolor=const.COLORS_TK['yellow'],
                            highlightbackground=const.DRAG_GRAY)
            toplevel.bind('<Escape>', func=lambda _: utils.quit_gui(app))
            toplevel.bind('<Control-q>', func=lambda _: utils.quit_gui(app))

    def configure_main_window(self) -> None:
        """
        Settings and report window (mainloop, "app") keybindings,
        configurations, and grids for contour settings and reporting frames.

        Returns:
            None
        """

        # Color in the main (app) window and give it a yellow border;
        #   border highlightcolor changes to grey with loss of focus.
        app.config(
            bg=const.MASTER_BG,
            # bg=const.COLORS_TK['sky blue'],  # for development
            highlightthickness=5,
            highlightcolor=const.COLORS_TK['yellow'],
            highlightbackground=const.DRAG_GRAY,
        )

        # Default Frame() arguments work fine to display report text.
        # bg won't show when grid sticky EW for tk.Text; see utils.display_report().
        self.contour_selectors_frame.configure(relief='raised',
                                               bg=const.DARK_BG,
                                               # bg=const.COLORS_TK['sky blue'],  # for development
                                               borderwidth=5)

        # Config columns to allow only sliders to expand with window.
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.contour_report_frame.columnconfigure(0, weight=1)
        self.contour_selectors_frame.columnconfigure(1, weight=2)


        self.contour_report_frame.grid(column=0, row=0,
                                       columnspan=2,
                                       padx=(5, 5), pady=(5, 5),
                                       sticky=tk.EW)
        self.contour_selectors_frame.grid(column=0, row=1,
                                          columnspan=2,
                                          padx=5, pady=(0, 5),
                                          ipadx=4, ipady=4,
                                          sticky=tk.EW)

        # Note: the settings window (mainloop, app) is deiconified in
        #  display_windows() after all image windows so that it stacks
        #  on top at startup.

    def show_info_messages(self) -> None:
        """
        Informative note at bottom of settings (mainloop) window about
        the displayed size units. The Label text is also re-configured
        to display other informational messages.
        Called from __init__, but label is conditionally reconfigured in
        PI.randomwalk_segmentation()

        Returns:
            None
        """

        _info = ('When the entered pixel size is 1 and selected size standard\n'
                 'is None, displayed sizes are pixels.\n'
                 'Size units are millimeters for any pre-set size standard,\n'
                 'and whatever you want for custom standards.\n'
                 f'(Processing time elapsed: {self.elapsed})')

        self.info_label.config(text=_info,
                               font=const.WIDGET_FONT,
                               bg=const.MASTER_BG,
                               fg='black')
        self.info_label.grid(column=1, row=2, rowspan=4,
                             padx=0, sticky=tk.EW)

    def setup_buttons(self) -> None:
        """
        Assign and grid Buttons in the settings (mainloop) window.
        Called from __init__.

        Returns:
            None
        """
        manage.ttk_styles(mainloop=self)

        def _save_results():
            """
            A Button kw "command" caller to avoid messy lambda statements.
            """
            _sizes = ', '.join(str(i) for i in self.sorted_size_list)
            utils.save_settings_and_img(
                input_path=self.input_file,
                img2save=self.cvimg['sized'],
                txt2save=self.report_txt + _sizes,
                caller=utils.program_name())

            _folder = str(Path(self.input_file).parent)
            _info = ('\nSettings report and result image have been saved to:\n'
                     f'{utils.valid_path_to(_folder)}\n\n')
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)
            app.after(5000, self.show_info_messages)

        def _do_reset():
            """
            Separates setting default values from lengthy process calls,
            thus shortening response time.
            """
            self.slider_values.clear()
            self.set_defaults()
            self.widget_control('off')  # is turned 'on' in preprocess().
            self.preprocess()

            _info = ('\nClick "Run Random Walker" to update counts and sizes\n'
                     'with default settings.\n\n')
            self.info_label.config(fg=const.COLORS_TK['blue'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)

        button_params = dict(
            style='My.TButton',
            width=0)

        self.button['reset'].config(text='Reset settings',
                                    command=_do_reset,
                                    **button_params)

        self.button['save'].config(text='Save settings & sized image',
                                   command=_save_results,
                                   **button_params)

        self.button['process'].config(text='Run Random Walker',
                                      command=self.process_rw_and_sizes,
                                      **button_params)

        # Widget griding in the mainloop window.
        self.button['reset'].grid(column=0, row=2,
                                  padx=10,
                                  pady=5,
                                  sticky=tk.W)

        # Need to use cross-platform padding.
        process_padx = (self.button['reset'].winfo_reqwidth() + 20, 0)
        self.button['process'].grid(column=0, row=2,
                                    padx=process_padx,
                                    pady=5,
                                    sticky=tk.W)

        self.button['save'].grid(column=0, row=3,
                                padx=10,
                                pady=(0, 5),
                                sticky=tk.W)

    def config_sliders(self) -> None:
        """
        Configure arguments and mouse button bindings for all Scale
        widgets in the settings (mainloop) window.
        Called from __init__.

        Returns:
            None
        """
        # Set minimum width for the enclosing Toplevel by setting a length
        #  for a single Scale widget that is sufficient to fit everything
        #  in the Frame given current padding parameters. Need to use only
        #  for one Scale() in each Toplevel().
        scale_len = int(self.winfo_screenwidth() * 0.25)

        def _need_to_click(event=None):
            """
            Post notice when selecting peak_local_max, because plm slider
            values are used in randomwalk_segmentation(), which is called
            only from a Button().
            """
            if self.slider_val['plm_footprint'].get() == 1:
                self.info_label.config(fg=const.COLORS_TK['vermilion'])

                _info = ('\nClick "Run Random Walker" to update counts and sizes.\n'
                         'A peak_local_max footprint of 1 may take a while.\n\n')
            else:
                _info = '\n\nClick "Run Random Walker" to update counts and sizes.\n\n'
                self.info_label.config(fg=const.COLORS_TK['blue'])

            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)

            return event

        # Scale widgets that are pre-random_walker (contrast, noise,
        # filter and threshold) and size max/min are called with mouse
        # button release.
        # Peak-local-max params and select_and_size() are called with Button.
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
        self.slider['plm_mindist'].configure(from_=1, to=220,
                                             length=scale_len,
                                             tickinterval=20,
                                             variable=self.slider_val['plm_mindist'],
                                             **const.SCALE_PARAMETERS)

        self.slider['plm_footprint_lbl'].configure(text='peak_local_max footprint:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['plm_footprint'].configure(from_=1, to=40,
                                               length=scale_len,
                                               tickinterval=2,
                                               variable=self.slider_val['plm_footprint'],
                                               **const.SCALE_PARAMETERS)

        self.slider['circle_r_min_lbl'].configure(text='Circled radius size\n'
                                                       'minimum pixels:',
                                                  **const.LABEL_PARAMETERS)
        self.slider['circle_r_max_lbl'].configure(text='Circled radius size\n'
                                                       'maximum pixels:',
                                                  **const.LABEL_PARAMETERS)

        # Note: may need to adjust c_lim scaling with image size b/c
        #   large contours cannot be selected if max limit is too small.
        circle_r_min = self.metrics['max_circle_r'] // 8
        circle_r_max = self.metrics['max_circle_r']
        self.slider['circle_r_min'].configure(from_=1, to=circle_r_min,
                                              tickinterval=circle_r_min / 10,
                                              variable=self.slider_val['circle_r_min'],
                                              **const.SCALE_PARAMETERS)
        self.slider['circle_r_max'].configure(from_=1, to=circle_r_max,
                                              tickinterval=circle_r_max / 10,
                                              variable=self.slider_val['circle_r_max'],
                                              **const.SCALE_PARAMETERS)

        # To avoid processing all the intermediate values between normal
        #  slider movements, bind sliders to call functions only on
        #  left button release.
        # Most are bound to preprocess(); process_all() is initiated
        #  only with a Button(). To speed program responsiveness when
        #  changing the size range, only the sizing and reporting methods
        #  are called on mouse button release.
        # Note that the isinstance() condition doesn't improve performance,
        #  but is there for clarity's sake.
        for _name, _w in self.slider.items():
            if isinstance(_w, tk.Label):
                continue
            if 'circle_r' in _name:
                _w.bind('<ButtonRelease-1>', self.process_sizes)
            elif 'plm_' in _name:
                _w.bind('<ButtonRelease-1>', _need_to_click)
            else:  # is alpha, beta, noise_k, noise_iter, filter_k.
                _w.bind('<ButtonRelease-1>', self.preprocess)

    def config_comboboxes(self) -> None:
        """
        Configure arguments and mouse button bindings for all Comboboxes
        in the settings (mainloop) window.
        Called from __init__.

        Returns:
             None
        """

        # Different Combobox widths are needed to account for font widths
        #  and padding in different systems.
        width_correction = 2 if const.MY_OS == 'win' else 0  # is Linux or macOS

        # Combobox styles are set in manage.ttk_styles(), called in setup_buttons().
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
        self.cbox['filter'].config(textvariable=self.cbox_val['filter'],
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

        self.cbox['size_std_lbl'].config(text='Select the standard used in image:',
                                         **const.LABEL_PARAMETERS)
        self.cbox['size_std'].config(textvariable=self.cbox_val['size_std'],
                                     width=12 + width_correction,
                                     values=list(const.SIZE_STANDARDS.keys()),
                                     **const.COMBO_PARAMETERS)

        # Now bind functions to all Comboboxes.
        # Note that the isinstance() condition isn't needed for
        # performance; it just clarifies the bind intention.
        for _name, _w in self.cbox.items():
            if isinstance(_w, tk.Label):
                continue
            if 'size_' in _name:
                _w.bind('<<ComboboxSelected>>', func=self.process_sizes)
            else:  # is morphop, morphshape, filter, th_type, dt_type, dt_mask_size.
                _w.bind('<<ComboboxSelected>>', func=self.preprocess)

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for Entry widgets
        in the settings (mainloop) window.
        Called from __init__.

        Returns:
            None
        """

        self.size_std['px_entry'].config(textvariable=self.size_std['px_val'],
                                         width=6)
        self.size_std['px_lbl'].config(text='Enter px diameter of size standard:',
                                       **const.LABEL_PARAMETERS)

        self.size_std['custom_entry'].config(textvariable=self.size_std['custom_val'],
                                             width=8)
        self.size_std['custom_lbl'].config(text="Enter custom standard's size:",
                                           **const.LABEL_PARAMETERS)

        for _, _w in self.size_std.items():
            if isinstance(_w, tk.Entry):
                _w.bind('<Return>', func=self.process_sizes)
                _w.bind('<KP_Enter>', func=self.process_sizes)

    def widget_control(self, action: str) -> None:
        """
        Used to disable settings widgets when random walker is running.
        Provides a watch cursor while widgets are disabled.
        Gets Scale() values at time of disabling and resets them upon
        enabling, thus preventing user click events retained in memory
        from changing slider position post-processing.

        Args:
            action: Either 'off' to disable widgets, or 'on' to enable.
        """
        if action == 'off':
            for _name, _w in self.slider.items():
                _w.configure(state=tk.DISABLED)
                if isinstance(_w, tk.Scale):
                    self.slider_values.append(self.slider_val[_name].get())
            for _, _w in self.cbox.items():
                _w.configure(state=tk.DISABLED)
            for _, _w in self.button.items():
                _w.grid_remove()
            for _, _w in self.size_std.items():
                if not isinstance(_w, tk.StringVar):
                    _w.configure(state=tk.DISABLED)
            app.config(cursor='watch')
            app.update()
        else:  # is 'on'
            idx = 0
            for _name, _w in self.slider.items():
                _w.configure(state=tk.NORMAL)
                if self.slider_values and isinstance(_w, tk.Scale):
                    self.slider_val[_name].set(self.slider_values[idx])
                    idx += 1
            for _, _w in self.cbox.items():
                if isinstance(_w, tk.Label):
                    _w.configure(state=tk.NORMAL)
                else:
                    _w.configure(state='readonly')
            for _, _w in self.button.items():
                _w.grid()
            for _, _w in self.size_std.items():
                if not isinstance(_w, tk.StringVar):
                    _w.configure(state=tk.NORMAL)
            app.config(cursor='')
            app.update()
            self.slider_values.clear()

    def config_annotations(self) -> None:
        """
        Set key bindings to change font size and line thickness of
        annotations in the 'sized' cv2 image.

        Returns: None
        """

        def increase_font_size() -> None:
            self.metrics['font_scale'] *= 1.1
            self.select_and_size(contour_pointset=self.randomwalk_contours)

        def decrease_font_size() -> None:
            self.metrics['font_scale'] *= 0.9
            if self.metrics['font_scale'] < 0.1:
                self.metrics['font_scale'] = 0.1
            self.select_and_size(contour_pointset=self.randomwalk_contours)

        def increase_line_thickness() -> None:
            self.metrics['line_thickness'] += 1
            self.select_and_size(contour_pointset=self.randomwalk_contours)

        def decrease_line_thickness() -> None:
            self.metrics['line_thickness'] -= 1
            if self.metrics['line_thickness'] == 0:
                self.metrics['line_thickness'] = 1
            self.select_and_size(contour_pointset=self.randomwalk_contours)

        # Bindings are needed only for the settings and sized img windows,
        #  but is simpler to use bind_all() which does not depend on widget focus.
        # NOTE: On Windows, KP_* is not a recognized keysym string; works on Linux.
        #  Windows keysyms 'plus' & 'minus' are for both keyboard and keypad.
        self.bind_all('<Control-equal>', lambda _: increase_font_size())
        self.bind_all('<Control-minus>', lambda _: decrease_font_size())
        self.bind_all('<Control-KP_Subtract>', lambda _: decrease_font_size())

        self.bind_all('<Shift-Control-plus>', lambda _: increase_line_thickness())
        self.bind_all('<Shift-Control-KP_Add>', lambda _: increase_line_thickness())
        self.bind_all('<Shift-Control-underscore>', lambda _: decrease_line_thickness())

        # Need platform-specific keypad keysym.
        if const.MY_OS == 'win':
            self.bind_all('<Control-plus>', lambda _: increase_font_size())
            self.bind_all('<Shift-Control-minus>', lambda _: decrease_line_thickness())
        else:
            self.bind_all('<Control-KP_Add>', lambda _: increase_font_size())
            self.bind_all('<Shift-Control-KP_Subtract>', lambda _: decrease_line_thickness())

    def grid_widgets(self) -> None:
        """
        Developer: Grid as a method to clarify spatial relationships.
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

        # Widgets gridded in the self.contour_selectors_frame Frame.
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
        self.cbox['filter'].grid(column=1, row=6, **west_grid_params)

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
        # self.cbox['ws_connect'].grid(column=1, row=10, **east_grid_params)

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

        # Use update() because update_idletasks() doesn't always work to
        #  get the gridded widgets' correct winfo_reqwidth.
        self.update()

        # Now grid widgets with relative padx values based on widths of
        #  their corresponding partner widgets. Works across platforms.
        morphshape_padx = (0, self.cbox['morphshape'].winfo_reqwidth() + 10)
        self.cbox['morphshape_lbl'].grid(column=1, row=2,
                                         padx=morphshape_padx,
                                         **east_params_relative)

        thtype_padx = (0, self.cbox['th_type'].winfo_reqwidth() + 10)
        self.cbox['th_type_lbl'].grid(column=1, row=6,
                                      padx=thtype_padx,
                                      **east_params_relative)

        mask_lbl_padx = (self.cbox['dt_mask_size_lbl'].winfo_reqwidth() + 120, 0)
        self.cbox['dt_mask_size'].grid(column=1, row=10,
                                       padx=mask_lbl_padx,
                                       pady=(4, 0),
                                       sticky=tk.W)

        size_std_padx = (0, self.cbox['size_std'].winfo_reqwidth() + 10)
        self.cbox['size_std_lbl'].grid(column=1, row=19,
                                       padx=size_std_padx,
                                       **east_params_relative)

        custom_std_padx = (0, self.size_std['custom_entry'].winfo_reqwidth() + 10)
        self.size_std['custom_lbl'].grid(column=1, row=20,
                                         padx=custom_std_padx,
                                         **east_params_relative)
        # Remove initially; show only when Custom size is needed.
        self.size_std['custom_lbl'].grid_remove()

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

        self.img_label['contrast'].grid(**const.PANEL_LEFT)
        self.img_label['redux'].grid(**const.PANEL_RIGHT)

        self.img_label['filter'].grid(**const.PANEL_LEFT)
        self.img_label['thresh'].grid(**const.PANEL_RIGHT)

        self.img_label['dist_trans'].grid(**const.PANEL_LEFT)

        self.img_label['sized'].grid(**const.PANEL_LEFT)

    def display_windows(self) -> None:
        """
        Ready all image window for display. Show the input image in its window.
        Bind rt-click to save any displayed image.
        Called from __init__.
        Calls update_image(), utils.save_settings_and_img().

        Returns:
            None
        """

        # All image windows were withdrawn upon their creation in
        #  setup_image_windows() to keep things tidy.
        #  Now is the time to show them.
        for _, toplevel in self.img_window.items():
            toplevel.wm_deiconify()

        # Display the input image. It is static, so does not need
        #  updating, but for consistency's sake the
        #  statement structure used to display and update processed
        #  images is used here.
        # Note: here and throughout, use 'self' to scope the
        #  ImageTk.PhotoImage image in the Class, otherwise it will/may
        #  not display because of garbage collection.
        self.update_image(img_name='input',
                          img_array=self.cvimg['input'])

        def _on_click_save_img(image_name):
            """Save the current window image (Label) that was rt-clicked."""
            tkimg = self.tkimg[image_name]

            utils.save_settings_and_img(input_path=self.input_file,
                                        img2save=tkimg,
                                        txt2save='The displayed image',
                                        caller=image_name)

            # Provide user with a notice that a file was created and
            #  give user time to read the message before resetting it.
            folder = str(Path(self.input_file).parent)
            _info = (f'\nThe displayed image, "{image_name}", was saved to:\n'
                    f'{utils.valid_path_to(folder)}\n'
                    'with a timestamp.')
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)
            app.after(4444, self.show_info_messages)

        # macOS right mouse button has a different ID.
        rt_click = '<Button-3>' if const.MY_OS in 'lin, win' else '<Button-2>'

        # Do not specify the image array in this binding, but instead
        #  specify in _on_click_save_img() function so that the current image
        #  is saved. Use update() to ensure that the label image is current.
        for img_name, label in self.img_label.items():
            app.update()
            label.bind(rt_click,
                       lambda _, n=img_name: _on_click_save_img(image_name=n))

        # Now is time to show the mainloop (app) settings window that was
        #   hidden in manage_main_window().
        #   Deiconifying here stacks it on top of all windows at startup.
        self.wm_deiconify()

    def set_defaults(self) -> None:
        """
        Sets and resets selector widgets.
        Called from __init__ and "Reset" button.
        Returns:
            None
        """
        # Default settings are optimized for sample1.jpg input.

        # Set/Reset Scale widgets.
        self.slider_val['alpha'].set(1.0)
        self.slider_val['beta'].set(0)
        self.slider_val['noise_k'].set(7)
        self.slider_val['noise_iter'].set(3)
        self.slider_val['filter_k'].set(5)
        self.slider_val['plm_mindist'].set(40)
        self.slider_val['plm_footprint'].set(3)
        self.slider_val['circle_r_min'].set(8)
        self.slider_val['circle_r_max'].set(300)

        if self.do_inverse_th.get():
            self.cbox['th_type'].current(1)
            self.cbox_val['th_type'].set('cv2.THRESH_OTSU_INVERSE')
        else:
            self.cbox['th_type'].current(0)
            self.cbox_val['th_type'].set('cv2.THRESH_OTSU')

        # Set/Reset Combobox widgets.
        self.cbox['morphop'].current(0)  # 'cv2.MORPH_OPEN' == 2
        self.cbox['morphshape'].current(2)  # 'cv2.MORPH_ELLIPSE' == 2
        self.cbox['filter'].current(0)  # 'cv2.blur' == 0, cv2 default
        self.cbox['dt_type'].current(1)  # 'cv2.DIST_L2' == 2
        self.cbox['dt_mask_size'].current(1)  # '3' == cv2.DIST_MASK_3
        self.cbox['size_std'].current(0)  # 'None'

        # Set to 1 to avoid division by 0.
        self.size_std['px_val'].set('1')
        self.size_std['custom_val'].set('0.0')

    def set_size_std(self) -> None:
        """
        Assign a unit conversion factor to the observed pixel diameter
        of the chosen size standard and calculate the number of significant
        figure in any custom size entry.
        Called from process_rw_and_sizes(), process_sizes(), __init__.

        Returns:
            None
        """

        size_std_px: str = self.size_std['px_val'].get()
        custom_size: str = self.size_std['custom_val'].get()
        size_std: str = self.cbox_val['size_std'].get()
        preset_std_size: float = const.SIZE_STANDARDS[size_std]

        # Need to verify that the pixel diameter entry is a number:
        try:
            int(size_std_px)
            if int(size_std_px) <= 0:
                raise ValueError
        except ValueError:
            _m = 'Enter only integers > 0 for the pixel diameter'
            messagebox.showerror(title='Invalid entry',
                                 detail=_m)
            self.size_std['px_val'].set('1')
            size_std_px = '1'

        # For clarity, need to show the custom size Entry widget only
        #  when 'Custom' is selected.
        # Verify that entries are numbers and define self.num_sigfig.
        #  Custom sizes can be entered as integer, float, or power operator.
        # Number of significant figures is the lowest of that for the
        #  standard's size value or pixel diameter.
        if size_std == 'Custom':
            self.size_std['custom_entry'].grid()
            self.size_std['custom_lbl'].grid()

            try:
                float(custom_size)  # will raise ValueError if not a number.
                self.unit_per_px.set(float(custom_size) / int(size_std_px))
                if size_std_px == '1':
                    self.num_sigfig = utils.count_sig_fig(custom_size)
                else:
                    self.num_sigfig = min(utils.count_sig_fig(custom_size),
                                          utils.count_sig_fig(size_std_px))
            except ValueError:
                messagebox.showinfo(
                    title='Custom size',
                    detail='Enter a number.\n'
                           'Accepted types:\n'
                           '  integer: 26, 2651, 2_651\n'
                           '  decimal: 26.5, 0.265, .2\n'
                           '  exponent: 2.6e10, 2.6e-2')
                self.size_std['custom_val'].set('0.0')

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

    def select_and_size(self, contour_pointset: list) -> None:
        """
        Select object contours based on area size and position,
        draw an enclosing circle around contours, then display them
        on the input image. Objects are expected to be oblong so that
        circle diameter can represent the object's length.
        Called by process_rw_and_sizes(), process_sizes().
        Calls update_image().

        Args:
            contour_pointset: List of selected contours from
             cv2.findContours in ProcessImage.contour_rw_segments().

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

        # Set coordinate point limits to find contours along a file border.
        bottom_edge = self.cvimg['gray'].shape[0] - 1
        right_edge = self.cvimg['gray'].shape[1] - 1

        if not contour_pointset:
            utils.no_objects_found_msg()
            return

        flag = False
        for _c in contour_pointset:

            # Exclude None elements (generated by multiprocessing.Pool).
            # Exclude contours not in the specified size range.
            # Exclude contours that have a coordinate point intersecting the img edge.
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
            #  and save each object_size measurement to a list for reporting.
            ((_x, _y), _r) = cv2.minEnclosingCircle(_c)

            # Note: sizes are full-length floats.
            object_size: float = _r * 2 * self.unit_per_px.get()

            # Need to set sig. fig. to display sizes in annotated image.
            #  num_sigfig value is determined in set_size_std().
            size2display: str = to_p.to_precision(value=object_size,
                                                  precision=self.num_sigfig)

            # Convert size strings to float, assuming that individual
            #  sizes listed in the report will be used in a spreadsheet
            #  or other statistical analysis.
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

        self.update_image(img_name='sized',
                          img_array=self.cvimg['sized'])

        # Record total time to process for user's info message. Start
        #  time is set in process_rw_and_sizes(). Preprocessing time is
        #  negligible, so it is ignored.
        self.elapsed = round(time() - self.time_start, 3)

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from process_rw_and_sizes(), process_sizes(), __init__.
        Returns:
            None
        """

        # Note: recall that *_val dict are inherited from ProcessImage().
        px_w, px_h = self.cvimg['gray'].shape
        alpha: float = self.slider_val['alpha'].get()
        beta:int = self.slider_val['beta'].get()
        noise_iter: int = self.slider_val['noise_iter'].get()
        morph_op: str = self.cbox_val['morphop'].get()
        morph_shape: str = self.cbox_val['morphshape'].get()
        filter_selected: str = self.cbox_val['filter'].get()
        th_type: str = self.cbox_val['th_type'].get()
        circle_r_min: int = self.slider_val['circle_r_min'].get()
        circle_r_max: int = self.slider_val['circle_r_max'].get()
        min_dist: int = self.slider_val['plm_mindist'].get()
        dt_type: str = self.cbox_val['dt_type'].get()
        mask_size: int = int(self.cbox_val['dt_mask_size'].get())
        p_kernel: tuple = (self.slider_val['plm_footprint'].get(),
                           self.slider_val['plm_footprint'].get())

        # Only odd kernel integers are used for processing.
        _nk: int = self.slider_val['noise_k'].get()
        if noise_iter == 0:
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

        # Size units are mm for the preset size standards.
        unit = 'unknown unit' if size_std in 'None, Custom' else 'mm'

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
        space = 23
        tab = " " * space

        # Divider symbol is Box Drawings Double Horizontal from https://coolsymbol.com/
        divider = "═" * 20  # divider's unicode_escape: u'\u2550\'

        self.report_txt = (
            f'Image: {self.input_file}\nImage size: {px_w}x{px_h}\n'
            f'{"Contrast:".ljust(space)}convertScaleAbs alpha={alpha}, beta={beta}\n'
            f'{"Noise reduction:".ljust(space)}cv2.getStructuringElement ksize={noise_k},\n'
            f'{tab}cv2.getStructuringElement shape={morph_shape}\n'
            f'{tab}cv2.morphologyEx iterations={noise_iter}\n'
            f'{tab}cv2.morphologyEx op={morph_op},\n'
            f'{"Filter:".ljust(space)}{filter_selected} ksize={filter_k}\n'
            f'{"cv2.threshold:".ljust(space)}type={th_type}\n'
            f'{"cv2.distanceTransform:".ljust(space)}'
            f'distanceType={dt_type}, maskSize={mask_size}\n'
            f'skimage functions:\n'
            f'{"   peak_local_max:".ljust(space)}min_distance={min_dist}\n'
            f'{tab}footprint=np.ones({p_kernel}, np.uint8)\n'
            f'{divider}\n'
            f'{"# distTrans segments:".ljust(space)}{self.num_dt_segments}\n'
            f'{"Selected size range:".ljust(space)}{circle_r_min}--{circle_r_max} pixels, diameter\n'
            f'{"Selected size std.:".ljust(space)}{size_std},'
            f' {size_std_size} {unit} diameter\n'
            f'{tab}Pixel diameter entered: {self.size_std["px_val"].get()},'
            f' unit/px factor: {unit_per_px}\n'
            f'{"# Selected objects:".ljust(space)}{num_selected}\n'
            f'{"Object size metrics,".ljust(space)}mean: {mean_unit_dia}, median:'
            f' {median_unit_dia}, range: {size_range}\n'
        )

        utils.display_report(frame=self.contour_report_frame,
                             report=self.report_txt)

    def preprocess(self, event=None) -> None:
        """
        Run processing functions prior to randomwalk_segmentation() to
        allow calling them and updating their images independently of the
        lengthy processing time of randomwalk_segmentation().
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
        self.set_size_std()
        self.report_results()

        # Var first_run is reset to False in process_rw_and_sizes()
        #   during the first run.
        if not self.first_run:
            _info = ('\nPreprocessing completed.\n'
                     'Click "Run Random Walker" to update counts and sizes.\n\n')
            self.info_label.config(fg=const.COLORS_TK['blue'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)

        return event

    def process_rw_and_sizes(self, event=None) -> None:
        """
        Runs all image processing methods from ProcessImage(), plus
        sizing and reporting methods.

        Args:
            event: The implicit mouse button event.

        Returns:
            *event* as a formality; is functionally None.
        """

        self.widget_control('off')
        self.time_start: float = time()
        self.select_and_size(contour_pointset=self.randomwalk_segmentation)
        self.report_results()
        self.widget_control('on')

        # Here, at the end of the processing pipeline, is where the
        #  first_run flag is set to False.
        # self.elapsed is set at end of select_and_size().
        if self.first_run:
            self.first_run = False
            _info = (f'Time to process image: {self.elapsed}\n'
                     'Default settings were used. Settings that increase or\n'
                     'decrease number of detected objects will respectively\n'
                     'increase or decrease the processing time.\n')
            self.info_label.config(fg=const.COLORS_TK['blue'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)
            app.after(ms=4444, func=self.show_info_messages)
        else:
            _info = ('\nContours found and sizes calculated. Report updated.\n'
                     f'Processing time elapsed: {self.elapsed}\n\n')
            self.info_label.config(fg=const.COLORS_TK['blue'])
            manage.info_message(widget=self.info_label,
                                toplevel=app, infotxt=_info)

        return event

    def process_sizes(self, event=None) -> None:
        """
        Call only sizing and reporting methods to improve performance.
        Called from the circle_r_min and circle_r_max sliders.

        Args:
            event: The implicit mouse button event.

        Returns:
            *event* as a formality; is functionally None.
        """
        self.set_size_std()
        self.select_and_size(contour_pointset=self.randomwalk_contours)
        self.elapsed = 'n/a'
        self.report_results()

        _info = '\n\nNew object size range selected. Report updated.\n\n'
        self.info_label.config(fg=const.COLORS_TK['blue'])
        manage.info_message(widget=self.info_label,
                            toplevel=app, infotxt=_info)
        app.after(3333, self.show_info_messages)

        return event


if __name__ == "__main__":
    # Program exits here if any of the module checks fail or if the
    #   argument --about is used, which prints info, then exits.
    # check_platform() also enables display scaling on Windows.
    utils.check_platform()
    vcheck.minversion('3.7')   # comment for PyInstaller
    vcheck.maxversion('3.11')  # comment for PyInstaller

    manage.arguments()  # comment for Pyinstaller

    try:
        print(f'{utils.program_name()} has launched...')
        app = ImageViewer()
        app.title(f'{utils.program_name()} Report & Settings')
        try:
            icon = tk.PhotoImage(file=utils.valid_path_to('images/sizeit_icon_512.png'))
            app.wm_iconphoto(True, icon)
        except tk.TclError as err:
            print('Cannot display program icon, so it will be blank or the tk default.')
            print(f'tk error message: {err}')
        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal command line. ***\n')
