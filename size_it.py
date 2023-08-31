#!/usr/bin/env python3
"""
A tkinter GUI for image processing to count and size objects in an
image to obtain sample size mean and range. The watershed algorithm is
used interactively by adjusting parameter values with slide bars and
pull-down menus. Adjusting contributing factors like contrast,
brightness, noise, and filter blurring is also interactive.

USAGE Example command lines, from within the count-and-size-main folder:
python3 -m size_it --about
python3 -m size_it

Windows systems may need to substitute 'python3' with 'py' or 'python'.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
settings windows, or from command line with Ctrl-C.
Save settings and the contoured image with Save button.

Requires Python3.7 or later and the packages opencv-python and numpy.
See this distribution's requirements.txt file for details.
Developed in Python 3.8-3.9.
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys

from pathlib import Path
from statistics import mean, median
from typing import List

# Local application imports.
from utility_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             to_precision as to_p)

# Third party imports.
# tkinter(Tk/Tcl) is included with most Python3 distributions,
#  but may sometimes need to be regarded as third-party.
try:
    import cv2
    import numpy as np
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from skimage.segmentation import watershed
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
    functions involved in identifying objects from an image file.

    Class methods:
    adjust_contrast
    reduce_noise
    filter_image
    watershed_segmentation
    update_image
    """

    __slots__ = (
        'cbox_val',
        'cvimg',
        'img_label',
        'num_dt_segments',
        'num_sigfig',
        'metrics',
        'slider_val',
        'sorted_size_list',
        'tkimg',
        'unit_per_px',
        'ws_max_contours',
        'tk',
    )

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
            'c_min_r': tk.IntVar(),
            'c_max_r': tk.IntVar(),
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
            'watershed': tk.PhotoImage(),
            'dist_trans': tk.PhotoImage(),
        }

        self.cvimg = {
            'input': const.STUB_ARRAY,
            'gray': const.STUB_ARRAY,
            'contrast': const.STUB_ARRAY,
            'redux': const.STUB_ARRAY,
            'filter': const.STUB_ARRAY,
            'ws_circled': const.STUB_ARRAY,
        }

        # img_label dictionary is set up in ImageViewer.setup_image_windows(),
        #  but is used in all Class methods here.
        self.img_label = {}

        # metrics dict is populated in ImageViewer.setup_start_window()
        self.metrics = {}

        self.num_dt_segments = 0
        self.ws_max_contours = []
        self.sorted_size_list = []
        self.unit_per_px = tk.DoubleVar()
        self.num_sigfig = 0

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

        Returns: None
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
        Called by process_all(). Calls manage.tk_image().

        Returns: None
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
        Called by process_all(). Calls manage.tk_image().

        Returns: None
        """

        # Need (sort of) kernel to be odd, to avoid an annoying shift of
        #   the displayed image.
        _k = self.slider_val['noise_k'].get()
        noise_k = _k + 1 if _k % 2 == 0 else _k

        # Need integers for the cv function parameters.
        morph_shape = const.CV_MORPH_SHAPE[self.cbox_val['morphshape'].get()]
        morph_op = const.CV_MORPHOP[self.cbox_val['morphop'].get()]
        border_type = cv2.BORDER_DEFAULT  # const.CV_BORDER[self.cbox_val['border'].get()]
        iteration = self.slider_val['noise_iter'].get()

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
            borderType=border_type
        )

        self.update_image(img_name='redux',
                          img_array=self.cvimg['redux'])


    def filter_image(self) -> None:
        """
        Applies a filter selection to blur the reduced noise image
        to prepare for threshold segmentation. Can also serve as a
        specialized noise reduction step.
        Called from watershed_segmentation() and process_all().
        Calls manage.tk_image().

        Returns: None
        """

        filter_selected = self.cbox_val['filter'].get()
        border_type = cv2.BORDER_DEFAULT

        # cv2.GaussianBlur and cv2.medianBlur need to have odd kernels,
        #   but cv2.blur and cv2.bilateralFilter will shift image between
        #   even and odd kernels, so just make it odd for everything.
        _k = self.slider_val['filter_k'].get()
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
            self.cvimg['filter'] = cv2.bilateralFilter(src=self.cvimg['redux'],
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
            self.cvimg['filter'] = cv2.GaussianBlur(src=self.cvimg['redux'],
                                            ksize=(filter_k, filter_k),
                                            sigmaX=0,
                                            sigmaY=0,
                                            borderType=border_type)
        elif filter_selected == 'cv2.medianBlur':
            self.cvimg['filter'] = cv2.medianBlur(src=self.cvimg['redux'],
                                          ksize=filter_k)
        elif filter_selected == 'cv2.blur':
            self.cvimg['filter'] = cv2.blur(src=self.cvimg['redux'],
                                    ksize=(filter_k, filter_k),
                                    borderType=border_type)
        else:  # there are no other choices, but include for future.
            self.cvimg['filter'] = cv2.blur(src=self.cvimg['redux'],
                                    ksize=(filter_k, filter_k),
                                    borderType=border_type)

        self.update_image(img_name='filter',
                          img_array=self.cvimg['filter'])

    def watershed_segmentation(self) -> None:
        """
        Identify object contours with cv2.threshold(), cv2.distanceTransform,
        and skimage.segmentation.watershed. Threshold types limited to
        Otsu and Triangle.
        Called by process_all(). Calls select_and_size() and manage.tk_image().
        Returns: None.
        """

        # watershed code inspiration sources:
        #   https://pyimagesearch.com/2015/11/02/watershed-opencv/
        # see also: http://scipy-lectures.org/packages/scikit-image/index.html

        connections = int(self.cbox_val['ws_connect'].get())
        th_type = const.THRESH_TYPE[self.cbox_val['th_type'].get()]
        dt_type = const.DISTANCE_TRANS_TYPE[self.cbox_val['dt_type'].get()]
        mask_size = int(self.cbox_val['dt_mask_size'].get())
        min_dist = self.slider_val['plm_mindist'].get()
        p_kernel = (self.slider_val['plm_footprint'].get(),
                    self.slider_val['plm_footprint'].get())
        plm_kernel = np.ones(p_kernel, np.uint8)

        # Note from doc: Currently, the Otsu's and Triangle methods
        #   are implemented only for 8-bit single-channel images.
        #   For other cv2.THRESH_*, thresh needs to be manually provided.
        # Convert values above thresh to a maxval of 255, white.
        _, thresh_img = cv2.threshold(src=self.cvimg['filter'],
                                      thresh=0,
                                      maxval=255,
                                      type=th_type)  # need *_INVERSE for black on white image.

        # Now we want to separate objects in image.
        # Generate the markers as local maxima of the distance to the background.
        # Calculate the distance transform of the input, by replacing each
        #   foreground (non-zero) element, with its shortest distance to
        #   the background (any zero-valued element).
        #   Returns a float64 ndarray.
        # Note that maskSize=0 calculates the precise mask size only for
        #   cv2.DIST_L2. cv2.DIST_L1 and cv2.DIST_C always use maskSize=3.
        distances_img = cv2.distanceTransform(src=thresh_img,
                                              distanceType=dt_type,
                                              maskSize=mask_size)
        print('Completed distance transform; looking for peaks...')

        # see: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html
        local_max = peak_local_max(distances_img,
                                   min_distance=min_dist,
                                   exclude_border=True,  # is =min_dist
                                   num_peaks=np.inf,
                                   footprint=plm_kernel,
                                   labels=thresh_img,
                                   num_peaks_per_label=np.inf,
                                   p_norm=np.inf,  # Chebyshev distance
                                   # p_norm=2,  # Euclidean distance
                                   )
        print('Found peaks; running skimage.segmentation watershed algorithm...')

        mask = np.zeros(distances_img.shape, dtype=bool)
        # Set background to True (not zero: True or 1)
        mask[tuple(local_max.T)] = True
        # Note that markers are single px, colored in gray series?
        labeled_array, self.num_dt_segments = ndimage.label(mask)

        # WHY minus sign? It separates objects much better than without it,
        #  minus symbol turns distances into threshold.
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_compact_watershed.html
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
        # Need watershed_line to show boundaries on displayed watershed_img contours.
        watershed_img = watershed(image=-distances_img,
                                  markers=labeled_array,
                                  connectivity=connections,  # 1, 4 or 8.
                                  mask=thresh_img,
                                  compactness=0.03,
                                  watershed_line=True)
        print('Completed watershed; now finding contours...')

        # NOTE: this cv2.watershed substitutes for the skimage implementation.
        #  The negative -image provides full-sized enclosing circles,
        #  but is not as good separation as the ~half-sized circles when
        #  left as positive.
        #  Need to add a channel to the src image for cv2.watershed to work.
        # dist3d = cv2.cvtColor(np.uint8(distances_img), cv2.COLOR_GRAY2BGR)
        # watershed_img = cv2.watershed(image=-dist3d,
        #                               markers=markers)

        self.ws_max_contours.clear()
        for label in np.unique(ar=watershed_img):

            # If the label is zero, we are examining the 'background',
            #   so simply ignore it.
            if label == 0:
                continue

            # ...otherwise, allocate memory for the label region and draw
            #   it on the mask.
            mask = np.zeros(shape=watershed_img.shape, dtype="uint8")
            mask[watershed_img == label] = 255

            # Detect contours in the mask and grab the largest one.
            contours, _ = cv2.findContours(image=mask.copy(),
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            # Grow the list used to draw circles around WS contours.
            self.ws_max_contours.append(max(contours, key=cv2.contourArea))

        # Convert from float32 to uint8 data type to make a PIL Imagetk
        #  Photoimage or find contours.
        distances_img = np.uint8(distances_img)
        watershed_gray = np.uint8(watershed_img)

        # Draw all watershed objects in 1 gray shade instead of each object
        #  decremented by 1 gray value in series; ws boundaries will be black.
        ws_contours, _ = cv2.findContours(watershed_gray,
                                          mode=cv2.RETR_EXTERNAL,
                                          method=cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(watershed_gray,
                         contours=ws_contours,
                         contourIdx=-1,  # all contours.
                         color=(120, 120, 120),  # mid-gray
                         thickness=-1,  # filled
                         lineType=cv2.LINE_AA)

        self.update_image(img_name='thresh',
                          img_array=thresh_img)
        self.update_image(img_name='dist_trans',
                          img_array=distances_img)
        self.update_image(img_name='watershed',
                          img_array=watershed_gray)

        print('Found contours. Segmentation completed.\n')

        # Now draw enclosing circles around watershed segments to get sizes.


class ImageViewer(ProcessImage):
    """
    A suite of methods to display cv contours based on chosen settings
    and parameters as applied in ProcessImage().
    Methods:
    manage_main_win
    setup_start_window
    start_now
    setup_image_windows
    setup_settings_window
    setup_explanation
    setup_buttons
    config_sliders
    config_comboboxes
    config_entries
    grid_widgets
    grid_img_labels
    display_input_and_others
    set_defaults
    set_size_std
    select_and_size
    report_results
    process_all
    process_sizes
    """

    __slots__ = (
        'cbox',
        'contour_report_frame',
        'contour_selectors_frame',
        'custom_size_entry',
        'do_inverse_th',
        'img_window',
        'input_file',
        'size_cust_entry',
        'size_cust_label',
        'size_settings_txt',
        'size_std_px',
        'size_std_px_entry',
        'size_std_px_label',
        'slider',
    )

    def __init__(self):
        super().__init__()

        self.contour_report_frame = tk.Frame()
        self.contour_selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        self.do_inverse_th = tk.StringVar()

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

            'dt_type': tk.Scale(master=self.contour_selectors_frame),
            'dt_type_lbl': tk.Label(master=self.contour_selectors_frame),

            'dt_mask_size': tk.Scale(master=self.contour_selectors_frame),
            'dt_mask_size_lbl': tk.Label(master=self.contour_selectors_frame),

            'plm_mindist': tk.Scale(master=self.contour_selectors_frame),
            'plm_mindist_lbl': tk.Label(master=self.contour_selectors_frame),

            'plm_footprint': tk.Scale(master=self.contour_selectors_frame),
            'plm_footprint_lbl': tk.Label(master=self.contour_selectors_frame),

            'c_min_r': tk.Scale(master=self.contour_selectors_frame),
            'c_min_r_lbl': tk.Label(master=self.contour_selectors_frame),

            'c_max_r': tk.Scale(master=self.contour_selectors_frame),
            'c_max_r_lbl': tk.Label(master=self.contour_selectors_frame),
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

            'ws_connect': ttk.Combobox(master=self.contour_selectors_frame),
            'ws_connect_lbl': tk.Label(master=self.contour_selectors_frame),

            # for size standards
            'size_std_lbl': tk.Label(master=self.contour_selectors_frame),
            'size_std': ttk.Combobox(master=self.contour_selectors_frame),
        }

        # User-entered pixel diameters of selected size standards.
        # There are only two, so no need to use dictionaries. (?)
        self.size_std_px_entry = tk.Entry(self.contour_selectors_frame)
        self.size_std_px = tk.StringVar()
        self.size_std_px_label = tk.Label(self.contour_selectors_frame)

        self.size_cust_entry = tk.Entry(self.contour_selectors_frame)
        self.custom_size_entry = tk.StringVar()
        self.size_cust_label = tk.Label(self.contour_selectors_frame)

        # Dictionary items are populated in setup_image_windows(), with
        #   tk.Toplevel as values; don't want tk windows created here.
        self.img_window = {}

        # Is an instance attribute here only because it is used in call
        #  to utils.save_settings_and_img() from the Save button.
        self.size_settings_txt = ''

        # Manage the starting windows, grab the input and run settings,
        #  then proceed with image processing and sizing.
        # This order of events allows macOS implementation to flow well.
        self.manage_main_win()
        self.input_file = ''
        self.setup_start_window()

    def manage_main_win(self):
        """
        For clarity, remove from view the Tk mainloop window created
        by the inherited ProcessImage() Class. But, to make window
        transitions smoother, first position it where it needs to go.
        Set this window toward the top right corner of the screen
        so that it doesn't cover up the img windows; also so that
        the bottom of the window is, hopefully, not below the bottom
        of the screen.
        """

        # Make geometry offset a function of the screen width.
        #  This is needed b/c of differences among platforms' window
        #  managers with how they place windows.
        w_offset = int(self.winfo_screenwidth() * 0.55)
        self.geometry(f'+{w_offset}+0')
        self.wm_withdraw()

    def setup_start_window(self) -> None:
        """
        Set up a basic Toplevel, then prompt for an input file, then
        proceed with configuring the window's widgets and set initial
        parameters of display scale, font color, and threshold type
        (inverse vs. not). A button will then trigger image processing
        steps. Window is destroyed once button is used.
        """

        # Need style of the ttk.Button to match main window button style.
        manage.ttk_styles(mainloop=self)

        def call_start(event=None) -> None:
            """Remove this start window, then call the suite of methods
            to get things going.
            Called from process_now_button and Return/Enter keys.
            Args:
                event: The implicit key action event, when used.
            Returns: *event* as a formality; is functionally None.
            """
            start_win.destroy()
            self.start_now()
            return event

        # Window basics:
        # Open with a temporary, instructional title.
        start_win = tk.Toplevel()
        start_win.title('First, select an image file')
        start_win.minsize(width=500, height=175)
        start_win.config(relief='raised',
                         bg=const.DARK_BG,
                         # bg=const.COLORS_TK['sky blue'],  # for development
                         highlightthickness=3,
                         highlightcolor=const.COLORS_TK['yellow'],
                         highlightbackground=const.DRAG_GRAY)
        start_win.columnconfigure(index=0, weight=1)
        start_win.columnconfigure(index=1, weight=1)

        # Need to allow complete tk mainloop shutdown with the system's
        #   window manager 'close' icon in the window bar.
        start_win.protocol(name='WM_DELETE_WINDOW',
                           func=lambda: utils.quit_gui(app))

        start_win.bind('<Escape>', lambda _: utils.quit_gui(app))
        start_win.bind('<Control-q>', lambda _: utils.quit_gui(app))
        start_win.bind('<Return>', func=call_start)
        start_win.bind('<KP_Enter>', func=call_start)

        # Take a break in configuring the window to grab the input.
        # For macOS, need to have the filedialog be a child of start_win.
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
            initialdir='images',
        )

        if self.input_file:
            self.cvimg['input'] = cv2.imread(self.input_file)
            self.cvimg['gray'] = cv2.cvtColor(self.cvimg['input'], cv2.COLOR_RGBA2GRAY)
            self.metrics = manage.input_metrics(self.cvimg['input'])
        else:
            utils.quit_gui(self)

        # Once a file is selected, the file dialog is removed, and the
        #  start window setup can proceed, now with its active title and
        #  at its full width.
        start_win.resizable(width=True, height=False)
        start_win.title('Set run settings')
        self.update_idletasks()

        # Window widgets:
        # Provide a header with file path and pixel dimensions.
        file_label = tk.Label(
            master=start_win,
            text=f'Image: {self.input_file}\n'
                 f'size:{self.cvimg["gray"].shape[0]}x{self.cvimg["gray"].shape[1]}'
                 ' (Larger files need a smaller scale factor.)',
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
                                 text='Use inverse threshold type for\n'
                                      'dark objects on light background?',
                                 **const.LABEL_PARAMETERS)
        inverse_yes = tk.Radiobutton(master=start_win,
                                     text='Yes',
                                     value='yes',
                                     variable=self.do_inverse_th,
                                     **const.RADIO_PARAMETERS)
        inverse_no = tk.Radiobutton(start_win,
                                    text='No',
                                    value='no',
                                    variable=self.do_inverse_th,
                                    ** const.RADIO_PARAMETERS)
        inverse_no.select()

        process_now_button = ttk.Button(master=start_win,
                                        text='Process now',
                                        style='My.TButton',
                                        width=0,
                                        command=call_start)

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

    def start_now(self) -> None:
        """
        Initiate the processing pipeline by setting up and configuring
        all settings widgets.
        Called from setup_start_window().
        """

        self.setup_image_windows()
        self.setup_settings_window()
        utils.wait4it(img=self.cvimg['gray'])
        self.setup_explanation()
        self.setup_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_entries()
        self.set_defaults()
        self.grid_widgets()
        self.grid_img_labels()
        self.display_image_windows()

    def setup_image_windows(self) -> None:
        """
        Create and configure all Toplevel windows and their Labels that
        are used to display and update processed images.

        Returns: None
        """

        def window_info():
            """
            Provide a notice in Terminal.
            Called from .protocol() in setup_image_windows().
            """
            print('Image windows cannot be closed from the window bar.\n'
                  'They can be minimized if they are in the way.\n'
                  'You can quit the program from the Count & Size Settings Report'
                  ' window bar or with the Esc or Ctrl-Q keys.')

        # NOTE: keys here must match corresponding keys in const.WIN_NAME.
        # Dictionary item order determines stack order of windows.
        self.img_window = {
            'input': tk.Toplevel(),
            'contrast': tk.Toplevel(),
            'filter': tk.Toplevel(),
            'dist_trans': tk.Toplevel(),
            'ws_contours': tk.Toplevel(),
        }

        # Prevent user from inadvertently resizing a window too small to use.
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        # Allow image label panels in image windows to resize with window.
        #  Note that images don't proportionally resize, just their boundaries;
        #    images will remain anchored at their top left corners.
        # Configure windows the same as the settings window, to give a yellow
        #  border when it has focus and light grey when being dragged.
        for _name, toplevel in self.img_window.items():
            toplevel.wm_withdraw()
            toplevel.minsize(width=200, height=100)
            toplevel.protocol(name='WM_DELETE_WINDOW', func=window_info)
            toplevel.columnconfigure(index=0, weight=1)
            toplevel.columnconfigure(index=1, weight=1)
            toplevel.rowconfigure(index=0, weight=1)
            toplevel.title(const.WIN_NAME[_name])
            toplevel.config(
                bg=const.MASTER_BG,
                highlightthickness=5,
                highlightcolor=const.COLORS_TK['yellow'],
                highlightbackground=const.DRAG_GRAY,
            )

        # The Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods.
        #  Labels are gridded in their respective img_window in
        #  ImageViewer.grid_img_labels().
        self.img_label = {
            'input': tk.Label(self.img_window['input']),
            'gray': tk.Label(self.img_window['input']),

            'contrast': tk.Label(self.img_window['contrast']),
            'redux': tk.Label(self.img_window['contrast']),

            'filter': tk.Label(self.img_window['filter']),
            'thresh': tk.Label(self.img_window['filter']),

            'dist_trans': tk.Label(self.img_window['dist_trans']),
            'watershed': tk.Label(self.img_window['dist_trans']),

            'ws_circled': tk.Label(self.img_window['ws_contours']),
        }

    def setup_settings_window(self) -> None:
        """
        Settings and report window (mainloop, "app") keybindings,
        configurations, and grids for contour settings and reporting frames.
        """

        app.resizable(width=True, height=False)

        # Color in all the master (app) Frame and use a yellow border;
        #   border highlightcolor changes to grey with loss of focus.
        self.config(
            bg=const.MASTER_BG,
            # bg=const.COLORS_TK['sky blue'],  # for development
            highlightthickness=5,
            highlightcolor=const.COLORS_TK['yellow'],
            highlightbackground=const.DRAG_GRAY,
        )

        # Need to provide exit info msg to Terminal.
        self.protocol(name='WM_DELETE_WINDOW',
                      func=lambda: utils.quit_gui(app))

        self.bind_all('<Escape>', lambda _: utils.quit_gui(app))
        self.bind_all('<Control-q>', lambda _: utils.quit_gui(app))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        # Default Frame() arguments work fine to display report text.
        # bg won't show when grid sticky EW for tk.Text; see utils.display_report().
        # self.contour_report_frame.configure(relief='flat')  # 'flat' is default.

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

        # Finally, show mainloop window that was withdrawn at program start
        #  in manage_main_win(). Deiconifying after everything is
        #  configured shortens the visual transition time.
        app.wm_deiconify()

    @staticmethod
    def setup_explanation() -> None:
        """
        Informative note at bottom of settings (mainloop) window about
        the displayed size units.
        Called from __init__.
        """

        explain_label = tk.Label(
            text='When the entered pixel size is 1 and the selected\n'
                 ' size standard is None, then circled diameters'
                 ' are pixels.\nDiameters are millimeters for any pre-set'
                 ' size standard,\nand whatever you want for custom '
                 'standards.',
            font=const.WIDGET_FONT,
            bg=const.MASTER_BG)
        explain_label.grid(column=1, row=2, rowspan=2,
                           padx=10, sticky=tk.EW)

    def setup_buttons(self) -> None:
        """
        Assign and grid Buttons in the settings (mainloop) window.
        Called from __init__.

        Returns: None
        """
        manage.ttk_styles(mainloop=self)

        def save_results():
            """
            A Button kw "command" caller to avoid messy lambda statements.
            """
            sizes = ', '.join(str(i) for i in self.sorted_size_list)
            utils.save_settings_and_img(
                inputpath=self.input_file,
                img2save=self.cvimg['ws_circled'],
                txt2save=self.size_settings_txt + sizes,
                caller='sizeit')

        def do_reset():
            """
            Separates setting default values from process calls during
            startup, thus shortening startup time.
            """
            self.set_defaults()
            self.process_all()

        button_params = dict(
            style='My.TButton',
            width=0)

        reset_btn = ttk.Button(text='Reset settings',
                               command=do_reset,
                               **button_params)

        save_btn = ttk.Button(text='Save settings & sized image',
                              command=save_results,
                              **button_params)

        # Widget grid in the mainloop window.
        reset_btn.grid(column=0, row=2,
                       padx=10,
                       pady=5,
                       sticky=tk.EW)
        save_btn.grid(column=0, row=3,
                      padx=10,
                      pady=(0, 5),
                      sticky=tk.EW)

    def config_sliders(self) -> None:
        """
        Configure arguments and mouse button bindings for all Scale
        widgets in the settings (mainloop) window.
        Called from __init__.

        Returns: None
        """
        # Set minimum width for the enclosing Toplevel by setting a length
        #  for a single Scale widget that is sufficient to fit everything
        #  in the Frame given current padding parameters. Need to use only
        #  for one Scale() in each Toplevel().
        scale_len = int(self.winfo_screenwidth() * 0.25)

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

        self.slider['noise_iter_lbl'].configure(text='Reduce noise, iterations:',
                                                **const.LABEL_PARAMETERS)
        self.slider['noise_iter'].configure(from_=1, to=5,
                                            tickinterval=1,
                                            variable=self.slider_val['noise_iter'],
                                            command=self.process_all,
                                            **const.SCALE_PARAMETERS)

        self.slider['filter_k_lbl'].configure(text='Filter kernel size\n'
                                                   '(only odd integers or 0 used):',
                                              **const.LABEL_PARAMETERS)
        self.slider['filter_k'].configure(from_=0, to=111,
                                          tickinterval=9,
                                          variable=self.slider_val['filter_k'],
                                          **const.SCALE_PARAMETERS)

        self.slider['plm_mindist_lbl'].configure(text='peak_local_max min_distance:',
                                                 **const.LABEL_PARAMETERS)
        self.slider['plm_mindist'].configure(from_=1, to=200,
                                             length=scale_len,
                                             tickinterval=20,
                                             variable=self.slider_val['plm_mindist'],
                                             **const.SCALE_PARAMETERS)

        self.slider['plm_footprint_lbl'].configure(text='peak_local_max footprint:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['plm_footprint'].configure(from_=1, to=20,
                                               length=scale_len,
                                               tickinterval=2,
                                               variable=self.slider_val['plm_footprint'],
                                               **const.SCALE_PARAMETERS)

        self.slider['c_min_r_lbl'].configure(text='Circled radius size\n'
                                                  'minimum pixels:',
                                             **const.LABEL_PARAMETERS)
        self.slider['c_max_r_lbl'].configure(text='Circled radius size\n'
                                                  'maximum pixels:',
                                             **const.LABEL_PARAMETERS)

        # Note: may need to adjust c_lim scaling with image size b/c
        #   large contours cannot be selected if max limit is too small.
        c_min_r = self.metrics['max_circle_r'] // 8
        c_max_r = self.metrics['max_circle_r']
        self.slider['c_min_r'].configure(from_=1, to=c_min_r,
                                         tickinterval=c_min_r / 10,
                                         variable=self.slider_val['c_min_r'],
                                         **const.SCALE_PARAMETERS)
        self.slider['c_max_r'].configure(from_=1, to=c_max_r,
                                         tickinterval=c_max_r / 10,
                                         variable=self.slider_val['c_max_r'],
                                         **const.SCALE_PARAMETERS)

        # To avoid grabbing all the intermediate values between normal
        #  click and release movement, bind sliders to call the main
        #  processing and reporting function only on left button release.
        # Most are bound to process_all(), but to speed program
        # responsiveness when changing the size range, only call the
        # sizing method to avoid image processing overhead.
        # Note that the <if '_lbl'> condition doesn't seem to be needed to
        #   improve performance, but is there for clarity's sake.
        for name, widget in self.slider.items():
            if '_lbl' not in name and 'c_lim' not in name:
                widget.bind('<ButtonRelease-1>', self.process_all)
            elif 'c_lim' in name:
                widget.bind('<ButtonRelease-1>', self.process_sizes)

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

        # Note: functions are bound to Combobox actions at the end of this method.
        #   Combobox styles are set in manage.ttk_styles(), called in setup_buttons().
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

        self.cbox['ws_connect_lbl'].config(text='Watershed connectivity:',
                                           **const.LABEL_PARAMETERS)
        self.cbox['ws_connect'].config(textvariable=self.cbox_val['ws_connect'],
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
        # Note that the  <if '_lbl'> condition doesn't seem to be needed for
        # performance; it just clarifies the bind intention.
        for name, widget in self.cbox.items():
            if '_lbl' in name:
                continue
            if 'size_' in name:
                widget.bind('<<ComboboxSelected>>', func=self.process_sizes)
            else:
                widget.bind('<<ComboboxSelected>>', func=self.process_all)

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for Entry widgets
        in the settings (mainloop) window.
        Called from __init__.

        Returns: None
        """

        self.size_std_px_entry.config(textvariable=self.size_std_px,
                                      width=6)
        self.size_std_px_label.config(text='Enter px diameter of size standard:',
                                      **const.LABEL_PARAMETERS)

        self.size_cust_entry.config(textvariable=self.custom_size_entry,
                                    width=8)
        self.size_cust_label.config(text="Enter custom standard's size:",
                                    **const.LABEL_PARAMETERS)

        self.size_std_px_entry.bind('<Return>', func=self.process_sizes)
        self.size_std_px_entry.bind('<KP_Enter>', func=self.process_sizes)

        self.size_cust_entry.bind('<Return>', func=self.process_sizes)
        self.size_cust_entry.bind('<KP_Enter>', func=self.process_sizes)

    def grid_widgets(self) -> None:
        """
        Developer: Grid as a method to clarify spatial relationships.
        Called from __init__.

        Returns: None
        """

        # Use the dict() function with keyword arguments to mimic the
        #  keyword parameter structure of the grid() function.
        east_grid_params = dict(
            padx=5,
            pady=(4, 0),
            sticky=tk.E)
        east_params_rel = dict(
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
        self.cbox['ws_connect'].grid(column=1, row=10, **east_grid_params)

        self.slider['plm_mindist_lbl'].grid(column=0, row=12, **east_grid_params)
        self.slider['plm_mindist'].grid(column=1, row=12, **slider_grid_params)

        self.slider['plm_footprint_lbl'].grid(column=0, row=13, **east_grid_params)
        self.slider['plm_footprint'].grid(column=1, row=13, **slider_grid_params)

        self.slider['c_min_r_lbl'].grid(column=0, row=17, **east_grid_params)
        self.slider['c_min_r'].grid(column=1, row=17, **slider_grid_params)

        self.slider['c_max_r_lbl'].grid(column=0, row=18, **east_grid_params)
        self.slider['c_max_r'].grid(column=1, row=18, **slider_grid_params)

        self.size_std_px_label.grid(column=0, row=19, **east_grid_params)
        self.size_std_px_entry.grid(column=1, row=19, **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['size_std'].grid(column=1, row=19, **east_grid_params)

        self.size_cust_entry.grid(column=1, row=20, **east_grid_params)

        # Use update() because update_idletasks() doesn't always work to
        #  get the gridded widgets' correct winfo_width.
        self.update()

        # Now grid widgets with relative padx values based on widths of
        #  their corresponding partner widgets. Works across platforms.
        morphshape_padx = (0, self.cbox['morphshape'].winfo_width() + 10)
        self.cbox['morphshape_lbl'].grid(column=1, row=2,
                                         padx=morphshape_padx,
                                         **east_params_rel)

        thtype_padx = (0, self.cbox['th_type'].winfo_width() + 10)
        self.cbox['th_type_lbl'].grid(column=1, row=6,
                                      padx=thtype_padx,
                                      **east_params_rel)

        mask_lbl_padx = (self.cbox['dt_mask_size_lbl'].winfo_width() + 120, 0)
        self.cbox['dt_mask_size'].grid(column=1, row=10,
                                       padx=mask_lbl_padx,
                                       pady=(4, 0),
                                       sticky=tk.W)

        ws_connect_padx = (0, self.cbox['ws_connect'].winfo_width() + 10)
        self.cbox['ws_connect_lbl'].grid(column=1, row=10,
                                         padx=ws_connect_padx,
                                         **east_params_rel)

        size_std_padx = (0, self.cbox['size_std'].winfo_width() + 10)
        self.cbox['size_std_lbl'].grid(column=1, row=19,
                                       padx=size_std_padx,
                                       **east_params_rel)

        custom_std_padx = (0, self.size_cust_entry.winfo_width() + 10)
        self.size_cust_label.grid(column=1, row=20,
                                  padx=custom_std_padx,
                                  **east_params_rel)
        # Remove initially; show only when Custom size is needed.
        self.size_cust_label.grid_remove()

    def grid_img_labels(self) -> None:
        """
        Grid all image Labels inherited from ProcessImage().
        Labels' 'master' argument for the img window is defined in
        ProcessImage.setup_image_windows(). Label 'image' param is
        updated with .configure() in each PI processing method.
        Called from __init__.

        Returns: None
        """

        self.img_label['input'].grid(**const.PANEL_LEFT)

        self.img_label['contrast'].grid(**const.PANEL_LEFT)
        self.img_label['redux'].grid(**const.PANEL_RIGHT)

        self.img_label['filter'].grid(**const.PANEL_LEFT)
        self.img_label['thresh'].grid(**const.PANEL_RIGHT)

        self.img_label['dist_trans'].grid(**const.PANEL_LEFT)
        self.img_label['watershed'].grid(**const.PANEL_RIGHT)

        self.img_label['ws_circled'].grid(**const.PANEL_RIGHT)

    def display_image_windows(self) -> None:
        """
        Show the input image in its window.
        Ready all image window for display.
        Called from __init__.
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

    def set_defaults(self) -> None:
        """
        Sets and resets selector widgets.
        Called from __init__ and "Reset" button.
        """
        # Settings are optimized for the default sample1.jpg input.

        # Set/Reset Scale widgets.
        self.slider_val['alpha'].set(1.0)
        self.slider_val['beta'].set(0)
        self.slider_val['noise_k'].set(5)
        self.slider_val['noise_iter'].set(3)
        self.slider_val['filter_k'].set(5)
        self.slider_val['plm_mindist'].set(40)
        self.slider_val['plm_footprint'].set(3)
        self.slider_val['c_min_r'].set(8)
        self.slider_val['c_max_r'].set(300)

        if self.do_inverse_th.get() == 'yes':
            self.cbox['th_type'].current(1)
            self.cbox_val['th_type'].set('cv2.THRESH_OTSU_INVERSE')
        else:
            self.cbox['th_type'].current(0)
            self.cbox_val['th_type'].set('cv2.THRESH_OTSU')

        # Set/Reset Combobox widgets.
        self.cbox['morphop'].current(0)  # 'cv2.MORPH_OPEN' == 2
        self.cbox['morphshape'].current(0)  # 'cv2.MORPH_RECT' == 0, cv2 default
        self.cbox['filter'].current(0)  # 'cv2.blur' == 0, cv2 default
        self.cbox['dt_type'].current(1)  # 'cv2.DIST_L2' == 2
        self.cbox['dt_mask_size'].current(1)  # '3' == cv2.DIST_MASK_3
        self.cbox['ws_connect'].current(1)  # '4'
        self.cbox['size_std'].current(0)  # 'None'

        # Set to 1 to avoid division by 0.
        self.size_std_px.set('1')

        self.custom_size_entry.set('0.0')

    def set_size_std(self) -> None:
        """
        Assign a unit conversion factor to the observed pixel diameter
        of the chosen size standard and calculate the number of significant
        figure in any custom size entry.
        Called from process_all(), process_sizes(), __init__.

        Returns: None
        """

        custom_size: str = self.custom_size_entry.get()
        size_std_px: str = self.size_std_px.get()
        size_std: str = self.cbox_val['size_std'].get()

        # Need to verify that the pixel diameter entry is a number:
        try:
            int(size_std_px)
            if int(size_std_px) <= 0:
                raise ValueError
        except ValueError:
            _m = 'Enter only integers > 0 for the pixel diameter'
            messagebox.showerror(title='Invalid entry',
                                 detail=_m)
            self.size_std_px.set('1')
            size_std_px = '1'

        # For clarity, need to not show the custom size Entry widgets
        #  only when 'Custom' is selected.
        # Verify that entries are numbers and define self.num_sigfig.
        #  Custom sizes can be entered as integer, float, or power operator.
        if size_std == 'Custom':
            self.size_cust_entry.grid()
            self.size_cust_label.grid()

            try:
                float(custom_size)  # will raise ValueError if not a number.
                self.unit_per_px.set(float(custom_size) / int(size_std_px))
                self.num_sigfig = utils.count_sig_fig(custom_size)
            except ValueError:
                messagebox.showinfo(
                    title='Custom size',
                    detail='Enter a number.\n'
                           'Accepted types:\n'
                           '  integer: 26, 2651, 2_651\n'
                           '  decimal: 26.5, 0.265, .2\n'
                           '  exponent: 2.6e10, 2.6e-2')
                self.custom_size_entry.set('0.0')

        else:  # is one of the preset size standards
            self.size_cust_entry.grid_remove()
            self.size_cust_label.grid_remove()
            self.custom_size_entry.set('0.0')

            preset_std_size = const.SIZE_STANDARDS[size_std]
            self.unit_per_px.set(preset_std_size / int(size_std_px))
            self.num_sigfig = utils.count_sig_fig(preset_std_size)

    def select_and_size(self, contour_pointset: list) -> None:
        """
        Select object contours based on area size and position,
        draw an enclosing circle around contours, then display them
        on the input image. Objects are expected to be oblong so that
        circle diameter can represent the object's length.
        Called by process_all(), process_sizes().
        Calls manage.tk_image().

        Args:
            contour_pointset: List of selected contours from
             cv2.findContours in ProcessImage.watershed_segmentation().

        Returns: None
        """
        # Note that cvimg['ws_circled'] is an instance attribute because
        #  it is the image also used for utils.save_settings_and_img().
        self.cvimg['ws_circled'] = self.cvimg['input'].copy()
        self.sorted_size_list.clear()

        selected_sizes: List[float] = []
        preferred_color: tuple = const.COLORS_CV[self.cbox_val['color'].get()]
        font_scale: float = self.metrics['font_scale']
        line_thickness: int = self.metrics['line_thickness']

        # The size range slider values are radii pixels. This is done b/c:
        #  1) Displayed values have fewer digits, so a cleaner slide bar.
        #  2) Sizes are diameters, so radii are conceptually easier than areas.
        #  So, need to convert to area for the cv2.contourArea function.
        c_area_min = self.slider_val['c_min_r'].get() ** 2 * np.pi
        c_area_max = self.slider_val['c_max_r'].get() ** 2 * np.pi

        # Set coordinate point limits to find contours along a file border.
        bottom_edge = self.cvimg['gray'].shape[0] - 1
        right_edge = self.cvimg['gray'].shape[1] - 1

        # Exclude contours not in the specified size range.
        # Exclude contours that have a coordinate point intersecting the img edge.
        if contour_pointset:
            flag = False
            for _c in contour_pointset:
                if not c_area_max > cv2.contourArea(_c) >= c_area_min:
                    continue

                # Skip contours that touch top or left edge.
                if {0, 1}.intersection(set(_c.ravel())):
                    continue

                # Skip contours that touch bottom or right edge.
                # Break from inner loop when any touch is found.
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
                object_size = _r * 2 * self.unit_per_px.get()

                # Need to set sig. fig. to display sizes in annotated image.
                #  num_sigfig value is determined in set_size_std().
                size2display: str = to_p.to_precision(value=object_size,
                                                      precision=self.num_sigfig)

                # Convert sizes to float on assumption that individual
                #  sizes listed in the report will be used in a spreadsheet
                #  or other statistical analysis.
                selected_sizes.append(float(size2display))

                ((txt_width, _), baseline) = cv2.getTextSize(
                    text=size2display,
                    fontFace=const.FONT_TYPE,
                    fontScale=font_scale,
                    thickness=line_thickness)
                offset_x = txt_width / 2

                cv2.circle(img=self.cvimg['ws_circled'],
                           center=(round(_x), round(_y)),
                           radius=round(_r),
                           color=preferred_color,
                           thickness=line_thickness,
                           lineType=cv2.LINE_AA,
                           )
                cv2.putText(img=self.cvimg['ws_circled'],
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
        else:
            utils.no_objects_found_msg()

        # Circled sized objects are in their own window.
        self.update_image(img_name='ws_circled',
                          img_array=self.cvimg['ws_circled'])

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from process_all(), process_sizes(), __init__.
        """

        # Note: recall that *_val dict are inherited from ProcessImage().
        px_w = self.cvimg['gray'].shape[0]
        px_h = self.cvimg['gray'].shape[1]
        alpha = self.slider_val['alpha'].get()
        beta = self.slider_val['beta'].get()
        noise_iter = self.slider_val['noise_iter'].get()
        morph_op = self.cbox_val['morphop'].get()
        morph_shape = self.cbox_val['morphshape'].get()
        filter_selected = self.cbox_val['filter'].get()
        th_type = self.cbox_val['th_type'].get()
        c_min_r = self.slider_val['c_min_r'].get()
        c_max_r = self.slider_val['c_max_r'].get()
        min_dist = self.slider_val['plm_mindist'].get()
        connections = int(self.cbox_val['ws_connect'].get())
        dt_type = self.cbox_val['dt_type'].get()
        mask_size = int(self.cbox_val['dt_mask_size'].get())
        p_kernel = (self.slider_val['plm_footprint'].get(),
                    self.slider_val['plm_footprint'].get())

        # Only odd kernel integers are used for processing.
        _nk = self.slider_val['noise_k'].get()
        noise_k = _nk + 1 if _nk % 2 == 0 else _nk

        _fk = self.slider_val['filter_k'].get()
        if _fk == 0:
            filter_k = _fk
        else:
            filter_k = _fk + 1 if _fk % 2 == 0 else _fk

        size_std = self.cbox_val['size_std'].get()
        if size_std == 'Custom':
            size_std_size = self.custom_size_entry.get()
        else:
            size_std_size = const.SIZE_STANDARDS[size_std]

        # Size units are mm for the preset size standards.
        unit = 'unknown unit' if size_std in 'None, Custom' else 'mm'

        # Work up some summary metrics.
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
        divider = "═" * 20  # divider's unicode_escape: b'\\u2550\'

        self.size_settings_txt = (
            f'Image: {self.input_file} {px_h}x{px_w}\n\n'
            f'{"Contrast:".ljust(space)}convertScaleAbs alpha={alpha}, beta={beta}\n'
            f'{"Noise reduction:".ljust(space)}cv2.getStructuringElement ksize={noise_k},\n'
            f'{tab}cv2.getStructuringElement shape={morph_shape}\n'
            f'{tab}cv2.morphologyEx iterations={noise_iter}\n'
            f'{tab}cv2.morphologyEx op={morph_op},\n'
            f'{"Filter:".ljust(space)}{filter_selected} ksize=({filter_k},{filter_k})\n'
            f'{"cv2.threshold:".ljust(space)}type={th_type}\n'
            f'{"cv2.distanceTransform:".ljust(space)}'
            f'distanceType={dt_type}, maskSize={mask_size}\n'
            f'skimage functions:\n'
            f'{"   peak_local_max:".ljust(space)}min_distance={min_dist}\n'
            f'{tab}footprint=np.ones({p_kernel}, np.uint8\n'
            f'{"   watershed:".ljust(space)}connectivity={connections}\n'
            f'{tab}compactness=0.03\n'  # NOTE: change if changed in watershed method.
            f'{divider}\n'
            f'{"# distTrans segments:".ljust(space)}{self.num_dt_segments}\n'
            f'{"Selected size range:".ljust(space)}{c_min_r}--{c_max_r} pixels, diameter\n'
            f'{"Selected size std.:".ljust(space)}{size_std},'
            f' {size_std_size} {unit} diameter\n'
            f'{tab}Pixel diameter entered: {self.size_std_px.get()},'
            f' unit/px factor: {unit_per_px}\n'
            f'{"# Selected objects:".ljust(space)}{num_selected}\n'
            f'{"Object size metrics,".ljust(space)}mean: {mean_unit_dia}, median:'
            f' {median_unit_dia}, range: {size_range}\n'
        )

        utils.display_report(frame=self.contour_report_frame,
                             report=self.size_settings_txt)

    def process_all(self, event=None) -> None:
        """
        Runs all image processing methods from ProcessImage().

        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.
        """
        self.adjust_contrast()
        self.reduce_noise()
        self.filter_image()
        self.set_size_std()
        self.watershed_segmentation()
        self.select_and_size(contour_pointset=self.ws_max_contours)
        self.report_results()

        return event

    def process_sizes(self, event=None) -> None:
        """
        Call only sizing and reporting methods to improve performance.
        Called from the c_min_r and c_max_r sliders.
        Args:
            event: The implicit mouse button event.
        Returns: *event* as a formality; is functionally None.
        """
        self.set_size_std()
        self.select_and_size(contour_pointset=self.ws_max_contours)
        self.report_results()

        return event


if __name__ == "__main__":
    # Program exits here if any of the module checks fail or if the
    #   argument --about is used, which prints info, then exits.
    utils.check_platform()
    vcheck.minversion('3.7')
    manage.arguments()

    try:
        print(f'{Path(__file__).name} has launched...')
        app = ImageViewer()
        app.title('Count & Size Settings Report')
        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal command line. ***\n')
