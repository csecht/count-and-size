#!/usr/bin/env python3
"""
A tkinter GUI for image processing to count and size objects in an
image to obtain sample size mean and range. The watershed algorithm is
used interactively by adjusting parameter values with slide bars and
pull-down menus. Adjusting contributing factors like contrast,
brightness, noise, and filter blurring is also interactive.

USAGE Example command lines, from within the count-and-size-main folder:
python3 -m size_it --help
python3 -m size_it --about
python3 -m size_it   -> default settings, is the same as:
python3 -m size_it --input images/sample1.jpg --scale 0.5 --color red
python3 -m size_it -i images/sample2.jpg -s 0.25 -c green --inverse

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

# Local application imports.
from utility_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             )

# Third party imports.
# tkinter(Tk/Tcl) is included with most Python3 distributions,
#  but may sometimes need to be regarded as third-party.
try:
    import cv2
    import numpy as np
    import tkinter as tk
    from tkinter import ttk
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


# pylint: disable=use-dict-literal, no-member

class ProcessImage(tk.Tk):
    """
    A suite of OpenCV methods for applying various image processing
    functions involved in identifying objects from an image file.

    Class methods:
    adjust_contrast
    reduce_noise
    filter_image
    watershed_segmentation
    select_and_size
    """

    __slots__ = (
        'cbox_val'
        'circled_ws_segments'
        'contrasted_img'
        'custom_size'
        'filtered_img'
        'mean_px_size'
        'mm_size_list'
        'num_dt_segments'
        'reduced_noise_img'
        'size_std'
        'size_std_px_d'
        'size_std_unit'
        'slider_val'
        'sorted_d'
        'tkimg'
        'unit_per_px'
        'ws_max_cntrs',
        'tk',
    )

    def __init__(self):
        super().__init__()

        # Note: The matching selector widgets for the following
        #  control variables are in ContourViewer __init__.
        self.slider_val = {
            # Used for contours.
            'alpha': tk.DoubleVar(),
            'beta': tk.IntVar(),
            'noise_k': tk.IntVar(),
            'noise_iter': tk.IntVar(),
            'filter_k': tk.IntVar(),
            'plm_mindist': tk.IntVar(),
            'plm_footprint': tk.IntVar(),
            'c_min_r': tk.IntVar(),
            'c_max_r': tk.IntVar(),
        }
        self.cbox_val = {
            'morphop': tk.StringVar(),
            'morphshape': tk.StringVar(),
            'filter': tk.StringVar(),
            'th_type': tk.StringVar(),
            'dt_type': tk.StringVar(),
            'dt_mask_size': tk.StringVar(),
            'ws_connect': tk.StringVar(),
            'size_std': tk.StringVar(),
            'size_custom': tk.StringVar(),
        }

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  with Label images displayed in their respective img_window Toplevel.
        self.tkimg = {
            'input': tk.PhotoImage(),
            'gray': tk.PhotoImage(),
            'contrast': tk.PhotoImage(),
            'redux': tk.PhotoImage(),
            'filter': tk.PhotoImage(),
            'watershed': tk.PhotoImage(),
            'dist_trans': tk.PhotoImage(),
            'thresh': tk.PhotoImage(),
        }

        # The image used to display final result and passed to
        #   utils.save_settings_and_img.
        self.circled_ws_segments = None

        # img_label dictionary is set up in ImageViewer.setup_image_windows(),
        #  but is used in all Class methods here.
        self.img_label = None

        self.contrasted_img = const.STUB_ARRAY
        self.reduced_noise_img = const.STUB_ARRAY
        self.filtered_img = const.STUB_ARRAY
        self.num_dt_segments = 0
        self.ws_max_cntrs = []
        self.sorted_d = []
        self.mm_size_list = []
        self.mean_px_size = 0
        self.size_std = ''
        self.size_std_unit = 0
        self.unit_per_px = tk.DoubleVar()
        self.size_std_px_d = tk.StringVar()
        self.custom_size = tk.StringVar()

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the input GRAY_IMG image.
        Updates contrast and brightness via alpha and beta sliders.
        Displays contrasted and redux noise images.
        Called by process_all(). Calls manage.tk_image().

        Returns: None
        """
        # Source concepts:
        # https://docs.opencv2.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        # https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv

        self.contrasted_img = (
            cv2.convertScaleAbs(
                src=GRAY_IMG,
                alpha=self.slider_val['alpha'].get(),
                beta=self.slider_val['beta'].get(),
            )
        )

        # Using .configure to update image avoids the white flash each time an
        #  image is updated were a Label() to be re-made here each call.
        self.tkimg['contrast'] = manage.tk_image(image=self.contrasted_img,
                                                 colorspace='bgr')
        self.img_label['contrast'].configure(image=self.tkimg['contrast'])

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
        self.reduced_noise_img = cv2.morphologyEx(
            src=self.contrasted_img,
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=border_type
        )

        self.tkimg['redux'] = manage.tk_image(image=self.reduced_noise_img,
                                              colorspace='bgr')
        self.img_label['redux'].configure(image=self.tkimg['redux'])


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

        if filter_selected == 'cv2.bilateralFilter':
            filtered_img = cv2.bilateralFilter(src=self.reduced_noise_img,
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
            filtered_img = cv2.GaussianBlur(src=self.reduced_noise_img,
                                            ksize=(filter_k, filter_k),
                                            sigmaX=0,
                                            sigmaY=0,
                                            borderType=border_type)
        elif filter_selected == 'cv2.medianBlur':
            filtered_img = cv2.medianBlur(src=self.reduced_noise_img,
                                          ksize=filter_k)
        elif filter_selected == 'cv2.blur':
            filtered_img = cv2.blur(src=self.reduced_noise_img,
                                    ksize=(filter_k, filter_k),
                                    borderType=border_type)
        else:
            filtered_img = cv2.blur(src=self.reduced_noise_img,
                                    ksize=(filter_k, filter_k),
                                    borderType=border_type)

        # NOTE: filtered_img dtype is uint8
        self.tkimg['filter'] = manage.tk_image(image=filtered_img,
                                               colorspace='bgr')
        self.img_label['filter'].configure(image=self.tkimg['filter'])

        self.filtered_img = filtered_img

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
        _, thresh_img = cv2.threshold(self.filtered_img, #self.reduced_noise_img,
                                      thresh=0,
                                      maxval=255,
                                      type=th_type)  # need *_INVERSE for black on white image.

        ################################ Distance transformation:
        # Now we want to separate the two objects in image.
        # Generate the markers as local maxima of the distance to the background.
        # Compute the exact Euclidean distance from every binary
        #   pixel to the nearest zero pixel, then find peaks in this distance map.
        # Calculate the distance transform of the input, by replacing each
        #   foreground (non-zero) element, with its shortest distance to
        #   the background (any zero-valued element).
        #   Returns a float64 ndarray.
        # Note that maskSize=0 calculates the precise mask size only for
        #   cv2.DIST_L2. cv2.DIST_L1 and cv2.DIST_C always use maskSize=3.
        distances_img = cv2.distanceTransform(src=thresh_img,
                                              distanceType=dt_type,
                                              maskSize=mask_size)

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

        # NOTE: this cv2.watershed substitutes for the skimage implementation.
        #  The negative -image provides full-sized enclosing circles,
        #  but is not as good separation as the ~half-sized circles when
        #  left as positive.
        #  Need to add a channel to the src image for cv2.watershed to work.
        # dist3d = cv2.cvtColor(np.uint8(distances_img), cv2.COLOR_GRAY2BGR)
        # watershed_img = cv2.watershed(image=-dist3d,
        #                               markers=markers)

        self.ws_max_cntrs.clear()
        for label in np.unique(ar=watershed_img):

            # If the label is zero, we are examining the 'background',
            #   so simply ignore it.
            if label == 0:
                continue

            # ...otherwise, allocate memory for the label region and draw
            #   it on the mask.
            mask = np.zeros(watershed_img.shape, dtype="uint8")
            mask[watershed_img == label] = 255

            # Detect contours in the mask and grab the largest one.
            contours, _ = cv2.findContours(mask.copy(),
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            # Grow the list used to draw circles around WS contours.
            self.ws_max_cntrs.append(max(contours, key=cv2.contourArea))

        # Convert from float32 to uint8 data type to make a PIL Imagetk
        #  Photoimage or find contours.
        distances_img = np.uint8(distances_img)

        # Draw all watershed objects in 1 gray shade instead of each object
        #  decremented by 1 gray value in series; ws boundaries will be black.
        watershed_gray = np.uint8(watershed_img)

        ws_contours, _ = cv2.findContours(watershed_gray,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(watershed_gray,
                         contours=ws_contours,
                         contourIdx=-1,  # all contours.
                         color=120, # (120, 120, 120),  # mid-gray
                         thickness=-1,  # filled
                         lineType=cv2.LINE_AA,
                         )

        # Put thresh image panel next to filtered panel in the same window.
        self.tkimg['thresh'] = manage.tk_image(image=thresh_img,
                                               colorspace='bgr')
        self.img_label['thresh'].configure(image=self.tkimg['thresh'])

        # Dist trans and watershed images are in same window
        self.tkimg['dist_trans'] = manage.tk_image(image=distances_img,
                                                   colorspace='bgr')
        self.img_label['dist_trans'].configure(image=self.tkimg['dist_trans'])

        # Note: use watershed_gray or watershed_img with skimage watershed,
        #  but use watershed_img with cv2.watershed.
        watershed_img = np.uint8(watershed_img)
        self.tkimg['watershed'] = manage.tk_image(
            image=watershed_gray,  # better with skimage watershed.
            colorspace='bgr')
        self.img_label['watershed'].configure(image=self.tkimg['watershed'])

        # Now draw enclosing circles around watershed segments to get sizes.
        self.select_and_size(contour_pointset=self.ws_max_cntrs)

    def select_and_size(self, contour_pointset: list) -> None:
        """
        Select object contours based on area size and position,
        draw an enclosing circle around contours, then display them
        on the input image. Objects are expected to be oblong so circle
        diameter can represent the object's length.
        Called by watershed_segmentation(), process_all(), process_sizes().
        Calls manage.tk_image().

        Args:
            contour_pointset: List of selected contours from cv2.findContours.

        Returns: None
        """
        # Note that circled_ws_segments is an instance attribute (self) because
        #  it is used to save the result with utils.save_settings_and_img().
        self.circled_ws_segments = INPUT_IMG.copy()

        contour_size_list = []
        preferred_color = arguments['color']
        font_scale = input_metrics['font_scale']
        line_thickness = input_metrics['line_thickness']
        unit_per_px = self.unit_per_px.get()

        # The size range slider values are radii pixels. This is done b/c:
        #  1) Displayed values have fewer digits, so a cleaner slide bar.
        #  2) Sizes are diameters, so radii are conceptually easier than areas.
        #  So, need to convert to area for the cv2.contourArea function.
        c_area_min = self.slider_val['c_min_r'].get() ** 2 * np.pi
        c_area_max = self.slider_val['c_max_r'].get() ** 2 * np.pi

        # Set coordinate point limits to find contours along a file border.
        bottom_edge = GRAY_IMG.shape[0] - 1
        right_edge = GRAY_IMG.shape[1] - 1

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
                #  and save each unit_size measurement to a list for reporting.
                ((_x, _y), _r) = cv2.minEnclosingCircle(_c)
                unit_size: float = _r * 2 * unit_per_px
                contour_size_list.append(unit_size)
                offset_x, offset_y = manage.text_offset(txt2size=f'{round(unit_size)}')

                cv2.circle(img=self.circled_ws_segments,
                           center=(int(_x), int(_y)),
                           radius=int(_r),
                           color=preferred_color,
                           thickness=line_thickness,
                           lineType=cv2.LINE_AA,
                           )
                cv2.putText(img=self.circled_ws_segments,
                            text=f'{round(unit_size)}',
                            org=(round(_x - offset_x), round(_y + offset_y)),
                            fontFace=const.FONT_TYPE,
                            fontScale=font_scale,
                            color=preferred_color,
                            thickness=line_thickness,
                            lineType=cv2.LINE_AA,
                            )

            # Grab some metrics for reporting.
            # Conversion factors are set in set_size_std().
            if contour_size_list:
                self.mm_size_list = [round(_d, 1) for _d in contour_size_list]
                self.sorted_d = sorted(self.mm_size_list)

        else:
            print('No objects were found to size. Try changing threshold type.\n'
                  '   Use threshold type *_INVERSE for light-on-dark, not for'
                  ' dark-on-light contrasts.\n'
                  '   Also, "Contour area size" sliders may need adjusting.')

        # Circled sized objects are in their own window.
        self.tkimg['ws_circled'] = manage.tk_image(image=self.circled_ws_segments,
                                                   colorspace='bgr')
        self.img_label['ws_circled'].configure(image=self.tkimg['ws_circled'])

class ImageViewer(ProcessImage):
    """
    A suite of methods to display cv contours based on chosen settings
    and parameters as applied in ProcessImage().
    Methods:
    no_exit_on_x
    setup_image_windows
    setup_settings_window
    setup_explanation
    setup_buttons
    display_input_images
    config_sliders
    config_comboboxes
    config_entries
    grid_widgets
    grid_img_labels
    set_defaults
    set_size_std
    report_results
    process_all
    process_sizes
    """

    __slots__ = (
        'cbox', 'contour_report_frame', 'contour_selectors_frame',
        'img_label', 'img_window', 'mean_px_size',
        'size_settings_txt', 'size_std', 'size_cust_entry',
        'size_cust_label', 'size_std_px_entry',
        'size_std_px_label', 'size_cust_entry',
        'size_cust_label', 'size_std_unit', 'slider',
    )

    def __init__(self):
        super().__init__()
        self.contour_report_frame = tk.Frame()
        self.contour_selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        # Note: The matching control variable attributes for the
        #   following 14 selector widgets are in ProcessImage __init__.
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

        # User entered pixel diameter of size standard object
        #  px_dia textvariable is in ProcessImage __init__
        self.size_std_px_entry = tk.Entry(self.contour_selectors_frame)
        self.size_std_px_label = tk.Label(self.contour_selectors_frame,
                                          text='Enter px diameter of size standard:',
                                          **const.LABEL_PARAMETERS)

        self.size_cust_entry = tk.Entry(self.contour_selectors_frame)
        self.size_cust_label = tk.Label(self.contour_selectors_frame,
                                       text='Enter custom diameter or length:',
                                       **const.LABEL_PARAMETERS)

        # NOTE: dict item order affects the order that windows are
        #  drawn, so here use an inverse order of processing steps to
        #  arrange windows overlaid from first to last, e.g.,
        #  input on bottom, sized or shaped layered on top.
        # NOTE: keys here must match corresponding keys in const.WIN_NAME.
        self.img_window = {
            'ws_contours': tk.Toplevel(),
            'dist_trans': tk.Toplevel(),
            'filtered': tk.Toplevel(),
            'contrasted': tk.Toplevel(),
            'input': tk.Toplevel(),
        }

        # Is an instance attribute here only because it is used in call
        #  to utils.save_settings_and_img() from the Save button.
        self.size_settings_txt = ''

        # Put everything in place, establish initial settings and displays.
        self.setup_image_windows()
        self.setup_settings_window()
        self.setup_explanation()
        self.setup_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_entries()
        self.set_defaults()
        self.grid_widgets()
        self.set_size_std()  # Call here to remove size_cust_label at startup.
        self.grid_img_labels()
        self.display_input_images()

    @staticmethod
    def no_exit_on_x():
        """
        Provide a notice in Terminal.
        Called from .protocol() in setup_image_windows().
        """
        print('This window cannot be closed from the window bar.\n'
              'It can be minimized to get it out of the way.\n'
              'You can quit the program from the OpenCV Settings Report'
              '  window bar or Esc or Ctrl-Q keys.'
              )

    def setup_image_windows(self) -> None:
        """
        Create and configure all Toplevel windows and their Labels that
        are used to display and update processed images.

        Returns: None
        """

        # Prevent user from inadvertently resizing a window too small to use.
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        # Allow image label panels in image windows to resize with window.
        #  Note that images don't proportionally resize, just their boundaries;
        #    images will remain anchored at their top left corners.
        # Configure windows the same as the settings window, to give a yellow
        #  border when it has focus and light grey when being dragged.
        for _name, toplevel in self.img_window.items():
            toplevel.minsize(200, 100)
            toplevel.protocol('WM_DELETE_WINDOW', self.no_exit_on_x)
            toplevel.columnconfigure(0, weight=1)
            toplevel.columnconfigure(1, weight=1)
            toplevel.rowconfigure(0, weight=1)
            toplevel.title(const.WIN_NAME[_name])
            toplevel.config(
                bg=const.MASTER_BG,
                highlightthickness=5,
                highlightcolor=const.CBLIND_COLOR_TK['yellow'],
                highlightbackground=const.DRAG_GRAY,
            )

        # The Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods.
        #  Labels are gridded in their respective img_window in
        #  ImageViewer.grid_img_labels().
        self.img_label = {
            'input': tk.Label(self.img_window['input']),
            'gray': tk.Label(self.img_window['input']),

            'contrast': tk.Label(self.img_window['contrasted']),
            'redux': tk.Label(self.img_window['contrasted']),

            'filter': tk.Label(self.img_window['filtered']),
            'thresh': tk.Label(self.img_window['filtered']),

            'dist_trans': tk.Label(self.img_window['dist_trans']),
            'watershed': tk.Label(self.img_window['dist_trans']),

            'ws_circled': tk.Label(self.img_window['ws_contours']),
        }

    def setup_settings_window(self) -> None:
        """
        Settings and report window (mainloop, "app") keybindings,
        configurations, and grids for contour settings and reporting frames.
        """

        #  Need to set this window toward the top right corner of the screen
        #  so that it doesn't cover up the img windows; also so that
        #  the bottom of the window is, hopefully, not below the bottom
        #  of the screen. Make geometry offset a function of the screen width.
        #  This is needed b/c of differences among platforms' window managers
        #  for how they place windows.
        w_offset = int(self.winfo_screenwidth() * 0.55)
        self.geometry(f'+{w_offset}+0')

        # Color in all the master (app) Frame and use a yellow border;
        #   border highlightcolor changes to grey with loss of focus.
        self.config(
            bg=const.MASTER_BG,
            # bg=const.CBLIND_COLOR_TK['sky blue'],  # for development
            highlightthickness=5,
            highlightcolor=const.CBLIND_COLOR_TK['yellow'],
            highlightbackground=const.DRAG_GRAY,
        )
        # Need to provide exit info msg to Terminal.
        self.protocol('WM_DELETE_WINDOW', lambda: utils.quit_gui(app))

        self.bind_all('<Escape>', lambda _: utils.quit_gui(app))
        self.bind_all('<Control-q>', lambda _: utils.quit_gui(app))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        self.contour_report_frame.configure(relief='flat',
                                            bg=const.CBLIND_COLOR_TK['sky blue'],
                                            )  # bg doesn't show with grid sticky EW.

        self.contour_selectors_frame.configure(relief='raised',
                                               bg=const.DARK_BG,
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

    @staticmethod
    def setup_explanation() -> None:
        """
        Informative note at bottom of settings (mainloop) window about
        the displayed size units.
        Called from __init__.
        """

        explain_label = tk.Label(text='When the entered pixel size is 1 and the selected\n'
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
        manage.ttk_styles(self)

        def save_settings():
            """
            A Button kw "command" caller to avoid messy lambda statements.
            """
            sizes = ', '.join(str(i) for i in self.sorted_d)
            # ",".join(str(bit) for bit in self.sorted_d)
            utils.save_settings_and_img(img2save=self.circled_ws_segments,
                                        txt2save=self.size_settings_txt + sizes,
                                        caller='sizes')

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
                              command=save_settings,
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

    def display_input_images(self) -> None:
        """
        Converts input image to tk image formate and displays it as a
        panel gridded in its toplevel window.
        Called from __init__.
        Calls manage.tkimage(), which applies scaling, cv -> tk array
        conversion, and updates the panel Label's image parameter.
        """

        # Display the input image and its grayscale; both are static, so
        #  they do not need updating, but for consistency's sake they
        #  retain the image display statement structure used for processed
        #  images, which do need updating.
        # Note: here and throughout, use 'self' to scope the
        #  ImageTk.PhotoImage image in the Class, otherwise it will/may
        #  not display b/c of garbage collection.
        self.tkimg['input'] = manage.tk_image(INPUT_IMG, colorspace='bgr')
        self.img_label['input'].configure(image=self.tkimg['input'])
        self.img_label['input'].grid(column=0, row=0, padx=5, pady=5)

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
        c_min_r = manage.input_metrics()['max_circle_r'] // 8
        c_max_r = manage.input_metrics()['max_circle_r']
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
        # responsiveness when changing the size range, just the sizing
        # method is called.
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

        Returns: None
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
        # Note that the if condition doesn't seem to be needed to improve
        # performance or affect bindings; it just clarifies the intention.
        for name, widget in self.cbox.items():
            if '_lbl' not in name:
                widget.bind('<<ComboboxSelected>>', func=self.process_all)

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for all Entry
        widgets in the settings (mainloop) window.
        Provide on-the-fly validation that entries are only digits.
        Called from __init__.

        Returns: None
        """
        def enter_only_digits(entry, action_type) -> bool:
            """
            Only digits are accepted and displayed in Entry field.
            Used with register() to configure Entry kw validatecommand. Example:
            myentry.configure(
                validate='key', textvariable=myvalue,
                validatecommand=(myentry.register(enter_only_digits), '%P', '%d')
                )

            :param entry: value entered into an Entry() widget (%P).
            :param action_type: edit action code (%d).
            :return: True or False
            """
            # Need to restrict entries to only digits,
            #   MUST use action type parameter to allow user to delete first number
            #   entered then re-enter following backspace deletion.
            # source: https://stackoverflow.com/questions/4140437/
            # %P = value of the entry if the edit is allowed
            # Desired action type 1 is "insert", %d.
            if action_type == '1' and not entry.isdigit():
                return False

            return True

        self.size_std_px_entry.config(
            textvariable=self.size_std_px_d,
            width=6,
            validate='all',
            validatecommand=(self.size_std_px_entry.register(enter_only_digits), '%P', '%d')
        )

        self.size_cust_entry.config(
            textvariable=self.custom_size,
            width=6,
            validate='all',
            validatecommand=(self.size_cust_entry.register(enter_only_digits), '%P', '%d')
        )

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
        self.slider['alpha_lbl'].grid(column=0, row=0,
                                      **east_grid_params)
        self.slider['alpha'].grid(column=1, row=0,
                                  **slider_grid_params)

        self.slider['beta_lbl'].grid(column=0, row=1,
                                     **east_grid_params)
        self.slider['beta'].grid(column=1, row=1,
                                 **slider_grid_params)

        self.cbox['morphop_lbl'].grid(column=0, row=2,
                                      **east_grid_params)
        self.cbox['morphop'].grid(column=1, row=2,
                                  **west_grid_params)

        # Note: Put morph shape on same row as morph op.
        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['morphshape'].grid(column=1, row=2,
                                     **east_grid_params)

        self.slider['noise_k_lbl'].grid(column=0, row=4,
                                        **east_grid_params)
        self.slider['noise_k'].grid(column=1, row=4,
                                    **slider_grid_params)

        self.slider['noise_iter_lbl'].grid(column=0, row=5,
                                           **east_grid_params)
        self.slider['noise_iter'].grid(column=1, row=5,
                                       **slider_grid_params)

        self.cbox['filter_lbl'].grid(column=0, row=6,
                                     **east_grid_params)
        self.cbox['filter'].grid(column=1, row=6,
                                 **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['th_type'].grid(column=1, row=6,
                                  **east_grid_params)

        self.slider['filter_k_lbl'].grid(column=0, row=9,
                                         **east_grid_params)
        self.slider['filter_k'].grid(column=1, row=9,
                                     **slider_grid_params)

        self.cbox['dt_type_lbl'].grid(column=0, row=10,
                                      **east_grid_params)
        self.cbox['dt_type'].grid(column=1, row=10,
                                  **west_grid_params)

        # May not be optimized placement for non-Linux platforms, but
        #  is easy to understand.
        self.cbox['dt_mask_size_lbl'].grid(column=1, row=10,
                                           padx=(125, 0),
                                           pady=(4, 0),
                                           sticky=tk.W)
        self.cbox['dt_mask_size'].grid(column=1, row=10,
                                       padx=(200, 0),
                                       pady=(4, 0),
                                       sticky=tk.W)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['ws_connect'].grid(column=1, row=10,
                                     **east_grid_params)

        self.slider['plm_mindist_lbl'].grid(column=0, row=12,
                                            **east_grid_params)
        self.slider['plm_mindist'].grid(column=1, row=12,
                                        **slider_grid_params)

        self.slider['plm_footprint_lbl'].grid(column=0, row=13,
                                              **east_grid_params)
        self.slider['plm_footprint'].grid(column=1, row=13,
                                          **slider_grid_params)


        self.slider['c_min_r_lbl'].grid(column=0, row=17,
                                          **east_grid_params)
        self.slider['c_min_r'].grid(column=1, row=17,
                                      **slider_grid_params)

        self.slider['c_max_r_lbl'].grid(column=0, row=18,
                                          **east_grid_params)
        self.slider['c_max_r'].grid(column=1, row=18,
                                      **slider_grid_params)

        self.size_std_px_label.grid(column=0, row=19,
                                    **east_grid_params)
        self.size_std_px_entry.grid(column=1, row=19,
                                    **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['size_std'].grid(column=1, row=19,
                                   **east_grid_params)

        self.size_cust_entry.grid(column=1, row=20,
                                 **east_grid_params)

        # Use update() because update_idletasks() doesn't always work to
        #  get the gridded widgets' correct winfo_width.
        self.update()

        # Now grid widgets with relative padx values based on widths of
        #  their corresponding partner widgets. Works across platforms.
        morphshape_padx = (0, self.cbox['morphshape'].winfo_width() + 10)
        self.cbox['morphshape_lbl'].grid(column=1, row=2,
                                         padx=morphshape_padx,
                                         pady=(4, 0),
                                         sticky=tk.E)

        thtype_padx = (0, self.cbox['th_type'].winfo_width() + 10)
        self.cbox['th_type_lbl'].grid(column=1, row=6,
                                      padx=thtype_padx,
                                      pady=(4, 0),
                                      sticky=tk.E)

        ws_connect_padx = (0, self.cbox['ws_connect'].winfo_width() + 10)
        self.cbox['ws_connect_lbl'].grid(column=1, row=10,
                                         padx=ws_connect_padx,
                                         pady=(4, 0),
                                         sticky=tk.E)

        size_std_padx = (0, self.cbox['size_std'].winfo_width() + 10)
        self.cbox['size_std_lbl'].grid(column=1, row=19,
                                       padx=size_std_padx,
                                       pady=(4, 0),
                                       sticky=tk.E)

        custom_std_padx = (0, self.size_cust_entry.winfo_width() + 10)
        self.size_cust_label.grid(column=1, row=20,
                                  padx=custom_std_padx,
                                pady=(4, 0),
                                sticky=tk.E)

    def grid_img_labels(self) -> None:
        """
        Grid all image Labels inherited from ProcessImage().
        Labels' 'master' argument for the img window is defined in
        ProcessImage.setup_image_windows(). Label 'image' param is
        updated with .configure() in each PI processing method.
        Called from __init__.

        Returns: None
        """

        self.img_label['contrast'].grid(**const.PANEL_LEFT)
        self.img_label['redux'].grid(**const.PANEL_RIGHT)

        self.img_label['filter'].grid(**const.PANEL_LEFT)
        self.img_label['thresh'].grid(**const.PANEL_RIGHT)

        self.img_label['dist_trans'].grid(**const.PANEL_LEFT)
        self.img_label['watershed'].grid(**const.PANEL_RIGHT)

        self.img_label['ws_circled'].grid(**const.PANEL_RIGHT)

    def set_defaults(self) -> None:
        """
        Sets and resets selector widgets.
        Called from __init__ and "Reset" button.
        """
        # Set/Reset Scale widgets.
        self.slider_val['alpha'].set(1.0)
        self.slider_val['beta'].set(0)
        self.slider_val['noise_k'].set(3)
        self.slider_val['noise_iter'].set(3)
        self.slider_val['filter_k'].set(5)
        self.slider_val['plm_mindist'].set(40)
        self.slider_val['plm_footprint'].set(3)
        self.slider_val['c_min_r'].set(8)
        self.slider_val['c_max_r'].set(300)

        # Set/Reset Combobox widgets.
        self.cbox['morphop'].current(0)
        self.cbox['morphshape'].current(0)
        self.cbox['filter'].current(0)
        if arguments['inverse']:
            self.cbox['th_type'].current(1)  # cv2.THRESH_OTSU_INVERSE
        else:
            self.cbox['th_type'].current(0)  # cv2.THRESH_OTSU

        self.cbox['dt_type'].current(1)  # cv2.DIST_L2
        self.cbox['dt_mask_size'].current(1)  # 3
        self.cbox['ws_connect'].current(1)
        self.cbox['size_std'].current(0)

        # Set to 1 to avoid division by 0.
        self.size_std_px_d.set('1')
        self.mean_px_size = 1

        self.custom_size.set('0')

    def set_size_std(self) -> None:
        """
        Assign a unit conversion factor to the observed pixel diameter
        of the chosen size standard.
        Called from process_all(), process_sizes(), __init__.

        Returns: None
        """

        # Need to avoid division by zero if that's what the user entered.
        #   1 is the default value for this reason.
        # Pre-validation of Entry() values occurs in config_entries().
        if self.size_std_px_d.get() in '0, ""':
            self.size_std_px_d.set('1')

        if self.custom_size.get() == "":
            self.custom_size.set('0')

        self.size_std = self.cbox_val['size_std'].get()
        # For clarity, need to not show the custom size Entry widgets when
        #  'Custom' is not selected, but show them when it is.
        if self.size_std != 'Custom':
            self.size_std_unit = const.SIZE_STANDARDS[self.size_std]
            self.custom_size.set('0')
            self.size_cust_entry.grid_remove()
            self.size_cust_label.grid_remove()
        else:  # is Custom
            self.size_cust_entry.grid()
            self.size_cust_label.grid()
            if int(self.custom_size.get()) > 0:
                self.size_std_unit = int(self.custom_size.get())
            else:
                print('Enter an integer >0 for your custom size standard.')

        self.unit_per_px.set(round(self.size_std_unit / int(self.size_std_px_d.get()), 2))

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from process_all(), process_sizes(), __init__.
        """

        # Note: recall that *_val dict are inherited from ProcessImage().
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
        num_selected = len(self.mm_size_list)
        mean_unit_dia = round(mean(self.mm_size_list), 1)
        median_unit_dia = round(median(self.mm_size_list))
        mm_range = f'{min(self.mm_size_list)}--{max(self.mm_size_list)}'

        # Only odd kernel integers are used for processing.
        _nk = self.slider_val['noise_k'].get()
        _fk = self.slider_val['filter_k'].get()

        noise_k = _nk + 1 if _nk % 2 == 0 else _nk

        if _fk != 0:
            filter_k = _fk + 1 if _fk % 2 == 0 else _fk
        else:
            filter_k = _fk

        # Text is formatted for clarity in window, terminal, and saved file.
        space = 23
        tab = " ".ljust(space)
        # Divider symbol is Box Drawings Double Horizontal from https://coolsymbol.com/
        divider = "═" * 20  # divider's unicode_escape: b'\\u2550\'

        self.size_settings_txt = (
            f'Image: {arguments["input"]} {INPUT_IMG.shape[0]}x{INPUT_IMG.shape[1]}\n\n'
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
            f'{tab}compactness=0.03\n'  # NOTE: change if changes in watershed method.
            f'{divider}\n'
            f'{"Total distT segments:".ljust(space)}{self.num_dt_segments} <- Match'
            f'  # selected objects for better sizing.\n'
            f'{"Circled radius range:".ljust(space)}{c_min_r}--{c_max_r} pixels\n'
            f'{"Selected size std.:".ljust(space)}{self.size_std},'
            f' {self.size_std_unit} unit dia.\n'
            f'{tab}Pixel diameter entered: {self.size_std_px_d.get()},'
            f' unit/px factor: {self.unit_per_px.get()}\n'
            f'{"# Selected objects:".ljust(space)}{num_selected}\n'
            f'{"Object size metrics,".ljust(space)}mean: {mean_unit_dia}, median:'
            f' {median_unit_dia}, range: {mm_range}\n'
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
        self.report_results()

        return event

    def process_sizes(self, event=None) -> None:
        """
        Improve performance by running only select_and_size
        from ProcessImage and report_results() from ImageViewer.
        Called from the c_min_r and c_max_r sliders.
        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.
        """
        self.set_size_std()
        self.select_and_size(self.ws_max_cntrs)
        self.report_results()

        return event

def patience_needed():
    """An informational message to users in a hurry."""
    if GRAY_IMG.shape[1] > 2000:
        print('Images over 2000 pixels wide will take longer to process...'
              ' patience Grasshopper.\n  If the threshold image shows up as'
              ' black-on-white, then use the --inverse command line option.')

if __name__ == "__main__":
    # Program exits here if any of the module checks fail.
    utils.check_platform()
    vcheck.minversion('3.7')
    arguments = manage.arguments()

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values and define often used values...
    input_metrics = manage.input_metrics()
    INPUT_IMG = input_metrics['input_img']
    GRAY_IMG = input_metrics['gray_img']

    patience_needed()

    try:
        print(f'{Path(__file__).name} has launched...')
        app = ImageViewer()
        app.title('Count & Size Settings Report')
        app.resizable(width=True, height=False)
        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal command line. ***\n')
