#!/usr/bin/env python3
"""
A tkinter GUI, size_it_cs.py, for OpenCV processing of an image to obtain
sizes, means, and ranges of objects in a sample population. Object
segmentation is by use of a matte color screen ('cs'), such as with a
green screen. Different matte colors can be selected. Noise reduction
is interactive with live updating of resulting images.

A report is provided of parameter settings, object count, individual
object sizes, and sample size mean and range, along with an annotated
image file of labeled objects.

USAGE
For command line execution, from within the count-and-size-main folder:
python3 -m size_it_cs --about
python3 -m size_it_cs --help
python3 -m size_it_cs
python3 -m size_it_cs --terminal
Windows systems may need to substitute 'python3' with 'py' or 'python'.

Commands for saving results and settings, adjusting images' screen
size, and annotation styles are available from the "Report & Settings"
window's menubar and buttons.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
report window or File menubar. From the command line, use Ctrl-C.

Requires Python 3.7 or later and the packages opencv-python, numpy,
scikit-image, scipy, and psutil.
See this distribution's requirements.txt file for details.
Developed in Python 3.8 and 3.9, tested up to 3.11.
"""
# Copyright (C) 2024 C.S. Echt, under GNU General Public License

# Standard library imports.
from datetime import datetime
from json import loads
from pathlib import Path
from signal import signal, SIGINT
from statistics import mean, median
from sys import exit as sys_exit
from time import time
from typing import Union, List

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
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy import ndimage

except (ImportError, ModuleNotFoundError) as import_err:
    sys_exit(
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
    reduce_noise
    matte_segmentation
    watershed_segmentation
    """

    def __init__(self):
        super().__init__()

        # Note: The matching selector widgets for the following
        #  control variables are in ContourViewer __init__.
        self.slider_val = {
            # For Scale() widgets in config_sliders()...
            'noise_k': tk.IntVar(),
            'noise_iter': tk.IntVar(),
            'circle_r_min': tk.IntVar(),
            'circle_r_max': tk.IntVar(),
            # For Scale() widgets in setup_watershed_window()...
            'plm_mindist': tk.IntVar(),
            'plm_footprint': tk.IntVar(),
        }

        self.scale_factor = tk.DoubleVar()

        self.cbox_val = {
            # For textvariables in config_comboboxes()...
            'morph_op': tk.StringVar(),
            'morph_shape': tk.StringVar(),
            'size_std': tk.StringVar(),
            # For setup_start_window()...
            'annotation_color': tk.StringVar(),
            'matte_color': tk.StringVar(),
        }

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  with Label images displayed in their respective tkimg_window Toplevel.
        #  The cvimg images are numpy arrays.
        self.tkimg: dict = {}
        self.cvimg: dict = {}
        for _name in const.CS_IMAGE_NAMES:
            self.tkimg[_name] = tk.PhotoImage()
            self.cvimg[_name] = const.STUB_ARRAY

        # img_label dictionary is set up in SetupApp.setup_image_windows(),
        #  but is used in all Class methods here.
        self.img_label: dict = {}

        # metrics dict is populated in SetupApp.open_input().
        self.metrics: dict = {}
        self.line_thickness: int = 0
        self.font_scale: float = 0
        self.matte_contours: tuple = ()
        self.watershed_basins: tuple = ()
        self.sorted_size_list: list = []
        self.unit_per_px = tk.DoubleVar()
        self.num_sigfig: int = 0
        self.time_start: float = 0
        self.elapsed: Union[float, int, str] = 0

    def update_image(self, image_name: str) -> None:
        """
        Process a cv2 image array to use as a tk PhotoImage and update
        (configure) its window label for immediate display, at scale.
        Calls module manage.tk_image(). Called from all methods that
        display an image.

        Args:
            image_name: An item name used in the image_name tuple, for
                use as key in tkimg, cvimg, and img_label dictionaries.

        Returns:
            None
        """

        self.tkimg[image_name] = manage.tk_image(
            image=self.cvimg[image_name],
            scale_factor=self.scale_factor.get()
        )
        self.img_label[image_name].configure(image=self.tkimg[image_name])

    def reduce_noise(self, img: np.ndarray) -> None:
        """
        Reduce noise in the matte mask image using erode and dilate
        actions of cv2.morphologyEx operations.
        Called by matte_segmentation(). Calls update_image().

        Args:
            img: The color matte mask from matte_segmentation().
        Returns: None
        """

        iteration = self.slider_val['noise_iter'].get()
        if iteration == 0:
            self.cvimg['redux_mask'] = img
            self.update_image(image_name='redux_mask')
            return

        noise_k = self.slider_val['noise_k'].get()

        # Need integers for the cv function parameters.
        morph_shape = const.CV['morph_shape'][self.cbox_val['morph_shape'].get()]
        morph_op = const.CV['morph_op'][self.cbox_val['morph_op'].get()]

        # See: https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/filtering.html
        #  on page, see: cv2.getStructuringElement(shape, ksize[, anchor])
        # see: https://docs.opencv2.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        element = cv2.getStructuringElement(
            shape=morph_shape,
            ksize=(noise_k, noise_k))

        # Use morphologyEx as a shortcut for erosion followed by dilation.
        # Read https://docs.opencv2.org/3.4/db/df6/tutorial_erosion_dilatation.html
        # https://theailearner.com/tag/cv-morphologyex/
        # The op argument from const.CV['morph_op'] options:
        #   MORPH_OPEN is useful to remove noise and small features.
        #   MORPH_CLOSE is better for certain images, but generally is worse.
        #   MORPH_HITMISS helps to separate close objects by shrinking them.
        # morph_op HITMISS works best for colorseg, with a low kernel size.
        # cvimg['redux_mask'] is used in matte_segmentation() and watershed_segmentation().
        self.cvimg['redux_mask'] = cv2.morphologyEx(
            src=img,
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=cv2.BORDER_DEFAULT,
        )

        self.update_image(image_name='redux_mask')

    def matte_segmentation(self) -> None:
        """
        An optional segmentation method to use on color matte masks,
        e.g., green screen.

        Returns: None
        """

        # Convert the input image to HSV colorspace for better color segmentation.
        hsv_img = cv2.cvtColor(src=self.cvimg['input'], code=cv2.COLOR_BGR2HSV)

        # see: https://stackoverflow.com/questions/47483951/
        #  how-can-i-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-ima
        #  /47483966#47483966

        # Dict values are the lower and upper (light & dark)
        #   BGR colorspace range boundaries to use for HSV color discrimination.
        # Note that cv2.inRange thresholds all elements within the
        # color bounds to white and everything else to black.
        lower, upper = const.MATTE_COLOR_RANGE[self.cbox_val['matte_color'].get()]
        matte_mask = cv2.inRange(src=hsv_img, lowerb=lower, upperb=upper)

        # Run the mask through noise reduction, then use inverse of image for
        #  finding matte_objects contours.
        self.reduce_noise(matte_mask)
        self.cvimg['matte_objects'] = cv2.bitwise_not(self.cvimg['redux_mask'])

        # matte_contours is used in select_and_size_objects() and select_and_export_objects(),
        #  where selected contours are used to draw and size enclosing circles or
        #  draw and export the ROI.
        # The data type is a tuple of lists of contour pointsets.
        self.matte_contours, _ = cv2.findContours(image=np.uint8(self.cvimg['matte_objects']),
                                                  mode=cv2.RETR_EXTERNAL,
                                                  method=cv2.CHAIN_APPROX_NONE)

        self.cvimg['matte_objects'] = cv2.cvtColor(src=self.cvimg['matte_objects'],
                                                   code=cv2.COLOR_GRAY2BGR)

        # Need to avoid contour colors that cannot be seen well against
        #  a black or white background.
        if self.cbox_val['annotation_color'].get() in 'white, black, dark blue':
            line_color = const.COLORS_CV['orange']
        else:
            line_color = const.COLORS_CV[self.cbox_val['annotation_color'].get()]

        cv2.drawContours(image=self.cvimg['matte_objects'],
                         contours=self.matte_contours,
                         contourIdx=-1,  # do all contours
                         color=line_color,
                         thickness=self.line_thickness,
                         lineType=cv2.LINE_AA)

        self.update_image(image_name='matte_objects')

        # Now need to draw enclosing circles around selected segments and
        #  annotate with object sizes in ViewImage.select_and_size_objects().

    def watershed_segmentation(self) -> None:
        """
        Separate groups of objects in a matte mask. Inverts the noise-reduced
        mask, then applies a distance transform and watershed algorithm
        to separate objects that are touching. Intended for use when the
        color matte is not sufficient for complete object separation.

        Returns: None
        """

        # Calculate the distance transform of the objects' masks by
        #  replacing each foreground (non-zero) element with its
        #  shortest distance to the background (any zero-valued element).
        #  Returns a float64 ndarray.
        # Note that maskSize=0 calculates the precise mask size only for cv2.DIST_L2.
        #  cv2.DIST_L1 and cv2.DIST_C always use maskSize=3.
        inv_img = cv2.bitwise_not(self.cvimg['redux_mask'])
        transformed: np.ndarray = cv2.distanceTransform(
            src=inv_img,
            distanceType=cv2.DIST_L2,
            maskSize=0
        )

        p_kernel: tuple = (self.slider_val['plm_footprint'].get(),
                           self.slider_val['plm_footprint'].get())

        local_max: ndimage = peak_local_max(
            image=transformed,
            min_distance=self.slider_val['plm_mindist'].get(),
            exclude_border=False,  # True uses min_distance.
            num_peaks=np.inf,
            footprint=np.ones(shape=p_kernel, dtype=np.uint8),
            labels=inv_img,
            num_peaks_per_label=np.inf,
            p_norm=np.inf,  # Chebyshev distance
        )

        mask = np.zeros(shape=transformed.shape, dtype=bool)
        # Set background to True (not zero: True or 1)
        mask[tuple(local_max.T)] = True

        # Note that markers are single px, colored in grayscale by their label index.
        # Source: http://scipy-lectures.org/packages/scikit-image/index.html
        # From the doc: labels: array of ints, of same shape as data without channels dimension.
        #  Array of seed markers labeled with different positive integers for
        #  different phases. Zero-labeled pixels are unlabeled pixels.
        #  Negative labels correspond to inactive pixels that are not taken into
        #  account (they are removed from the graph).
        labeled_array, _ = ndimage.label(input=mask)
        labeled_array[labeled_array == inv_img] = -1

        # Note that the minus symbol with the image argument converts the
        #  distance transform into a threshold. Watershed can work without
        #  that conversion, but does a better job identifying segments with it.
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_compact_watershed.html
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
        # compactness=1.0 based on: DOI:10.1109/ICPR.2014.181
        # Watershed_line=True is necessary to separate touching objects.
        watershed_img: np.ndarray = watershed(
            image=-transformed,
            markers=labeled_array,
            connectivity=4,
            mask=inv_img,
            compactness=1.0,
            watershed_line=True
        )

        # watershed_basins are the contours to be passed to select_and_size_objects(),
        #  where selected contours are used to draw and size enclosing circles.
        # The data type is a tuple of lists of contour pointsets.
        self.watershed_basins, _ = cv2.findContours(
            image=np.uint8(watershed_img),
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_NONE,
        )


class ViewImage(ProcessImage):
    """
    A suite of methods to display cv segments based on selected settings
    and parameters that are in ProcessImage() methods.
    Methods:
    set_auto_scale_factor
    import_settings
    delay_size_std_info_msg
    show_info_message
    configure_circle_r_sliders
    widget_control
    noise_widget_control
    validate_px_size_entry
    validate_custom_size_entry
    set_size_standard
    is_selected_contour
    measure_object
    annotate_object
    select_and_size_objects
    preview_export
    mask_for_export
    define_roi
    select_and_export_objects
    report_results
    process_ws
    process_matte
    process_sizes
    """

    def __init__(self):
        super().__init__()

        self.first_run: bool = True

        self.report_frame = tk.Frame()
        self.selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        # The control variables with matching names for these Scale() and
        #  Combobox() widgets are instance attributes in ProcessImage.
        # plm_mindist and plm_footprint are for watershed segmentation.
        # 'plm' is for peak_local_max, a function from skimage.feature.
        self.slider = {

            'noise_k': tk.Scale(master=self.selectors_frame),
            'noise_k_lbl': tk.Label(master=self.selectors_frame),

            'noise_iter': tk.Scale(master=self.selectors_frame),
            'noise_iter_lbl': tk.Label(master=self.selectors_frame),

            'circle_r_min': tk.Scale(master=self.selectors_frame),
            'circle_r_min_lbl': tk.Label(master=self.selectors_frame),

            'circle_r_max': tk.Scale(master=self.selectors_frame),
            'circle_r_max_lbl': tk.Label(master=self.selectors_frame),

            'plm_mindist': tk.Scale(),
            'plm_mindist_lbl': tk.Label(),

            'plm_footprint': tk.Scale(),
            'plm_footprint_lbl': tk.Label(),
        }

        self.cbox = {
            'morph_op': ttk.Combobox(master=self.selectors_frame),
            'morph_op_lbl': tk.Label(master=self.selectors_frame),

            'morph_shape': ttk.Combobox(master=self.selectors_frame),
            'morph_shape_lbl': tk.Label(master=self.selectors_frame),

            'size_std': ttk.Combobox(master=self.selectors_frame),
            'size_std_lbl': tk.Label(master=self.selectors_frame),

            'matte_color': ttk.Combobox(master=self.selectors_frame),
            'matte_lbl': tk.Label(master=self.selectors_frame),
        }

        self.size_std = {
            'px_entry': tk.Entry(master=self.selectors_frame),
            'px_val': tk.StringVar(master=self.selectors_frame),
            'px_lbl': tk.Label(master=self.selectors_frame),
            'custom_entry': tk.Entry(master=self.selectors_frame),
            'custom_val': tk.StringVar(master=self.selectors_frame),
            'custom_lbl': tk.Label(master=self.selectors_frame),
        }

        self.button = {
            'process_matte': ttk.Button(master=self),
            'save_results': ttk.Button(master=self),
            'new_input': ttk.Button(master=self),
            'export_objects': ttk.Button(master=self),
            'export_settings': ttk.Button(master=self),
            'reset': ttk.Button(master=self),
        }

        # Screen pixel width is defined in setup_main_window().
        self.screen_width: int = 0

        # Info label is gridded in configure_main_window().
        self.info_txt = tk.StringVar()
        self.info_label = tk.Label(master=self, textvariable=self.info_txt)

        # Defined in widget_control() to reset values that user may have
        #  tried to change during prolonged processing times.
        self.slider_values: list = []

        # The following group of attributes is set in SetupApp.open_input().
        self.input_file_path: str = ''
        self.input_file_name: str = ''
        self.input_folder_name: str = ''
        self.input_folder_path: str = ''
        self.input_ht: int = 0
        self.input_w: int = 0
        self.settings_file_path = Path('')

        self.use_saved_settings: bool = False

        self.imported_settings: dict = {}

        # The following group of attributes is set in select_and_size_objects().
        self.num_objects_selected: int = 0
        self.selected_sizes: List[float] = []
        self.object_labels: List[list] = []

        self.report_txt: str = ''

        # This window, used to display the watershed segmentation controllers,
        #  is defined in SetupApp.setup_watershed_window().
        self.watershed_window = None

    def set_auto_scale_factor(self) -> None:
        """
        As a convenience for user, set a default scale factor to that
        needed for images to fit easily on the screen, either 1/3
        screen px width or 2/3 screen px height, depending
        on input image orientation.
        Called from open_input() and _Command.apply_default_settings().

        Returns: None
        """

        # Note that the scale factor is not included in saved_settings.json.
        if self.input_w >= self.input_ht:
            estimated_scale = round((self.screen_width * 0.33) / self.input_w, 2)
        else:
            estimated_scale = round((self.winfo_screenheight() * 0.66) / self.input_ht, 2)

        self.scale_factor.set(estimated_scale)

    def import_settings(self) -> None:
        """
        Uses a dictionary of saved settings, imported via json.loads(),
        that are to be applied to the input image. Includes all settings
        except the scale_factor for window image size.
        Called from check_for_saved_settings(), set_cololr_defaults().

        Returns: None
        """

        try:
            with open(self.settings_file_path, mode='rt', encoding='utf-8') as _fp:
                settings_json = _fp.read()
                self.imported_settings: dict = loads(settings_json)
        except FileNotFoundError as fnf:
            print('The settings JSON file could not be found.\n'
                  f'{fnf}')
        except OSError as oserr:
            print('There was a problem reading the settings JSON file.\n'
                  f'{oserr}')

        # Set/Reset Scale widgets.
        for _name in self.slider_val:
            self.slider_val[_name].set(self.imported_settings[_name])

        # Set/Reset Combobox widgets.
        for _name in self.cbox_val:
            self.cbox_val[_name].set(self.imported_settings[_name])

        self.font_scale = self.imported_settings['font_scale']
        self.line_thickness = self.imported_settings['line_thickness']

        self.size_std['px_val'].set(self.imported_settings['px_val'])
        self.size_std['custom_val'].set(self.imported_settings['custom_val'])

    def delay_size_std_info_msg(self) -> None:
        """
        When no size standard values are entered  display, after a few
        seconds, size standard instructions in the mainloop (app)
        window.
        Called from process_ws(), process_matte(), process_sizes(), and
        configure_buttons._new_input().
        Internal function calls show_info_message().

        Returns: None
        """

        def _show_msg() -> None:
            _info = ('\nWhen entered pixel size is 1 AND size standard is "None",\n'
                     'then size units are pixels.\n'
                     'Size units are millimeters for any pre-set size standard.\n'
                     f'(Processing time elapsed: {self.elapsed})\n')

            self.show_info_message(info=_info, color='black')

        if (self.size_std['px_val'].get() == '1' and
                self.cbox_val['size_std'].get() == 'None'):
            self.after(ms=7777, func=_show_msg)

    def show_info_message(self, info: str, color: str) -> None:
        """
        Configure for display and update the informational message in
        the report and settings window.
        Args:
            info: The text string of the message to display.
            color: The font color string, either as a key in the
                   const.COLORS_TK dictionary or as a Tk compatible fg
                   color string, i.e. hex code or X11 named color.

        Returns: None
        """
        self.info_txt.set(info)

        # Need to handle cases when color is defined as a dictionary key,
        #  hex code, or X11 named color.
        try:
            tk_color = const.COLORS_TK[color]
        except KeyError:
            tk_color = color

        self.info_label.config(fg=tk_color)

    def configure_circle_r_sliders(self) -> None:
        """
        Adjust the Scale() widgets for circle radius min and max based
        on the input image size.
        Called from config_sliders() and open_input().
        Returns: None
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

    def widget_control(self, action: str) -> None:
        """
        Used to disable settings widgets when segmentation is running.
        Provides a watch cursor while widgets are disabled.
        Gets Scale() values at time of disabling and resets them upon
        enabling, thus preventing user click events retained in memory
        during processing from changing slider position post-processing.

        Args:
            action: Either 'off' to disable widgets, or 'on' to enable.
        Returns:
            None
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
            self.watershed_window.config(cursor='watch')
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

            # Need to keep the noise reduction widgets disabled when
            #  iterations are zero.
            self.noise_widget_control()

            self.config(cursor='')
            self.watershed_window.config(cursor='')
            self.slider_values.clear()

        # Use update(), not update_idletasks, here to improve promptness
        #  of windows' response.
        self.update()

    def noise_widget_control(self) -> None:
        """
        Disables noise reduction settings widgets when iterations are
        zero and enable when not. Used to refine the broad actions of
        widget_control(action='on').
        Called by widget_control(), process_matte(), _Command.process().
        Calls widget_control().

        Returns: None
        """

        if self.slider_val['noise_iter'].get() == 0:
            for _name, _widget in self.cbox.items():
                if 'morph' in _name:
                    _widget.configure(state=tk.DISABLED)
            for _name, _widget in self.slider.items():
                if 'noise_k' in _name:
                    _widget.configure(state=tk.DISABLED)
        else:  # is > 0, so widget now relevant.
            # Need to re-enable the noise reduction widgets, but is simplest
            #  to re-enable all widgets.
            for _, _widget in self.cbox.items():
                if isinstance(_widget, tk.Label):
                    _widget.configure(state=tk.NORMAL)
                else:  # is tk.Combobox
                    _widget.configure(state='readonly')
            for _, _widget in self.slider.items():
                _widget.configure(state=tk.NORMAL)

    def validate_px_size_entry(self) -> None:
        """
        Check whether pixel size Entry() value is a positive integer.
        Post a message if the entry is not valid.
        Called by set_size_standard() process_ws(), process_matte().
        Calls widget_control().

        Returns: None
        """

        size_std_px: str = self.size_std['px_val'].get()
        try:
            size_std_px_int = int(size_std_px)
            if size_std_px_int <= 0:
                raise ValueError
        except ValueError:

            # Need widget_control to prevent runaway sliders, if clicked.
            self.widget_control(action='off')
            _post = ('Enter a whole number > 0 for the pixel diameter.\n'
                     f'{size_std_px} was entered. Defaulting to 1.')
            messagebox.showerror(title='Invalid entry',
                                 detail=_post)
            self.size_std['px_val'].set('1')
            self.widget_control(action='on')

    def validate_custom_size_entry(self) -> None:
        """
        Check whether custom size Entry() value is a real number.
        Post a message if the entry is not valid.
        Called by set_size_standard() process_ws(), process_matte().
        Calls widget_control().

        Returns: None
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
            # float() will raise ValueError if custom_size is not a number.
            custom_size_float = float(custom_size)
            if custom_size_float <= 0:
                raise ValueError

            self.unit_per_px.set(custom_size_float / int(size_std_px))

            if size_std_px == '1':
                self.num_sigfig = utils.count_sig_fig(custom_size)
            else:
                self.num_sigfig = min(utils.count_sig_fig(custom_size),
                                      utils.count_sig_fig(size_std_px))
        except ValueError:

            # Need widget_control() to prevent runaway sliders, if clicked.
            self.widget_control(action='off')
            messagebox.showinfo(title='Custom size',
                                detail='Enter a number > 0.\n'
                                       'Accepted types:\n'
                                       '  integer: 26, 2651, 2_651\n'
                                       '  decimal: 26.5, 0.265, .2\n'
                                       '  exponent: 2.6e10, 2.6e-2')
            self.size_std['custom_val'].set('0.0')
            self.widget_control(action='on')

    def set_size_standard(self) -> None:
        """
        Assign a unit conversion factor to the observed pixel diameter
        of the chosen size standard and calculate the number of
        significant figures for preset or custom size entries.
        Called from start_now(), process_sizes().
        Calls validate_px_size_entry(), validate_custom_size_entry(),
        and utils.count_sig_fig().

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

        else:  # is one of the preset size standards or 'None'.
            self.size_std['custom_entry'].grid_remove()
            self.size_std['custom_lbl'].grid_remove()
            self.size_std['custom_val'].set('0.0')

            self.unit_per_px.set(preset_std_size / int(size_std_px))

            if size_std_px == '1':
                self.num_sigfig = utils.count_sig_fig(preset_std_size)
            else:
                self.num_sigfig = min(utils.count_sig_fig(preset_std_size),
                                      utils.count_sig_fig(size_std_px))

    def is_selected_contour(self, contour: np.ndarray) -> bool:
        """
        Filter each contour segment based on size and position.
        Exclude None elements, contours not in the specified size range,
        and contours that have a coordinate point intersecting an image border.

        Args:
            contour: A numpy array of contour points.

        Returns:
            True if the contour is within the specified parameters,
            False if not.
        """
        # Filter each contour segment based on size and position.
        # Exclude None elements.
        # Exclude contours not in the specified size range.
        # Exclude contours that have a coordinate point intersecting an img edge.
        #  ... those that touch top or left edge or are background.
        #  ... those that touch bottom or right edge.

        # The size range slider values are radii pixels. This is done b/c:
        #  1) Displayed values have fewer digits, so a cleaner slide bar.
        #  2) Sizes are diameters, so radii are conceptually easier than areas.
        #  So, need to convert to area for the cv2.contourArea function.
        c_area_min: float = self.slider_val['circle_r_min'].get() ** 2 * np.pi
        c_area_max: float = self.slider_val['circle_r_max'].get() ** 2 * np.pi

        # Set limits for coordinate points to identify contours that
        # are within 1 px of an image file border (edge).
        bottom_edge: int = self.input_ht - 1
        right_edge: int = self.input_w - 1

        if contour is None:
            return False
        if not c_area_max > cv2.contourArea(contour) >= c_area_min:
            return False
        if {0, 1}.intersection(set(contour.ravel())):
            return False
        for xy_point in contour:
            if xy_point[0][0] == right_edge or xy_point[0][1] == bottom_edge:
                return False
        return True

    def measure_object(self, contour: np.ndarray) -> tuple:
        """
        Measure object's size as an enclosing circle diameter of its contour.
        Calls to_p.to_precision() to set the number of significant figures.

        Args: contour: A numpy array of contour points.
        Returns: tuple of the circle's center coordinates, radius, and
            size diameter to display.
        """

        # Measure object's size as an enclosing circle diameter of its contour.
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # Note: sizes are full-length floats.
        object_size: float = radius * 2 * self.unit_per_px.get()

        # Need to set sig. fig. to display sizes in annotated image.
        #  num_sigfig value is determined in set_size_standard().
        display_size: str = to_p.to_precision(value=object_size,
                                              precision=self.num_sigfig)

        # Need to have pixel diameters as integers. Because...
        #  When num_sigfig is 4, as is case for None:'1.001' in
        #  const.SIZE_STANDARDS, then for px_val==1, with lower
        #  sig.fig., objects <1000 px diameter would display as
        #  decimal fractions. So, round()...
        if (self.size_std['px_val'].get() == '1' and
                self.cbox_val['size_std'].get() == 'None'):
            display_size = str(round(float(display_size)))
        return x, y, radius, display_size

    def annotate_object(self, x_coord: float,
                        y_coord: float,
                        radius: float,
                        size: str) -> None:
        """
        Draw a circle around the object's coordinates and annotate
        its size.

        Args: x_coord, y_coord, radius, size: The object's center x and y,
            radius, and size diameter to display.
        Returns: None
        """

        color: tuple = const.COLORS_CV[self.cbox_val['annotation_color'].get()]

        # Need to properly center text in the circled object.
        ((txt_width, _), baseline) = cv2.getTextSize(
            text=size,
            fontFace=const.FONT_TYPE,
            fontScale=self.font_scale,
            thickness=self.line_thickness)
        offset_x = txt_width / 2

        cv2.circle(img=self.cvimg['sized'],
                   center=(round(x_coord),
                           round(y_coord)),
                   radius=round(radius),
                   color=color,
                   thickness=self.line_thickness,
                   lineType=cv2.LINE_AA,
                   )
        cv2.putText(img=self.cvimg['sized'],
                    text=size,
                    org=(round(x_coord - offset_x),
                         round(y_coord + baseline)),
                    fontFace=const.FONT_TYPE,
                    fontScale=self.font_scale,
                    color=color,
                    thickness=self.line_thickness,
                    lineType=cv2.LINE_AA,
                    )

    def select_and_size_objects(self) -> None:
        """
        Select object contour ROI based on area size and position,
        draw an enclosing circle around contours, then display the
        diameter size over the input image. Objects are expected to be
        oblong so that circle diameter can represent the object's length.
        Called by process*() methods and annotation methods in call_cmd().
        Calls is_selected_object(), measure_object(), annotate_object(),
            to_p.to_precision(), utils.no_objects_found_msg() as needed,
            and update_image().

        Returns: None
        """

        # Sized image copy is used in annotate_object().
        self.cvimg['sized'] = self.cvimg['input'].copy()

        # Need to determine whether the watershed algorithm is in use,
        #  which is the case when the ws control window is visible.
        if self.watershed_window.wm_state() in 'normal, zoomed':
            contour_pointset = self.watershed_basins
        else:  # is 'withdrawn' or 'iconic'
            contour_pointset = self.matte_contours

        # Note that with matte screens, the contour_pointset may contain
        #  a single element of the entire image, if no objects are found.
        if contour_pointset is None or len(contour_pointset) == 1:
            self.update_image(image_name='sized')
            utils.no_objects_found_msg(caller=PROGRAM_NAME)
            return

        # Need to reset selected_sizes list for each call.
        self.selected_sizes.clear()
        self.object_labels.clear()

        # Need to reset the number of objects selected for each call.
        self.num_objects_selected = 0

        for contour in contour_pointset:
            if self.is_selected_contour(contour=contour):
                _x, _y, _r, size_to_display = self.measure_object(contour)
                self.annotate_object(x_coord=_x,
                                     y_coord=_y,
                                     radius=_r,
                                     size=size_to_display)

                # Save each object_size measurement to the selected_sizes
                #  list for reporting.
                # Convert size_to_display string to float, assuming that individual
                #  sizes listed in the report may be used in a spreadsheet
                #  or for other statistical analysis.
                self.selected_sizes.append(float(size_to_display))

                # Specify object data to export for CNN training, as a convenience.
                _x, _y, _w, _h = cv2.boundingRect(contour)
                label_class = f'{manage.arguments()["prefix"]}'
                self.object_labels.append([label_class,
                                           _x, _y, _w, _h,
                                           self.input_file_name,
                                           self.input_w, self.input_ht,
                                           size_to_display]
                                          )

                self.num_objects_selected += 1

        # The sorted size list is used to report individual sizes and
        #  their summary metrics. Display message if no objects found.
        if self.selected_sizes:
            self.sorted_size_list = sorted(self.selected_sizes)
        else:
            utils.no_objects_found_msg(caller=PROGRAM_NAME)

        self.update_image(image_name='sized')

    def preview_export(self, export_img: tk.PhotoImage) -> tk.Toplevel:
        """
        Display a sample of the export image and import image, from
        select_and_export_objects(), in a Toplevel window.
        Args:
            export_img: A PhotoImage of the sample ROI.
        Returns:
            preview_win: A Toplevel window with the export image.
        """
        w_offset = int(self.screen_width * 0.50)

        preview_win = tk.Toplevel(**const.WINDOW_PARAMETERS)
        preview_win.title('Export sample, zoom view')
        preview_win.minsize(width=300, height=150)
        preview_win.geometry(f'+{w_offset}+55')
        preview_win.columnconfigure(index=0, weight=1)
        preview_win.columnconfigure(index=1, weight=1)
        preview_win.bind('<Escape>', func=lambda _: utils.quit_gui(self))
        preview_win.bind('<Control-q>', func=lambda _: utils.quit_gui(self))

        tk.Label(master=preview_win, image=export_img).pack()

        return preview_win

    def mask_for_export(self, contour: np.ndarray) -> None:
        """
        Create a binary mask of the segment on the input image.
        Args:
            contour: The numpy array of a single set of contour points.

        Returns:
            None
        """

        # Idea for masking from: https://stackoverflow.com/questions/70209433/
        #   opencv-creating-a-binary-mask-from-the-image
        # Need to use full input img b/c that is what _c pointset refers to.
        # Steps: Make a binary mask of segment on full input image.
        #        Crop the mask to a slim border around the segment.
        mask = np.zeros_like(cv2.cvtColor(src=self.cvimg['input'],
                                          code=cv2.COLOR_BGR2GRAY))

        cv2.drawContours(image=mask,
                         contours=[contour],
                         contourIdx=-1,
                         color=(255, 255, 255),
                         thickness=cv2.FILLED)

        # Note: this contour step provides a cleaner border around the
        #  segment that includes less of the color matte.
        cv2.drawContours(image=mask,
                         contours=[contour],
                         contourIdx=-1,
                         color=(0, 0, 0),
                         thickness=4)

    def define_roi(self, contour: np.ndarray) -> tuple:
        """
        Define a region of interest (ROI) slice from the input image
        based on the bounding rectangle of the contour segment.
        Args:
            contour: A numpy array of contour points.
        Returns:
            roi: A numpy array of the ROI slice, and its y and x slices.
        """

        # Idea for segment extraction from:
        #  https://stackoverflow.com/questions/21104664/
        #   extract-all-bounding-boxes-using-opencv-python
        #  https://stackoverflow.com/questions/46441893/

        # To preview and export masked segments on a background,
        #  other than the matte color, uncomment the commented statements,
        #  replace img2exp=roi with img2exp=segment as the
        #  export_each_segment() arg used in select_and_export_objects().
        #  In select_and_export_objects(), may want to replace if roi...
        #  with if segment...
        #  To extract a segment from input to a black background, use the
        #   statements below. To extract a segment from input to a white
        #   background use: segment[roi_mask == 0] = 255, 255, 255.
        #   Use these statements below...
        #   roi_mask = mask[y_slice, x_slice]
        #   segment = cv2.bitwise_and(src1=roi, src2=roi, mask=roi_mask)
        #   and export the segment instead of the roi.
        #  See size_it.py for the full code. Modify as needed for here.

        _x, _y, _w, _h = cv2.boundingRect(contour)

        # ROI slices account for 0-based indexing.
        y_slice = slice(_y-1, (_y + _h - 1))
        x_slice = slice(_x-1, (_x + _w - 1))
        # Slightly expand the _c segment's ROI bounding box on the input image.
        # y_slice = slice(_y - 6, (_y + _h + 5))
        # x_slice = slice(_x - 6, (_x + _w + 5))
        roi = self.cvimg['input'][y_slice, x_slice]

        return roi, y_slice, x_slice

    def select_and_export_objects(self) -> None:
        """
        Takes a list of contour segments to select, mask and extract
        each to a bounding rectangle for export of ROI to file.
        Generates a preview window for the first segment, then asks for
        user confirmation to export all selected segments.
        Calls configure_buttons() and from utility_modules/utils:
        no_objects_found_msg(), export_each_segment(), export_object_labels();
        from manage module calls no_objects_found_msg().
        Called from setup_main_menu().

        Returns: None
        """

        # Grab current time to pass to utils.export_each_segment() module.
        #  This is done here, outside the for loop, to avoid having the
        #  export timestamp change (by one or two seconds) during processing.
        # The index count is also passed as an export_each_segment() argument.
        time_now = datetime.now().strftime(const.TIME_STAMP_FORMAT)
        selected_roi_idx = 0

        ok2export = False

        # Need to determine whether the watershed algorithm is in use,
        #  which is the case when the ws control window is visible.
        if self.watershed_window.wm_state() in 'normal, zoomed':
            contour_pointset = self.watershed_basins
        else:  # is 'withdrawn' or 'iconic'
            contour_pointset = self.matte_contours

        # Note that with matte screens, the contour_pointset may contain
        #  a single element of the entire image, if no objects are found.
        if contour_pointset is None or len(contour_pointset) == 1:
            self.update_image(image_name='sized')
            utils.no_objects_found_msg(caller=PROGRAM_NAME)
            return

        # See comments in define_roi() for how to extract segments to a
        #  black or white background. The commented statements below
        #  are for extracting segments instead of an ROI. See size_it.py
        for contour in contour_pointset:
            if self.is_selected_contour(contour=contour):
                selected_roi_idx += 1
                roi, _, _ = self.define_roi(contour=contour)
                # roi, y_slice, x_slice = self.define_roi(contour=contour)
                # roi_mask = self.mask_for_export(contour)[y_slice, x_slice]
                # seg = cv2.bitwise_and(src1=roi, src2=roi, mask=roi_mask)
                # if seg is not None:  # ..and uncomment roi statements as needed.
                if roi is not None:
                    if selected_roi_idx == 1:
                        _scale = 1.5 if max(roi.shape) >= 200 else 3.0
                        roi_img = manage.tk_image(image=roi, scale_factor=_scale)
                        preview_window = self.preview_export(export_img=roi_img)
                        # mask_img = manage.tk_image(image=segment, scale_factor=_scale)
                        # preview_window = self.preview_export(export_img=mask_img)
                        ok2export = messagebox.askokcancel(
                            parent=self,
                            title='Export preview',
                            message='Sampled from selected objects:\n'
                                    f'#{selected_roi_idx}, size = {self.selected_sizes[selected_roi_idx - 1]}.',
                            detail=f'OK: looks good, export all {self.num_objects_selected}'
                                   ' selected objects.\n\n'
                                   'Cancel: export nothing; try different\n'
                                   'noise settings or matte color.')
                        preview_window.destroy()

                        if not ok2export:
                            return

                        utils.export_each_segment(path2folder=self.input_folder_path, img2exp=roi,
                                                  index=selected_roi_idx, timestamp=time_now)

                    # First previewed ROI is exported, now export the rest.
                    # If exporting segments, replace roi with seg variable, from above.
                    utils.export_each_segment(path2folder=self.input_folder_path, img2exp=roi,
                                              index=selected_roi_idx, timestamp=time_now)

                else:
                    # Need to pause between messages in case there are multiple
                    #  problems in the contour pointset.
                    _info = (f'\n\nThere was a problem with segment # {selected_roi_idx},\n'
                             'so it was not exported.\n\n')
                    self.show_info_message(info=_info, color='vermilion')
                    self.after(ms=500)

        if ok2export:
            self.show_info_message(
                info=f'\n{self.num_objects_selected} selected objects were individually\n'
                     ' exported to the input image folder:\n'
                     f'{self.input_folder_name}, including a timestamp.\n'
                     'A CSV file of object detection labels was exported too.\n',
                color='blue')

            utils.export_object_labels(path2input=self.input_file_path,
                                       object_label=self.object_labels,
                                       timestamp=time_now)

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from process_ws(), process_matte(), process_sizes(), process().

        Returns: None
        """

        # Note: recall that *_val dictionaries are inherited from ProcessImage().
        _k: int = self.slider_val['noise_k'].get()
        noise_k = (_k, _k)

        _iter: int = self.slider_val['noise_iter'].get()
        if _iter == 0:
            noise_iter = 'noise reduction not applied'
        else:
            noise_iter = _iter

        morph_op: str = self.cbox_val['morph_op'].get()
        morph_shape: str = self.cbox_val['morph_shape'].get()
        circle_r_min: int = self.slider_val['circle_r_min'].get()
        circle_r_max: int = self.slider_val['circle_r_max'].get()
        color: str = self.cbox_val['matte_color'].get()
        lo, hi = const.MATTE_COLOR_RANGE[color]

        if self.watershed_window.wm_state() in 'normal, zoomed':
            num_segments = len(self.watershed_basins)
            watershed_status = 'Yes'
        else:  # is 'withdrawn' or 'iconic'
            num_segments = len(self.matte_contours)
            num_segments = 0 if num_segments == 1 else num_segments
            watershed_status = 'No'

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
            smallest: str = to_p.to_precision(value=min(self.sorted_size_list),
                                              precision=self.num_sigfig)
            biggest: str = to_p.to_precision(value=max(self.sorted_size_list),
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
            f'\nImage: {self.input_file_path}\n'
            f'Image size, pixels (w x h): {self.input_w}x{self.input_ht}\n'
            f'{divider}\n'
            f'{"Matte color:".ljust(space)}{color}, RGB range: {lo}--{hi}\n'
            f'{"Noise reduction:".ljust(space)}cv2.getStructuringElement ksize={noise_k},\n'
            f'{tab}cv2.getStructuringElement shape={morph_shape}\n'
            f'{tab}cv2.morphologyEx iterations={noise_iter}\n'
            f'{tab}cv2.morphologyEx op={morph_op},\n'
            f'{divider}\n'
            f'{"Watershed segments:".ljust(space)}{watershed_status},\n'
            f'{"# Selected objects:".ljust(space)}{num_selected},'
            f' out of {num_segments} total segments\n'
            f'{"Selected size range:".ljust(space)}{circle_r_min}--{circle_r_max} pixels\n'
            f'{"Selected size std.:".ljust(space)}{size_std}, with a diameter of'
            f' {size_std_size} {unit} units.\n'
            f'{tab}Pixel diameter entered: {self.size_std["px_val"].get()},'
            f' unit/px factor: {unit_per_px}\n'
            f'{"Object size metrics:".ljust(space)}mean = {mean_unit_dia}, median ='
            f' {median_unit_dia}, range = {size_range}'
        )

        utils.display_report(frame=self.report_frame,
                             report=self.report_txt)

    def process_ws(self) -> None:
        """
        Calls matte and watershed segmentation method from ProcessImage(),
        plus methods for annotation style, sizing, and reporting.
        Runs watershed segmentation on the color matted, noise-reduced
        mask image, then runs sizing, and reporting. To be used when the
        input image has objects that are not well separated by the color
        matte alone.
        Called from Button command in setup_watershed_window() and from
        call.cmd().open_watershed_controls().

        Returns: None
        """

        _info = '\n\nRunning Watershed segmentation...\n\n\n'
        self.show_info_message(info=_info, color='blue')

        # Need to first check that entered size values are okay.
        self.validate_px_size_entry()
        if self.cbox_val['size_std'].get() == 'Custom':
            self.validate_custom_size_entry()

        self.widget_control(action='off')

        self.time_start: float = time()

        # The redux (noise reduction) and matte segmentation pre-processing
        #  steps need to run before watershed segmentation. This is done in
        #  self.call_cmd().open_watershed_controls() or self.call_cmd().process(),
        #  which are called by the watershed_window's "Run" button, redux widgets,
        #  or a keyboard shortcut binding.
        self.watershed_segmentation()
        self.select_and_size_objects()

        # Record processing time for info_txt. When no contours are found,
        #  its list has 1 element, so the elapsed time is considered n/a.
        #  The sorted_size_list is cleared when no contours are found,
        #    otherwise would retain the last run's sizes.
        if len(self.watershed_basins) <= 1:
            self.elapsed = 'n/a'
            self.sorted_size_list.clear()
        else:
            self.elapsed = round(time() - self.time_start, 3)

        self.report_results()
        self.widget_control(action='on')

        _info = ('\n\nWatershed segmentation and sizing completed.\n'
                 f'{self.elapsed} processing seconds elapsed.\n\n')
        self.show_info_message(info=_info, color='blue')

        self.delay_size_std_info_msg()

    def process_matte(self, event=None) -> None:
        """
        Calls matte segmentation processing methods from ProcessImage(),
        plus methods for annotation style, sizing, and reporting.

        Args: event - A keyboard event, used to call this method from a
                    lamda function.
        Returns: None
        """

        # Need to clear the watershed_basins tuple of contours when ws not used,
        #  i.e., for reporting number of matte segments in report_results().
        self.watershed_basins = tuple()
        self.watershed_window.withdraw()

        # Need to check that entered size values are okay.
        if not self.first_run:
            self.validate_px_size_entry()
            if self.cbox_val['size_std'].get() == 'Custom':
                self.validate_custom_size_entry()

        # For clarity, disable noise widgets when they are irrelevant,
        #  as when noise reduction iterations are zero.
        self.noise_widget_control()

        self.time_start: float = time()
        self.matte_segmentation()
        self.select_and_size_objects()

        # Record processing time for info_txt. When no contours are found,
        #  its list has 1 element, so the elapsed time is considered n/a.
        #  The sorted_size_list is cleared when no contours are found,
        #    otherwise would retain the last run's sizes.
        if len(self.matte_contours) <= 1:
            self.elapsed = 'n/a'
            self.sorted_size_list.clear()
        else:
            self.elapsed = round(time() - self.time_start, 3)

        self.report_results()

        setting_type = 'Saved' if self.use_saved_settings else 'Default'
        if self.first_run:
            _info = (f'\n\nInitial processing time elapsed: {self.elapsed}\n'
                     f'{setting_type} settings were used.\n\n')
            self.show_info_message(info=_info, color='black')

        else:
            _info = ('\n\nMatte segmentation and sizing completed.\n'
                     f'{self.elapsed} processing seconds elapsed.\n\n')
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
            self.show_info_message(
                info='\n\nUsing a pixel size greater than 1 AND\n'
                     '"None" for a size standard will give\n'
                     'wrong object sizes.\n',
                color='vermilion')
        elif caller == 'circle_r':
            self.show_info_message(
                info='\n\nNew size range for selected objects.\n'
                     'Counts may have changed.\n\n',
                color='blue')
        elif caller in 'size_std, px_entry, custom_entry':
            self.show_info_message(
                info='\n\nNew object sizes calculated.\n\n\n',
                color='blue')
        else:
            print('Oops, an unrecognized process_sizes() caller argument\n'
                  f'was used: caller={caller}')

        self.select_and_size_objects()
        self.report_results()
        self.delay_size_std_info_msg()


class SetupApp(ViewImage):
    """
    The mainloop Class that configures windows and widgets.
    Methods:
    _delete_window_message
    bind_annotation_styles
    bind_focus_actions
    bind_main_commands
    bind_saving_images
    bind_scale_adjustment
    call_cmd
    call_start_now
    check_for_saved_settings
    config_comboboxes
    config_entries
    config_sliders
    configure_buttons
    configure_main_window
    display_windows
    grid_img_labels
    grid_widgets
    open_input
    open_watershed_window
    run_setups
    set_color_defaults
    set_defaults
    setup_image_windows
    setup_main_menu
    setup_main_window
    setup_start_window
    setup_watershed_window
    start_now
    withdraw_watershed_window
    """

    def __init__(self):
        super().__init__()

        # Dictionary items are populated in setup_image_windows(), with
        #   tk.Toplevel as values; don't want tk windows created here.
        self.tkimg_window: dict = {}

        # Attributes defined in setup_main_menu().
        self.menubar = tk.Menu()
        self.menu_labels: tuple = ()

        self.start_process_btn_txt = tk.StringVar()

    def call_cmd(self):
        """
        Groups methods that are shared by buttons, menus, and
        key bind commands in a nested Class.
        Called from setup_main_window(), add_menu_bar(),
        bind_annotation_styles(), bind_scale_adjustment(),
        configure_buttons().
        Usage example: self.call_cmd().save_results()

        Returns: A callable method from the _Command inner class.
        """

        # Inner class concept adapted from:
        # https://stackoverflow.com/questions/719705/
        #   what-is-the-purpose-of-pythons-inner-classes/722175
        cv_colors = list(const.COLORS_CV.keys())

        def _display_annotation_action(action: str, value: str):
            self.show_info_message(
                '\n\nA new annotation style was applied.\n'
                f'{action} was changed to {value}.\n\n',
                color='black')

        def _display_scale_action(value: float):
            """
            The scale_factor is applied in ProcessImage.update_image()
            Called from _Command.increase_scale, _Command.decrease_scale.

            Args:
                 value: the scale factor update to display, as float.
            """
            _sf = round(value, 2)
            self.show_info_message(f'\n\nA new scale factor of {_sf} was applied.\n\n\n',
                                   color='black')

            for _n in const.CS_IMAGE_NAMES:
                self.update_image(image_name=_n)

        class _Command:
            """
            Gives command-based methods access to all script methods and
            instance variables.
            Methods:
            process
            save_results
            new_input
            export_settings
            increase_font_size
            decrease_font_size
            increase_line_thickness
            decrease_line_thickness
            next_font_color
            preceding_font_color
            increase_scale_factor
            decrease_scale_factor
            apply_default_settings
            """

            # These methods are called from configure_buttons() and the
            # "File" menubar of add_menu_bar().

            @staticmethod
            def process():
                """
                Depending on state of the watershed_window watershed controller,
                calls either process_matte() or report_results(), matte_segmentation(),
                noise_widget_control(), with a show_info_message() prompt
                to run watershed segmentation with a button click.
                Called from bindings for watershed_window sliders and main_window
                noise reduction sliders and comboboxes.
                """

                try:
                    if self.watershed_window.state() in 'normal, zoomed':
                        # Update report with current plm_* slider values; don't wait
                        #  for the segmentation algorithm to run before reporting settings.
                        self.report_results()

                        # Note that matte_segmentation() calls reduce_noise(), and
                        #  updates the pre-watershed images for live viewing when
                        #  noise reduction settings change. PLM settings are not
                        #  applied until the "Run watershed" button is clicked.
                        # For visual clarity, disable noise widgets when they are irrelevant,
                        #  as when noise reduction iterations are zero.
                        self.matte_segmentation()
                        self.noise_widget_control()

                        # Preprocessing results have displayed, so the user can
                        # now choose to run watershed.
                        if self.slider_val['plm_footprint'].get() == 1:
                            self.show_info_message(
                                '\nA peak_local_max footprint of 1 may run a long time.\n'
                                'If that is a problem, then increase the value\n'
                                'before clicking the "Run watershed" button.\n\n',
                                color='vermilion')
                        else:
                            self.show_info_message(
                                '\n\nClick "Run watershed" to update\n'
                                'selected and sized objects.\n\n',
                                color='blue')

                    else:  # watershed_window is withdrawn or iconified.
                        self.process_matte()
                except AttributeError:
                    print('From call_cmd().process(), the watershed_window'
                          ' state cannot be determined, so will run process_matte().')
                    self.process_matte()

            @staticmethod
            def save_results():
                """
                Save annotated sized image and its Report text with
                individual object sizes appended.
                Calls utils.save_report_and_img(), show_info_message().
                Called from keybinding, menu, and button commands.
                """
                _sizes = ', '.join(str(i) for i in self.sorted_size_list)
                utils.save_report_and_img(
                    path2folder=self.input_file_path,
                    img2save=self.cvimg['sized'],
                    txt2save=self.report_txt + f'\n{_sizes}',
                    caller=PROGRAM_NAME,
                )
                self.show_info_message(
                    '\n\nSettings report and result image were saved to\n'
                    f'the input image folder: {self.input_folder_name}\n\n',
                    color='blue')

            @staticmethod
            def new_input():
                """
                Reads a new image file for preprocessing.
                Calls open_input(), update_image(), watershed_window.withdraw()
                self & process_matte(), or show_info_message() &
                delay_size_std_info_msg().
                Called from keybinding, menu, and button commands.

                Returns: None
                """

                if self.open_input(parent=self.master):
                    self.check_for_saved_settings()
                    self.update_image(image_name='input')
                    self.watershed_window.withdraw()
                    self.process_matte()
                else:  # User canceled input selection or closed messagebox window.
                    self.show_info_message('\n\nNo new input file was selected.\n\n\n',
                                           color='vermilion')
                    self.delay_size_std_info_msg()

            @staticmethod
            def export_settings():
                settings = {
                    'noise_iter': self.slider_val['noise_iter'].get(),
                    'morph_op': self.cbox_val['morph_op'].get(),
                    'morph_shape': self.cbox_val['morph_shape'].get(),
                    'circle_r_min': self.slider_val['circle_r_min'].get(),
                    'circle_r_max': self.slider_val['circle_r_max'].get(),
                    'noise_k': self.slider_val['noise_k'].get(),
                    'plm_mindist': self.slider_val['plm_mindist'].get(),
                    'plm_footprint': self.slider_val['plm_footprint'].get(),
                    'size_std': self.cbox_val['size_std'].get(),
                    'px_val': self.size_std['px_val'].get(),
                    'custom_val': self.size_std['custom_val'].get(),
                    'annotation_color': self.cbox_val['annotation_color'].get(),
                    'font_scale': self.font_scale,
                    'line_thickness': self.line_thickness,
                    'matte_color': self.cbox_val['matte_color'].get(),
                }

                utils.export_settings_to_json(path2folder=self.input_folder_path,
                                              settings2save=settings,
                                              called_by_cs=True)

                self.show_info_message('\nSettings have been exported to folder:\n'
                                       f'{self.input_folder_name}\n'
                                       'and are available to use with "New input" or at\n'
                                       'startup. Previous settings file was overwritten.\n',
                                       color='blue')

            # These methods are called from the "Style" menu of add_menu_bar()
            #  and as bindings from setup_main_window() or bind_annotation_styles().
            @staticmethod
            def increase_font_size() -> None:
                """Limit upper font size scale to a 5x increase."""
                self.font_scale *= 1.1
                self.font_scale = round(min(self.font_scale, 5), 2)
                self.select_and_size_objects()

            @staticmethod
            def decrease_font_size() -> None:
                """Limit lower font size scale to a 1/5 decrease."""
                self.font_scale *= 0.9
                self.font_scale = round(max(self.font_scale, 0.20), 2)
                self.select_and_size_objects()
                _display_annotation_action('Font scale', f'{self.font_scale}')

            @staticmethod
            def increase_line_thickness() -> None:
                """Limit upper thickness to 15."""
                self.line_thickness += 1
                self.line_thickness = min(self.line_thickness, 15)
                self.select_and_size_objects()
                _display_annotation_action('Line thickness', f'{self.line_thickness}')

            @staticmethod
            def decrease_line_thickness() -> None:
                """Limit lower thickness to 1."""
                self.line_thickness -= 1
                self.line_thickness = max(self.line_thickness, 1)
                self.select_and_size_objects()
                _display_annotation_action('Line thickness', f'{self.line_thickness}')

            @staticmethod
            def next_font_color() -> None:
                """Go to the next color key in const.COLORS_CV.keys."""
                current_color: str = self.cbox_val['annotation_color'].get()
                current_index = cv_colors.index(current_color)
                # Wraps around the list to the first color.
                next_color = (cv_colors[0]
                              if current_index == len(cv_colors) - 1
                              else cv_colors[current_index + 1])
                self.cbox_val['annotation_color'].set(next_color)
                self.select_and_size_objects()
                _display_annotation_action('Font color', f'{next_color}')

            @staticmethod
            def preceding_font_color() -> None:
                """Go to the prior color key in const.COLORS_CV.keys."""
                current_color: str = self.cbox_val['annotation_color'].get()
                current_index = cv_colors.index(current_color)
                # Wraps around the list to the last color.
                preceding_color = (cv_colors[current_index - 1]
                                   if current_index == 0
                                   else cv_colors[current_index - 1])
                self.cbox_val['annotation_color'].set(preceding_color)
                self.select_and_size_objects()
                _display_annotation_action('Font color', f'{preceding_color}')

            @staticmethod
            def increase_scale_factor() -> None:
                """
                Limit upper factor to a 5x increase to maintain performance.
                """
                scale_factor: float = self.scale_factor.get()
                self.scale_factor.set(round(min(scale_factor * 1.1, 5), 2))
                _display_scale_action(value=scale_factor)

            @staticmethod
            def decrease_scale_factor() -> None:
                """
                Limit lower factor to a 1/10 decrease to maintain readability.
                """
                scale_factor: float = self.scale_factor.get()
                self.scale_factor.set(round(max(scale_factor * 0.9, 0.10), 2))
                _display_scale_action(value=scale_factor)

            # This method is called from the "Help" menu of add_menu_bar()
            #  and from configure_buttons() and setup_main_window().
            @staticmethod
            def apply_default_settings():
                """
                Resets settings values and processes images.
                Calls set_auto_scale_factor(), set_defaults(), process_matte(), and
                show_info_message().
                Called from keybinding, menu, and button commands.
                """

                # Order of calls is important here.
                self.set_auto_scale_factor()
                self.set_defaults()
                self.widget_control('off')  # is turned 'on' in process_matte()
                self.process_matte()
                self.show_info_message(
                    '\n\nSettings have been reset to their defaults.\n'
                    'Check and adjust if needed.\n\n', color='blue')

        return _Command

    def run_setups(self):
        """
        Run the setup methods in the correct order.
        Called from main().

        Returns: None
        """
        # For proper menu functions, setup_main_menu() must be called
        #  after setup_main_window().
        self.setup_main_window()
        self.setup_main_menu()
        self.setup_start_window()
        self.setup_watershed_window()

    def call_start_now(self, parent) -> None:
        """
        Call the suite of methods to get things going, then destroy the
        start window.
        Called from setup_start_window() as button and bind commands.

        Args:
            parent: The named toplevel window object, e.g., start_win
        Returns: None.
        """

        # Use a spinning cursor to indicate that something is happening
        #  because larger images may take a while to process and show.
        self.start_process_btn_txt.set('Processing ...')
        parent.config(cursor='watch')
        self.start_now()
        parent.destroy()

    def start_now(self) -> None:
        """
        Initiate the processing pipeline by setting up and configuring
        all settings widgets.
        Called from setup_start_window() with the "Process now...".
        Returns:
            None
        """

        # This calling sequence produces a slight delay (longer for larger files)
        #  before anything is displayed, but ensures that everything displays
        #  nearly simultaneously for a visually cleaner start.
        # set_size_standard() and process_matte() are inherited from ViewImage().
        self.setup_image_windows()
        self.configure_main_window()
        self.configure_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_entries()
        self.bind_annotation_styles()
        self.bind_scale_adjustment()
        self.bind_focus_actions()
        self.bind_saving_images()
        if not self.use_saved_settings:
            self.set_defaults()
            # else, are using settings imported at initial open_input().
        self.grid_widgets()
        self.grid_img_labels()
        self.set_size_standard()

        # Lastly, call process_matte() and display_windows(), in this sequence,
        #  for best performance.
        self.process_matte()
        self.display_windows()
        self.first_run = False

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
        self.geometry(f'+{w_offset}+50')
        self.wm_minsize(width=500, height=450)

        # Need to provide exit info msg to Terminal.
        self.protocol(name='WM_DELETE_WINDOW',
                      func=lambda: utils.quit_gui(mainloop=self))

        self.bind_main_commands(parent=self)

    def setup_main_menu(self) -> None:
        """
        Create main (app) menu instance and hierarchical menus.
        For proper menu functions, must be called in main(), and after
        setup_main_window().

        Returns: None
        """

        # Accelerators use key binds from bind_main_commands() and
        #   bind_annotation_styles() and must be platform-specific.
        os_accelerator = 'Command' if const.MY_OS == 'dar' else 'Ctrl'

        # Unicode arrow symbols: left \u2190, right \u2192, up \u2101, down \u2193
        if const.MY_OS == 'dar':
            color_tip = 'shift-control-↑ & shift-control-↓'
            tip_scaling_text = 'with shift-control-← & shift-control-→.'
            plus_key = '+'
            minus_key = '-'
            watershed_key = 'command-W'
            zoom_accelerator = 'Command-Shift'
        else:
            color_tip = ' Ctrl-↑ & Ctrl-↓'
            tip_scaling_text = 'with Ctrl-← & Ctrl-→.'
            plus_key = '(plus)'
            minus_key = '(minus)'
            watershed_key = 'Ctrl-W'
            zoom_accelerator = 'Ctrl'

        menu_params = dict(
            tearoff=0,
            takefocus=False,
            type='menubar',
            font=const.MENU_FONT,
        )

        # Note: these labels are also used in bind_focus_actions().
        self.menu_labels = ('File', 'Style', 'View', 'Help')

        menu = {_l: tk.Menu(**menu_params) for _l in self.menu_labels}

        for _l in self.menu_labels:
            self.menubar.add_cascade(label=_l, menu=menu[_l])

        menu['File'].add_command(label='Save results',
                                 command=self.call_cmd().save_results,
                                 accelerator=f'{os_accelerator}+S')
        menu['File'].add_command(label='Export objects individually...',
                                 command=self.select_and_export_objects)
        menu['File'].add_command(label='New input...',
                                 command=self.call_cmd().new_input,
                                 accelerator=f'{os_accelerator}+N')
        menu['File'].add_command(label='Export current settings',
                                 command=self.call_cmd().export_settings)
        menu['File'].add(tk.SEPARATOR)
        menu['File'].add_command(label='Quit',
                                 command=lambda: utils.quit_gui(mainloop=self),
                                 # macOS doesn't recognize 'Command+Q' as an accelerator
                                 #   b/c cannot override that system's native Command-Q,
                                 accelerator=f'{os_accelerator}+Q')

        menu['Style'].add_command(label='Increase font size',
                                  command=self.call_cmd().increase_font_size,
                                  accelerator=f'{os_accelerator}+{plus_key}')
        menu['Style'].add_command(label='Decrease font size',
                                  command=self.call_cmd().decrease_font_size,
                                  accelerator=f'{os_accelerator}+{minus_key}')
        menu['Style'].add_command(label='Increase line thickness',
                                  command=self.call_cmd().increase_line_thickness,
                                  accelerator=f'Shift+{os_accelerator}+{plus_key}')
        menu['Style'].add_command(label='Decrease line thickness',
                                  command=self.call_cmd().decrease_line_thickness,
                                  accelerator=f'Shift+{os_accelerator}+{minus_key}')
        menu['Style'].add_command(label='Next color',
                                  command=self.call_cmd().next_font_color,
                                  accelerator=f'{os_accelerator}+↑')
        menu['Style'].add_command(label='Prior color',
                                  command=self.call_cmd().preceding_font_color,
                                  accelerator=f'{os_accelerator}+↓')

        menu['View'].add_command(label='Zoom images out',
                                 command=self.call_cmd().decrease_scale_factor,
                                 accelerator=f'{zoom_accelerator}+←')
        menu['View'].add_command(label='Zoom images in',
                                 command=self.call_cmd().increase_scale_factor,
                                 accelerator=f'{zoom_accelerator}+→')
        # Note that 'Update "Color matte segments"' is needed to just
        #  update the contour color and line thickness of segments image.
        #  Everything else is already up-to-date, but still need to run
        #  process_matte().
        menu['View'].add_command(label='Update "Color matte segments"',
                                 command=self.process_matte,
                                 accelerator=f'{os_accelerator}+M')

        # The Help menu lists a cascade of Tips as a submenu.
        menu['Help'].add_command(label='Improve segmentation...',
                                 command=self.open_watershed_window,
                                 accelerator=f'{os_accelerator}+W')
        menu['Help'].add_command(label='Apply default settings',
                                 command=self.call_cmd().apply_default_settings,
                                 accelerator=f'{os_accelerator}+D')

        tips = tk.Menu(**menu_params)
        menu['Help'].add_cascade(label='Tips...', menu=tips)
        # Bullet symbol from https://coolsymbol.com/, unicode_escape: u'\u2022'
        tip_text = (
            '• Images are auto-zoomed to fit windows at startup.',
            f'     Zoom can be changed with {tip_scaling_text}',
            f'• Font and line color can be changed with {color_tip}.',
            '• Font size can be changed with Ctrl-+(plus) & -(minus).',
            '• Boldness can be changed with Shift-Ctrl-+(plus) & -(minus).',
            '• Matte color selection can affect counts and sizes',
            '      ...so can noise reduction.',
            f'• Try {watershed_key} to separate clustered objects.',
            '• Right-click to save an image at its displayed zoom size.',
            '• Shift-Right-click to save the image at full scale.',
            '• "Export objects" saves each selected object as an image.',
            "• Color and noise reduction settings affect export quality.",
            "• More Tips are in the repository's README file.",
            '• Esc or Ctrl-Q from any window will exit the program.',
        )
        for _line in tip_text:
            tips.add_command(label=_line, font=const.TIPS_FONT)

        menu['Help'].add_command(label='About',
                                 command=utils.about_win)

    def setup_start_window(self) -> None:
        """
        Set up a basic Toplevel, prompt for an input file, set initial matte
        color and initial result annotation color. A button will then trigger
        image processing steps, launch the main window, and destroy the start
        window.

        Returns:
            None
        """

        # Need style of the ttk.Button to match main window button style.
        manage.ttk_styles(mainloop=self)

        file = tk.Menu(master=self.master, tearoff=0)

        # Button text is set to 'Processing started, wait...' in call_start_now().
        self.start_process_btn_txt.set('Process now')

        # Window basics:
        # Open with a temporary, instructional title.
        start_win = tk.Toplevel()
        start_win.title('First, select an input image file')
        start_win.wm_resizable(width=True, height=False)
        start_win.wm_minsize(width=500, height=100)
        start_win.config(**const.WINDOW_PARAMETERS)
        start_win.columnconfigure(index=0, weight=1)
        start_win.columnconfigure(index=1, weight=1)

        # Need to allow complete tk mainloop shutdown from the system's
        #   window manager 'close' icon in the start window bar.
        start_win.protocol(name='WM_DELETE_WINDOW',
                           func=lambda: utils.quit_gui(mainloop=self))

        start_win.bind('<Escape>', lambda _: utils.quit_gui(mainloop=self))
        start_win.bind('<Control-q>', lambda _: utils.quit_gui(mainloop=self))
        start_win.bind('<Return>', func=lambda _: self.call_start_now(start_win))
        start_win.bind('<KP_Enter>', func=lambda _: self.call_start_now(start_win))

        menubar = tk.Menu(master=start_win)
        start_win.config(menu=menubar)

        # Only need "Process now", "Quit", and "About" commands for the
        # start window menu, but need all commands for the main (settings) menu.
        menubar.add_cascade(label=utils.program_name(), menu=file)
        file.add_command(label='Process now',
                         command=lambda: self.call_start_now(parent=start_win),
                         accelerator='Return')  # macOS doesn't recognize 'Enter'
        file.add(tk.SEPARATOR)
        file.add_command(label='Quit',
                         command=lambda: utils.quit_gui(self),
                         # macOS doesn't recognize 'Command+Q' as an accelerator
                         #   b/c cannot override that system's native Command-Q,
                         accelerator=f'{const.C_KEY}+Q')

        help_menu = tk.Menu(master=start_win, tearoff=0)
        menubar.add_cascade(label='Help', menu=help_menu)

        help_menu.add_command(label='About',
                              command=utils.about_win)

        # Window widgets:
        # Provide a placeholder window header for input file info.
        window_header = tk.Label(
            master=start_win,
            text='Image: waiting to be selected...\nSize: TBD',
            **const.LABEL_PARAMETERS)

        color_label = tk.Label(master=start_win,
                               text='Annotation font color:',
                               **const.LABEL_PARAMETERS)
        color_cbox = ttk.Combobox(master=start_win,
                                  values=list(const.COLORS_CV.keys()),
                                  textvariable=self.cbox_val['annotation_color'],
                                  width=11,
                                  height=14,
                                  **const.COMBO_PARAMETERS)

        matte_label = tk.Label(master=start_win,
                               text='Matte color:',
                               **const.LABEL_PARAMETERS)
        matte_cbox = ttk.Combobox(master=start_win,
                                  values=list(const.MATTE_COLOR_RANGE.keys()),
                                  textvariable=self.cbox_val['matte_color'],
                                  width=11,
                                  height=14,
                                  **const.COMBO_PARAMETERS)

        process_now_button = ttk.Button(master=start_win,
                                        textvariable=self.start_process_btn_txt,
                                        style='My.TButton',
                                        width=0,
                                        command=lambda: self.call_start_now(start_win))

        self.set_color_defaults()
        self.cbox_val['matte_color'].set('green2')

        # Grid start win widgets; sorted by row.
        padding = dict(padx=5, pady=5)

        window_header.grid(row=0, column=0, columnspan=2, **padding, sticky=tk.EW)

        color_label.grid(row=2, column=0, **padding, sticky=tk.E)
        color_cbox.grid(row=2, column=1, **padding, sticky=tk.W)

        matte_label.grid(row=3, column=0, **padding, sticky=tk.E)
        matte_cbox.grid(row=3, column=1, **padding, sticky=tk.W)

        process_now_button.grid(row=3, column=1, **padding, sticky=tk.E)

        # Gray-out widget labels until an input file is selected.
        # The settings widgets themselves will be inactive while the
        #  filedialog window is open.
        color_label.config(state=tk.DISABLED)
        matte_label.config(state=tk.DISABLED)

        # Take a break in configuring the window to grab the input.
        # For macOS: Need to have the filedialog be a child of
        #   start_win as 'self'.
        if const.MY_OS == 'dar':
            self.open_input(parent=self.master)
        else:  # is Linux or Windows
            self.open_input(parent=start_win)

        self.check_for_saved_settings()

        # Finally, give start window its active title,...
        start_win.title('Set start parameters')

        # ...fill in window header with full input path and pixel dimensions,...
        window_header.config(text=f'Image: {self.input_file_path}\n'
                                  f'Image size, pixels (w x h): {self.input_w}x{self.input_ht}')

        # ...and make all widgets active.
        color_label.config(state=tk.NORMAL)
        matte_label.config(state=tk.NORMAL)

    def setup_watershed_window(self) -> None:
        """
        Open a window with watershed (local peak) sliders and a run
        watershed button. Includes widget configurations, grids,
        keybindings, and a WM_DELETE_WINDOW protocol.

        Returns: None
        """
        scale_len = int(self.screen_width * 0.20)

        # The watershed_window is hidden at startup, and is deiconified with a keybinding.
        # Note that the system window manager will position the window where it
        #  wants, despite what the geometry string is set to. So, just go with it.
        # To keep things tidy, the self.watershed_window attribute is defined as
        #  watershed_window at the end of the method.
        watershed_window = tk.Toplevel(master=self)
        watershed_window.wm_withdraw()

        # Window configuration is based on the main window, but with a dark bg
        watershed_window.title('Watershed segmentation controls')
        watershed_window.resizable(width=False, height=False)
        watershed_window.columnconfigure(index=0, weight=1)
        watershed_window.columnconfigure(index=1, weight=1)
        watershed_window.config(**const.WINDOW_PARAMETERS)

        # Increase PLM values for larger files to reduce the number
        #  of contours, thus decreasing startup/reset processing time.
        if self.metrics['img_area'] > 6 * 10e5:
            self.slider_val['plm_mindist'].set(200)
            self.slider_val['plm_footprint'].set(9)
        else:
            self.slider_val['plm_mindist'].set(40)
            self.slider_val['plm_footprint'].set(3)

        self.slider['plm_mindist_lbl'] = tk.Label(master=watershed_window,
                                                  text='peak_local_max min_distance:',
                                                  **const.LABEL_PARAMETERS)
        self.slider['plm_mindist'] = tk.Scale(master=watershed_window,
                                              from_=1, to=500,
                                              length=scale_len,
                                              tickinterval=40,
                                              variable=self.slider_val['plm_mindist'],
                                              **const.SCALE_PARAMETERS)
        self.slider['plm_footprint_lbl'] = tk.Label(master=watershed_window,
                                                    text='peak_local_max min_distance:',
                                                    **const.LABEL_PARAMETERS)
        self.slider['plm_footprint'] = tk.Scale(master=watershed_window,
                                                from_=1, to=50,
                                                length=scale_len,
                                                tickinterval=5,
                                                variable=self.slider_val['plm_footprint'],
                                                **const.SCALE_PARAMETERS)

        watershed_button = ttk.Button(master=watershed_window,
                                      text='Run Watershed segmentation',
                                      command=self.process_ws,
                                      width=0,
                                      style='My.TButton')

        self.slider['plm_mindist_lbl'].grid(column=0, row=0,
                                            padx=(10, 5), pady=(10, 0),
                                            sticky=tk.E)
        self.slider['plm_mindist'].grid(column=1, row=0,
                                        padx=(0, 10), pady=(10, 0),
                                        sticky=tk.EW)
        self.slider['plm_footprint_lbl'].grid(column=0, row=1,
                                              padx=(10, 5), pady=(10, 0),
                                              sticky=tk.E)
        self.slider['plm_footprint'].grid(column=1, row=1,
                                          padx=(0, 10), pady=(10, 0),
                                          sticky=tk.EW)
        watershed_button.grid(column=1, row=2,
                              padx=10, pady=10,
                              sticky=tk.EW)

        watershed_window.protocol(name='WM_DELETE_WINDOW',
                                  func=self.withdraw_watershed_window)

        self.watershed_window = watershed_window

    def open_watershed_window(self, event=None) -> None:
        """
        Opens the watershed control window and deactivate the
        "Update color matte segments" button.
        Calls matte_segmentation().
        Called from a keybinding and the Help menu in setup_main_window().

        Args: event: Unused event argument.
        Returns: None
        """

        # Note that the only way to run the watershed algorithm is
        # with the 'Run' button in the watershed_window.
        try:
            self.watershed_window.deiconify()
            self.matte_segmentation()
        except AttributeError:
            print('From call_cmd().open_watershed_controls(), the ws window'
                  ' cannot deiconify or run process_ws.')

        self.button['process_matte'].config(state=tk.DISABLED)

    def withdraw_watershed_window(self) -> None:
        """
        Hide the watershed control window and reactivate the "Update color
        matte segments" button in the main window.
        Returns: None
        """

        self.watershed_window.wm_withdraw()
        self.show_info_message(info='\nWatershed control window was withdrawn.\n'
                                    'Watershed segmentation will not be be used until\n'
                                    f'{f"{const.C_KEY}"}-W brings the window back\n\n',
                               color='black')

        # Reactivate the "Update color matte segments" button in the main window
        # that was disabled when the watershed_window was deiconified.
        self.button['process_matte'].config(state=tk.NORMAL)

    def open_input(self, parent: Union[tk.Toplevel, 'SetupApp']) -> bool:
        """
        Provides an open file dialog to select an initial or new input
        image file. Also sets a scale slider value for the displayed img.
        Called from setup_start_window() and _Commands.new_input via
        "New input" button.
        Args:
            parent: The window or mainloop Class over which to place the
                file dialog, e.g., start_win or self.

        Returns:
            True or False depending on whether input was selected.

        """
        self.input_file_path = filedialog.askopenfilename(
            parent=parent,
            title='Select input image',
            filetypes=[('JPG', '*.jpg'),
                       ('JPG', '*.jpeg'),
                       ('JPG', '*.JPG'),  # used for iPhone images
                       ('PNG', '*.png'),
                       ('TIFF', '*.tiff'),
                       ('TIFF', '*.tif'),
                       ('All', '*.*')],
        )

        # When user selects an input, check whether it can be used by OpenCV.
        # If so, open it, and proceed. If user selects "Cancel" instead of
        #  selecting a file, then quit if at the start window, otherwise
        #  simply close the filedialog (default action) because this was
        #  called from the "New input" button in the mainloop tk toplevel
        #  window (== self.master).
        # Need to call quit_gui() without confirmation b/c a confirmation
        #  dialog answer of "No" throws an error during file input.
        try:
            if self.input_file_path:
                self.cvimg['input'] = cv2.imread(self.input_file_path)
                self.input_ht, self.input_w = cv2.cvtColor(src=self.cvimg['input'],
                                                           code=cv2.COLOR_RGBA2GRAY).shape
                self.input_file_name = Path(self.input_file_path).name
                self.input_folder_path = str(Path(self.input_file_path).parent)
                self.input_folder_name = str(Path(self.input_folder_path).name)
                self.settings_file_path = Path(self.input_folder_path, const.CS_SETTINGS_FILE_NAME)
            elif parent != self.master:
                utils.quit_gui(mainloop=self, confirm=False)
            else:  # no input and parent is self.master (app tk.Toplevel).
                return False
        except cv2.error as cverr:
            msg = f'File: {self.input_file_name} cannot be used.'
            if self.first_run:
                print(f'{msg} Exiting with error:\n{cverr}')
                messagebox.showerror(
                    title="Bad input file",
                    message=msg + '\nRestart and try a different file.\nQuitting...')
                utils.quit_gui(mainloop=self, confirm=False)
            else:
                messagebox.showerror(
                    title="Bad input file",
                    message=msg + '\nUse "New input" to try another file.')
                return False

        # Auto-set images' scale factor based on input image size.
        #  Can be later reset with keybindings in bind_scale_adjustment().
        #  circle_r_slider ranges are a function of input image size.
        self.metrics = manage.input_metrics(img=self.cvimg['input'])
        self.line_thickness = self.metrics['line_thickness']
        self.font_scale = self.metrics['font_scale']
        self.set_auto_scale_factor()
        self.configure_circle_r_sliders()
        return True

    def check_for_saved_settings(self) -> None:
        """
        When open_input() is True, check whether user wants to
        use saved JSON settings file that is in the input image folder.
        Calls import_settings().
        """
        if self.settings_file_path.exists():
            if self.first_run:
                choice = ('Yes: use JSON file in the input folder:\n'
                          f'     {self.input_folder_name}\n'
                          'No (or close window): use default settings')
            else:
                choice = ('Yes: use JSON file in the folder:\n'
                          f'     {self.input_folder_name}\n'
                          'No (or close window): use current settings')

            self.use_saved_settings = messagebox.askyesno(
                title="Use saved settings?",
                detail=choice)

        if self.use_saved_settings:
            self.import_settings()

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
                 'Esc or Ctrl-Q keys can quit the program.\n\n')
        self.show_info_message(info=_info, color='vermilion')

        self.update_idletasks()

        # Give user time to read the _info before resetting it to
        #  the previous info text.
        self.after(ms=7777)
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
        # Each window contains a single Label image, so window title keys
        #  must match CS_IMAGE_NAMES tuple.
        self.tkimg_window = {_n: tk.Toplevel() for _n in const.CS_IMAGE_NAMES}
        titles = ('Input image',
                  'Mask with reduced noise',
                  'Color matte segments',
                  'Selected & Sized objects')
        window_title = dict(zip(const.CS_IMAGE_NAMES, titles))

        # Labels to display scaled images are updated using .configure()
        #  for 'image=' in their respective processing methods via ProcessImage.update_image().
        #  Labels are gridded in their respective tkimg_window in grid_img_labels().
        self.img_label = {_n: tk.Label(self.tkimg_window[_n]) for _n in const.CS_IMAGE_NAMES}

        # Need an image to replace blank tk desktop icon for each img window.
        #  Set correct path to the local 'images' directory and icon file.
        icon_path = None
        try:
            # If the icon file is not present, a Terminal notice will be
            #  printed from main() at startup.
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
            _toplevel.resizable(width=False, height=False)
            _toplevel.protocol(name='WM_DELETE_WINDOW', func=self._delete_window_message)
            _toplevel.columnconfigure(index=0, weight=1)
            _toplevel.columnconfigure(index=1, weight=1)
            _toplevel.rowconfigure(index=0, weight=1)
            _toplevel.title(window_title[_name])
            _toplevel.config(**const.WINDOW_PARAMETERS)

    def configure_main_window(self) -> None:
        """
        Settings and report window (mainloop, self) keybindings,
        configurations, and grids for contour settings and reporting frames.

        Returns:
            None
        """

        self.config(**const.WINDOW_PARAMETERS)
        self.config(bg=const.MASTER_BG)
        self.config(menu=self.menubar)

        # Default Frame() arguments work fine to display report text.
        # bg won't show when grid sticky EW for tk.Text; see utils.display_report().
        self.selectors_frame.configure(relief='raised',
                                       bg=const.DARK_BG,
                                       # bg=const.COLORS_TK['sky blue'],  # for development
                                       borderwidth=5)

        # Allow Frames and widgets to resize with main window.
        #  Row 0 is the report, row1 selectors, rows 2,3,4 are for Buttons().
        self.rowconfigure(index=0, weight=1)
        self.rowconfigure(index=1, weight=1)

        # Keep the report scrollbar active in the resized frame.
        self.report_frame.rowconfigure(index=0, weight=1)

        # Expect there to be 6 rows in the selectors Frame.
        for i in range(7):
            self.selectors_frame.rowconfigure(index=i, weight=1)

        self.columnconfigure(index=0, weight=1)
        self.columnconfigure(index=1, weight=1)
        self.report_frame.columnconfigure(index=0, weight=1)

        # Allow only sliders, not their labels, to expand with window.
        self.selectors_frame.columnconfigure(index=1, weight=1)

        self.report_frame.grid(column=0, row=0,
                               columnspan=2,
                               padx=(5, 5), pady=(5, 5),
                               sticky=tk.NSEW)
        self.selectors_frame.grid(column=0, row=1,
                                  columnspan=2,
                                  padx=5, pady=(0, 5),
                                  ipadx=4, ipady=4,
                                  sticky=tk.NSEW)

        # Width should fit any text expected without causing WINDOW shifting.
        self.info_label.config(font=const.TIPS_FONT,
                               width=50,  # width should fit any text expected without
                               justify='right',
                               bg=const.MASTER_BG,  # use 'pink' for development
                               fg='black')

        # Note: the main window (mainloop, self, app) is deiconified in
        #  display_windows() after all image windows so that, at startup,
        #  it stacks on top.

    def configure_buttons(self) -> None:
        """
        Assign and grid Buttons in the settings (mainloop, self) window.
        Called from __init__.

        Returns:
            None
        """
        manage.ttk_styles(mainloop=self)

        # Configure all items in the dictionary of ttk buttons.
        button_params = dict(
            width=0,
            style='My.TButton',
        )

        self.button['process_matte'].config(
            text='Update "Color matte segments"',
            command=self.process_matte,
            **button_params)

        self.button['save_results'].config(
            text='Save results',
            command=self.call_cmd().save_results,
            **button_params)

        self.button['export_settings'].config(
            text='Export settings',
            command=self.call_cmd().export_settings,
            **button_params)

        self.button['new_input'].config(
            text='New input',
            command=self.call_cmd().new_input,
            **button_params)

        self.button['export_objects'].config(
            text='Export objects',
            command=self.select_and_export_objects,
            **button_params)

        self.button['reset'].config(
            text='Reset',
            command=self.call_cmd().apply_default_settings,
            **button_params)

    def config_sliders(self) -> None:
        """
        Configure arguments and mouse button bindings for all Scale
        widgets in the settings (mainloop, self) window.
        Called from __init__.

        Returns:
            None
        """
        # Minimum width for any Toplevel window is set by the length
        #  of the longest widget, whether that be a Label() or Scale().
        #  So, for the main (app) window, set a Scale() length sufficient
        #  to fit everything in the Frame given current padding arguments.
        #  Keep in mind that a long input file path in the report_frame
        #   may be longer than this set scale_len in the selectors_frame.
        scale_len = int(self.screen_width * 0.20)

        self.slider['noise_k_lbl'].configure(text='... kernel size\n'
                                                  '(can affect object sizes):',
                                             **const.LABEL_PARAMETERS)
        self.slider['noise_k'].configure(from_=1, to=25,
                                         tickinterval=4,
                                         length=scale_len,
                                         variable=self.slider_val['noise_k'],
                                         **const.SCALE_PARAMETERS)

        self.slider['noise_iter_lbl'].configure(text='... iterations\n'
                                                     '(Zero disables noise reduction):',
                                                **const.LABEL_PARAMETERS)

        self.slider['noise_iter'].configure(from_=0, to=4,
                                            tickinterval=1,
                                            variable=self.slider_val['noise_iter'],
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
        # Note that the isinstance() condition doesn't improve performance,
        #  it just clarifies the bind intention.
        for _name, _w in self.slider.items():
            if isinstance(_w, tk.Label):
                continue
            if 'circle_r' in _name:
                _w.bind('<ButtonRelease-1>',
                        func=lambda _: self.process_sizes(caller='circle_r'))
            else:  # is noise_k, noise_iter, plm_footprint, or plm_mindist
                _w.bind('<ButtonRelease-1>', func=lambda _: self.call_cmd().process())

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
        self.cbox['morph_op_lbl'].config(text='Reduce noise, morph. operator:',
                                         **const.LABEL_PARAMETERS)
        self.cbox['morph_op'].config(textvariable=self.cbox_val['morph_op'],
                                     width=16 + width_correction,
                                     values=list(const.CV['morph_op'].keys()),
                                     **const.COMBO_PARAMETERS)

        self.cbox['morph_shape_lbl'].config(text='... shape:',
                                            **const.LABEL_PARAMETERS)
        self.cbox['morph_shape'].config(textvariable=self.cbox_val['morph_shape'],
                                        width=16 + width_correction,
                                        values=list(const.CV['morph_shape'].keys()),
                                        **const.COMBO_PARAMETERS)

        self.cbox['size_std_lbl'].config(text='Select the standard used:',
                                         **const.LABEL_PARAMETERS)
        self.cbox['size_std'].config(textvariable=self.cbox_val['size_std'],
                                     width=12 + width_correction,
                                     values=list(const.SIZE_STANDARDS.keys()),
                                     **const.COMBO_PARAMETERS)

        self.cbox['matte_lbl'].config(text='Select a matte color:',
                                      **const.LABEL_PARAMETERS)
        self.cbox['matte_color'].config(textvariable=self.cbox_val['matte_color'],
                                        width=10 + width_correction,
                                        values=list(const.MATTE_COLOR_RANGE.keys()),
                                        **const.COMBO_PARAMETERS)

        # Now bind functions to all Comboboxes.
        # Note that the isinstance() tk.Label condition isn't needed for
        # performance, it just clarifies the bind intention.
        for _name, _w in self.cbox.items():
            if isinstance(_w, tk.Label):
                continue
            if _name == 'size_std':
                _w.bind('<<ComboboxSelected>>',
                        func=lambda _: self.process_sizes(caller='size_std'))
            else:  # is morph_op or morph_shape
                _w.bind('<<ComboboxSelected>>', func=lambda _: self.call_cmd().process())

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for Entry widgets
        in the settings (mainloop, self) window.
        Called from __init__.

        Returns: None
        """

        self.size_std['px_entry'].config(textvariable=self.size_std['px_val'],
                                         width=5,
                                         font=const.WIDGET_FONT, )
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

    def bind_main_commands(self, parent: Union[tk.Toplevel, 'SetupApp']) -> None:
        """
        Bind key commands to a window specified by *parent*.
        Args:
            parent: Either a Toplevel or the mainloop (self) window.
        Returns: None
        """

        # Note: macOS Command-q will quit program without utils.quit_gui info msg.
        # Need os-specific control keybindings for macOS and Windows/Linux.
        c_key = const.C_BIND
        parent.bind_all('<Escape>', func=lambda _: utils.quit_gui(mainloop=self))
        parent.bind_all('<Control-q>', func=lambda _: utils.quit_gui(mainloop=self))
        parent.bind_all(f'<{f"{c_key}"}-m>',
                        func=self.process_matte)
        parent.bind_all(f'<{f"{c_key}"}-w>',
                        func=self.open_watershed_window)
        parent.bind_all(f'<{f"{c_key}"}-s>',
                        func=lambda _: self.call_cmd().save_results())
        parent.bind_all(f'<{f"{c_key}"}-n>',
                        func=lambda _: self.call_cmd().new_input())
        parent.bind_all(f'<{f"{c_key}"}-d>',
                        func=lambda _: self.call_cmd().apply_default_settings())

    def bind_focus_actions(self) -> None:
        """
        Configure menu bar headings with normal color when main has focus
        and grey-out when not. Called at startup from start_now().
        """

        def _got_focus(_) -> None:
            """The '_' is a placeholder for an event parameter."""
            try:
                for label in self.menu_labels:
                    self.menubar.entryconfig(index=label, state=tk.NORMAL)
            except tk.TclError:
                print('ignoring macOS Tcl entryconfig focus error')

        def _lost_focus(_) -> None:
            for label in self.menu_labels:
                self.menubar.entryconfig(index=label, state=tk.DISABLED)

        # Because we are retaining the macOS default menu bar, the menu
        #  headings are not greyed out when the main window loses focus,
        #  and remain active when any window has focus.
        if const.MY_OS == 'dar':
            self.bind_all('<FocusIn>', _got_focus)
            self.bind_all('<FocusOut>', _lost_focus)
        else:
            self.bind('<FocusIn>', _got_focus)
            self.bind('<FocusOut>', _lost_focus)

    def bind_annotation_styles(self) -> None:
        """
        Set key bindings to change font size, color, and line thickness
        of annotations in the 'sized' cv2 image.
        Called at startup.
        Calls methods from the inner _Command class of self.call_cmd().

        Returns: None
        """

        # Bindings are needed only for the settings and sized img windows,
        #  but is simpler to use bind_all() which does not depend on widget focus.
        # NOTE: On Windows, KP_* is not a recognized keysym string; works on Linux.
        #  Windows keysyms 'plus' & 'minus' are for both keyboard and keypad.
        self.bind_all('<Control-equal>',
                      lambda _: self.call_cmd().increase_font_size())
        self.bind_all('<Control-minus>',
                      lambda _: self.call_cmd().decrease_font_size())
        self.bind_all('<Control-KP_Subtract>',
                      lambda _: self.call_cmd().decrease_font_size())

        self.bind_all('<Shift-Control-plus>',
                      lambda _: self.call_cmd().increase_line_thickness())
        self.bind_all('<Shift-Control-KP_Add>',
                      lambda _: self.call_cmd().increase_line_thickness())
        self.bind_all('<Shift-Control-underscore>',
                      lambda _: self.call_cmd().decrease_line_thickness())

        self.bind_all('<Control-Up>',
                      lambda _: self.call_cmd().next_font_color())
        self.bind_all('<Control-Down>',
                      lambda _: self.call_cmd().preceding_font_color())

        # Need platform-specific keypad keysym.
        if const.MY_OS == 'win':
            self.bind_all('<Control-plus>',
                          lambda _: self.call_cmd().increase_font_size())
            self.bind_all('<Shift-Control-minus>',
                          lambda _: self.call_cmd().decrease_line_thickness())
        else:
            self.bind_all('<Control-KP_Add>',
                          lambda _: self.call_cmd().increase_font_size())
            self.bind_all('<Shift-Control-KP_Subtract>',
                          lambda _: self.call_cmd().decrease_line_thickness())

    def bind_scale_adjustment(self) -> None:
        """
        The displayed image scale is set when an image is imported, but
        can be adjusted in-real-time with these keybindings.

        Returns: None
        """

        self.bind_all('<Control-Right>', lambda _: self.call_cmd().increase_scale_factor())
        self.bind_all('<Control-Left>', lambda _: self.call_cmd().decrease_scale_factor())

    def bind_saving_images(self) -> None:
        """
        Save individual displayed images to file with mouse clicks; either
        at current tkimg scale or at original cvimg scale.

        Returns: None
        """

        # Note: the only way to place these internals in call_cmd._Command is
        #  to add an image=None parameter to call_cmd then pass the image_name
        #  to the on_click,,, methods in _Command.
        #  That seems a bit hacky, so just live with these internals.
        def _on_click_save_tkimg(image_name: str) -> None:
            tkimg = self.tkimg[image_name]

            click_info = (f'The displayed {image_name} image was saved at'
                          f' {self.scale_factor.get()} scale.')

            utils.save_report_and_img(path2folder=self.input_file_path,
                                      img2save=tkimg,
                                      txt2save=click_info,
                                      caller=image_name)

            _info = (f'\nThe result image, "{image_name}", was saved to\n'
                     f'the input image folder: {self.input_folder_name}\n'
                     f'with a timestamp, at a scale of {self.scale_factor.get()}.\n\n')
            self.show_info_message(info=_info, color='black')

        def _on_click_save_cvimg(image_name: str) -> None:
            cvimg = self.cvimg[image_name]

            click_info = (f'\nThe displayed {image_name} image was saved to\n'
                          f'the input image folder: {self.input_folder_name}\n'
                          'with a timestamp, at original pixel dimensions.\n\n')

            utils.save_report_and_img(path2folder=self.input_file_path,
                                      img2save=cvimg,
                                      txt2save=click_info,
                                      caller=image_name)

            self.show_info_message(info=click_info, color='black')

        # Right click will save displayed tk image to file, at current scale.
        # Shift right click saves (processed) cv image at full (original) scale.
        # macOS right mouse button has a different event ID.
        rt_click = '<Button-3>' if const.MY_OS in 'lin, win' else '<Button-2>'
        shift_rt_click = '<Shift-Button-3>' if const.MY_OS in 'lin, win' else '<Shift-Button-2>'
        for name, label in self.img_label.items():
            label.bind(rt_click,
                       lambda _, n=name: _on_click_save_tkimg(image_name=n))
            label.bind(shift_rt_click,
                       lambda _, n=name: _on_click_save_cvimg(image_name=n))

    def set_color_defaults(self):
        """
        Special case to set values for the matte and annotation color
        widgets used in both start and main windows.
        Called from set_start_window() and set_defaults().


        Returns: None
        """

        # use_saved_settings is set in check_for_saved_settings(),
        #  which is called each time open_input() is run.
        # This is where use_saved_settings is reset to False.
        if self.first_run and self.use_saved_settings:
            self.import_settings()
            self.use_saved_settings = False
            return

        # Note that the matte color, cbox_val['matte_color'], starting
        #  default is set in setup_start_window().
        #  It is not set here because it would use that default in the
        #  main window, thus overriding any start window user preference.
        self.cbox_val['annotation_color'].set('yellow')

    def set_defaults(self) -> None:
        """
        Sets and resets selector widgets and keybind scaling functions
        for image sizes and annotations. Evaluates selection of using
        saved or default settings.

        Called from start_now() and apply_default_settings().
        Returns:
            None
        """

        # Default settings are optimized for sample6.jpg input.

        # Set/Reset Scale widgets.
        self.slider_val['noise_k'].set(1)
        self.slider_val['noise_iter'].set(0)
        self.slider_val['circle_r_min'].set(int(self.input_w / 100))
        self.slider_val['circle_r_max'].set(int(self.input_w / 5))

        # Set/Reset Combobox widgets.
        self.cbox_val['morph_op'].set('cv2.MORPH_HITMISS')  # cv2.MORPH_HITMISS == 7
        self.cbox_val['morph_shape'].set('cv2.MORPH_ELLIPSE')  # cv2.MORPH_ELLIPSE == 2
        self.cbox_val['size_std'].set('None')
        self.set_color_defaults()

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

        # Label() widget is in the main window (self).
        # Note: with rowspan=5, there must be 5 return characters in
        #  each info string to prevent shifts of frame row spacing.
        #  Used 5 because that seems to be needed to cover the combined
        #  height of the last three main window rows (2, 3, 4) with buttons.
        #  Sticky is 'east' to prevent horizontal shifting when, during
        #  segmentation processing, all buttons in col 0 are removed.
        self.info_label.grid(column=1, row=2, rowspan=5, columnspan=2,
                             padx=(0, 5), sticky=tk.EW)

        # Widgets gridded in the self.selectors_frame Frame.
        # Sorted by row number:
        self.cbox['morph_op_lbl'].grid(column=0, row=0, **east_grid_params)
        self.cbox['morph_op'].grid(column=1, row=0, **west_grid_params)

        # Note: Put morph shape on same row as morph op.
        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['morph_shape'].grid(column=1, row=0, **east_grid_params)

        self.slider['noise_k_lbl'].grid(column=0, row=1, **east_grid_params)
        self.slider['noise_k'].grid(column=1, row=1, **slider_grid_params)

        self.slider['noise_iter_lbl'].grid(column=0, row=2, **east_grid_params)
        self.slider['noise_iter'].grid(column=1, row=2, **slider_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.slider['circle_r_min_lbl'].grid(column=0, row=3, **east_grid_params)
        self.slider['circle_r_min'].grid(column=1, row=3, **slider_grid_params)

        self.slider['circle_r_max_lbl'].grid(column=0, row=4, **east_grid_params)
        self.slider['circle_r_max'].grid(column=1, row=4, **slider_grid_params)

        self.size_std['px_lbl'].grid(column=0, row=5, **east_grid_params)
        self.size_std['px_entry'].grid(column=1, row=5, **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['size_std'].grid(column=1, row=5, **east_grid_params)

        self.size_std['custom_entry'].grid(column=1, row=6, **east_grid_params)

        self.cbox['matte_lbl'].grid(column=0, row=6, **east_grid_params)
        self.cbox['matte_color'].grid(column=1, row=6, **west_grid_params)

        # Buttons are in the mainloop window, not in a Frame().
        self.button['process_matte'].grid(column=0, row=2, **button_grid_params)
        self.button['save_results'].grid(column=0, row=3, **button_grid_params)
        self.button['export_objects'].grid(column=0, row=4, **button_grid_params)

        # Use update() because update_idletasks() doesn't always work to
        #  get the gridded widgets' correct winfo_reqwidth.
        self.update()

        # Now grid widgets with relative padx values based on widths of
        #  their corresponding partner widgets. Needed across platforms.
        morph_shape_padx = (0, self.cbox['morph_shape'].winfo_reqwidth() + 10)
        size_std_padx = (0, self.cbox['size_std'].winfo_reqwidth() + 10)
        custom_std_padx = (0, self.size_std['custom_entry'].winfo_reqwidth() + 10)
        export_settings_padx = (self.button['save_results'].winfo_reqwidth() + 15, 0)
        export_obj_w: int = self.button['export_objects'].winfo_reqwidth()

        self.cbox['morph_shape_lbl'].grid(column=1, row=0,
                                          padx=morph_shape_padx,
                                          **east_params_relative)

        self.cbox['size_std_lbl'].grid(column=1, row=5,
                                       padx=size_std_padx,
                                       **east_params_relative)

        self.size_std['custom_lbl'].grid(column=1, row=6,
                                         padx=custom_std_padx,
                                         **east_params_relative)

        # Remove initially; show only when Custom size is needed.
        self.size_std['custom_lbl'].grid_remove()

        # Buttons' grids in the mainloop (self) window.
        self.button['export_settings'].grid(
            column=0, row=3,
            padx=export_settings_padx,
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
        Grid all image Labels in the dictionary attribute inherited from
        ProcessImage(). Labels' 'master' argument for the img window are
        defined in setup_image_windows(). A Label's 'image' argument is
        updated with .configure() in PI() update_image().
        Called from __init__.

        Returns:
            None
        """

        for lbl in self.img_label:
            self.img_label[lbl].grid(**const.PANEL_LEFT)

    def display_windows(self) -> None:
        """
        Ready all image window for display. Show the input image in its window.
        Bind rt-click to save any displayed image.
        Called from __init__.
        Calls update_image().

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
        self.update_image(image_name='input')

        # Now is time to show the mainloop (self) report window that
        #  was hidden in setup_main_window().
        #  Deiconifying here stacks it on top of all windows at startup.
        #  Update() here speeds up display of image windows at startup.
        self.wm_deiconify()
        self.update()


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


def main() -> None:
    """
    Main function to launch the program. Initializes SetupApp() and
    sets up the mainloop window and all other windows and widgets.
    Through inheritance, SetupApp() also initializes ProcessImage(),
    which initializes ProcessImage() that inherits Tk, thus creating the
    mainloop window for settings and reporting. With this structure,
    instance attributes and methods are available to all classes only
    where needed.
    """

    # Check system, versions, and command line arguments. Exit if any
    #  critical check fails or if the argument --about is used.
    # Comment out run_checks to run PyInstaller. Windows: run utils.check_platform()
    run_checks()

    print(f'{PROGRAM_NAME} has launched...')
    app = SetupApp()
    app.title(f'{PROGRAM_NAME} Report & Settings')
    utils.set_icon(app)
    app.run_setups()

    # Allow user to quit from the Terminal command line using Ctrl-C
    #  without the delay of waiting for tk event actions.
    # Source: https://stackoverflow.com/questions/39840815/
    #   exiting-a-tkinter-app-with-ctrl-c-and-catching-sigint
    # Keep polling the mainloop to check for the SIGINT signal, Ctrl-C.
    # Can comment out next three statements when using PyInstaller.
    signal(
        signalnum=SIGINT,
        handler=lambda x, y: utils.quit_gui(app, confirm=False)
    )

    def tk_check():
        app.after(500, tk_check)

    app.after(500, tk_check)

    app.mainloop()


if __name__ == '__main__':
    main()
