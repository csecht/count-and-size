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
except (ImportError, ModuleNotFoundError) as import_err:
    sys_exit(
        '*** One or more required Python packages were not found'
        ' or need an update:\nOpenCV-Python, NumPy, tkinter (Tk/Tcl).\n\n'
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
    """

    def __init__(self):
        super().__init__()

        # Note: The matching selector widgets for the following
        #  control variables are in ContourViewer __init__.
        self.slider_val = {
            # For variables in config_sliders()...
            'noise_k': tk.IntVar(),
            'noise_iter': tk.IntVar(),
            'circle_r_min': tk.IntVar(),
            'circle_r_max': tk.IntVar(),
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
        self.image_names = ('input',
                            'redux_mask',
                            'matte_objects',
                            'sized')

        for _name in self.image_names:
            self.tkimg[_name] = tk.PhotoImage()
            self.cvimg[_name] = const.STUB_ARRAY

        # img_label dictionary is set up in SetupApp.setup_image_windows(),
        #  but is used in all Class methods here.
        self.img_label: dict = {}

        # metrics dict is populated in ViewImage.open_input().
        self.metrics: dict = {}
        self.line_thickness: int = 0
        self.font_scale: float = 0

        self.matte_contours: tuple = ()
        self.sorted_size_list: list = []
        self.unit_per_px = tk.DoubleVar()
        self.num_sigfig: int = 0
        self.time_start: float = 0
        self.elapsed: Union[float, int, str] = 0

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

    def reduce_noise(self, img: np.ndarray) -> None:
        """
        Reduce noise in the contrast adjust image erode and dilate actions
        of cv2.morphologyEx operations.
        Called by matte_segmentation(). Calls update_image().

        Args:
            img: The color matte mask from matte_segmentation().
        Returns:
            The *img* array with cv2.morphologyEx applied.
        """

        iteration = self.slider_val['noise_iter'].get()
        if iteration == 0:
            self.cvimg['redux_mask'] = img
            self.update_image(tkimg_name='redux_mask',
                              cvimg_array=self.cvimg['redux_mask'])

            return

        noise_k = self.slider_val['noise_k'].get()

        # Need integers for the cv function parameters.
        morph_shape = const.CV_MORPH_SHAPE[self.cbox_val['morph_shape'].get()]
        morph_op = const.CV_MORPH_OP[self.cbox_val['morph_op'].get()]

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
        # morph_op HITMISS works best for colorseg, with a low kernel size.
        self.cvimg['redux_mask'] = cv2.morphologyEx(
            src=img,
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=cv2.BORDER_DEFAULT,
        )

        self.update_image(tkimg_name='redux_mask',
                          cvimg_array=self.cvimg['redux_mask'])

    def matte_segmentation(self) -> None:
        """
        A segmentation method for use with mattes, e.g., green screen.
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
        lower, upper = const.MATTE_COLOR[self.cbox_val['matte_color'].get()]
        matte_mask = cv2.inRange(src=hsv_img, lowerb=lower, upperb=upper)

        # Run the mask through noise reduction, then use inverse of image for
        #  finding matte_objects contours.
        self.reduce_noise(matte_mask)
        self.cvimg['matte_objects'] = cv2.bitwise_not(self.cvimg['redux_mask'])

        # This list is used in select_and_size_objects() and select_and_export_objects()
        self.matte_contours, _ = cv2.findContours(
            image=np.uint8(self.cvimg['matte_objects']),
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

        self.update_image(tkimg_name='matte_objects',
                          cvimg_array=self.cvimg['matte_objects'])

        # Now need to draw enclosing circles around selected segments and
        #  annotate with object sizes in ViewImage.select_and_size_objects().


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
    _on_click_save_tkimg
    _on_click_save_cvimg
    widget_control
    validate_px_size_entry
    validate_custom_size_entry
    set_size_standard
    select_and_size_objects
    select_and_export_objects
    report_results
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
        self.slider = {

            'noise_k': tk.Scale(master=self.selectors_frame),
            'noise_k_lbl': tk.Label(master=self.selectors_frame),

            'noise_iter': tk.Scale(master=self.selectors_frame),
            'noise_iter_lbl': tk.Label(master=self.selectors_frame),

            'circle_r_min': tk.Scale(master=self.selectors_frame),
            'circle_r_min_lbl': tk.Label(master=self.selectors_frame),

            'circle_r_max': tk.Scale(master=self.selectors_frame),
            'circle_r_max_lbl': tk.Label(master=self.selectors_frame),
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

        self.input_file_path: str = ''
        self.input_file_name: str = ''
        self.input_folder_name: str = ''
        self.input_folder_path: str = ''
        self.input_ht: int = 0
        self.input_w: int = 0

        self.settings_file_path = Path('')
        self.use_saved_settings: bool = False
        self.imported_settings: dict = {}

        self.num_obj_selected: int = 0
        self.report_txt: str = ''

    def set_auto_scale_factor(self) -> None:
        """
        As a convenience for user, set a default scale factor to that
        needed for images to fit easily on the screen, either 1/3
        screen px width or 2/3 screen px height, depending
        on input image orientation.

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
        The dictionary of saved settings, imported via json.loads(),
        that are to be applied to a new image. Includes all settings
        except the scale_factor for window image size.
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
        When no size standard values ar entered, after a few seconds,
        display the size standard instructions in the mainloop (app)
        window. Internal function calls show_info_message().
        Called from process_matte(), process_sizes(), and
        configure_buttons._new_input().

        Returns: None
        """

        def _show_msg() -> None:
            _info = ('When the entered pixel size is 1 AND\n'
                     'size standard is "None", then the size\n'
                     'units shown are pixels. Size units\n'
                     'are mm for any pre-set size standard.\n'
                     f'(Processing time elapsed: {self.elapsed})\n')

            self.show_info_message(info=_info, color='black')

        if (self.size_std['px_val'].get() == '1' and
                self.cbox_val['size_std'].get() == 'None'):
            app.after(ms=6000, func=_show_msg)

    def show_info_message(self, info: str, color: str) -> None:
        """
        Configure for display and update the informational message in
        the report and settings window.
        Args:
            info: The text string of the message to display.
            color: The font color string, either as a key in the
                   const.COLORS_TK dictionary or as a Tk compatible fg
                   color string, i.e. hex code or X11 named color.

        Returns:
            None
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

    def widget_control(self, action: str) -> None:
        """
        Simply show a spinning watch/wheel cursor during the short
        processing time. No need to disable widgets for short intervals.

        Args:
            action: Either 'off' or 'on'.
        """

        if action == 'off':
            self.config(cursor='watch')
        else:  # is 'on'
            self.config(cursor='')

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
        Called from process_matte(), process_sizes(), __init__.

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

    def select_and_size_objects(self, contour_pointset: tuple) -> None:
        """
        Select object contour ROI based on area size and position,
        draw an enclosing circle around contours, then display them
        on the input image. Objects are expected to be oblong so that
        circle diameter can represent the object's length.
        Called by process_matte(), process_sizes(), bind_annotation_styles().
        Calls update_image().

        Args:
            contour_pointset: A tuple of contour coordinate lists
            generated by matte_segmentation().
        Returns:
            None
        """
        self.cvimg['sized'] = self.cvimg['input'].copy()

        # Note that with matte screens, the contour_pointset may contain
        #  a single element of the entire image, if no objects are found.
        if contour_pointset is None or len(contour_pointset) == 1:
            self.update_image(tkimg_name='sized',
                              cvimg_array=self.cvimg['sized'])
            utils.no_objects_found_msg(caller=PROGRAM_NAME)

            return

        selected_sizes: List[float] = []
        annotation_color: tuple = const.COLORS_CV[self.cbox_val['annotation_color'].get()]

        # The size range slider values are radii pixels. This is done b/c:
        #  1) Displayed values have fewer digits, so a cleaner slide bar.
        #  2) Sizes are diameters, so radii are conceptually easier than areas.
        #  So, need to convert to area for the cv2.contourArea function.
        c_area_min = self.slider_val['circle_r_min'].get() ** 2 * np.pi
        c_area_max = self.slider_val['circle_r_max'].get() ** 2 * np.pi

        # Set limits for coordinate points to identify contours that
        # are within 1 px of an image file border (edge).
        bottom_edge = self.input_ht - 1
        right_edge = self.input_w - 1

        self.num_obj_selected = 0

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

            self.num_obj_selected += 1

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
                fontScale=self.font_scale,
                thickness=self.line_thickness)
            offset_x = txt_width / 2

            cv2.circle(img=self.cvimg['sized'],
                       center=(round(_x), round(_y)),
                       radius=round(_r),
                       color=annotation_color,
                       thickness=self.line_thickness,
                       lineType=cv2.LINE_AA,
                       )
            cv2.putText(img=self.cvimg['sized'],
                        text=size2display,
                        org=(round(_x - offset_x), round(_y + baseline)),
                        fontFace=const.FONT_TYPE,
                        fontScale=self.font_scale,
                        color=annotation_color,
                        thickness=self.line_thickness,
                        lineType=cv2.LINE_AA,
                        )

        # The sorted size list is used for reporting individual sizes
        #   and size summary metrics.
        if selected_sizes:
            self.sorted_size_list = sorted(selected_sizes)
        else:
            utils.no_objects_found_msg(caller=PROGRAM_NAME)

        self.update_image(tkimg_name='sized',
                          cvimg_array=self.cvimg['sized'])

    def preview_export(self, export_img, import_img) -> tk.Toplevel:

        w_offset = int(self.screen_width * 0.50)

        sample_win = tk.Toplevel()
        sample_win.title('Sample: for export <- | -> as input')
        sample_win.minsize(400, 150)
        sample_win.geometry(f'+{w_offset}+55')
        sample_win.resizable(True, False)
        sample_win.columnconfigure(index=0, weight=1)
        sample_win.columnconfigure(index=1, weight=1)
        sample_win.config(**const.WINDOW_PARAMETERS)
        sample_win.bind('<Escape>', func=lambda _: utils.quit_gui(self))
        sample_win.bind('<Control-q>', func=lambda _: utils.quit_gui(self))

        l1 = tk.Label(master=sample_win, image=export_img, bg='red')
        l2 = tk.Label(master=sample_win, image=import_img, bg='black')
        l1.grid(**const.PANEL_LEFT)
        l2.grid(**const.PANEL_RIGHT)

        return sample_win

    def select_and_export_objects(self) -> None:
        """
        Takes a list of contour segments, selects, masks and extracts
        each, to a bounding rectangle, for export of ROI to file.
        Calls utility_modules/utils.export_each_segment().
        Called from Button command in configure_buttons().

        Returns: None
        """

        # Grab current time to pass to utils.export_each_segment() module.
        #  This is done here, outside the for loop, to avoid having the
        #  export timestamp change (by one or two seconds) during processing.
        # The index count is also passed as an export_each_segment() argument.
        time_now = datetime.now().strftime('%Y%m%d%I%M%S')
        roi_idx = 0

        # Use the identical selection criteria as in select_and_size_objects().
        c_area_min = self.slider_val['circle_r_min'].get() ** 2 * np.pi
        c_area_max = self.slider_val['circle_r_max'].get() ** 2 * np.pi
        bottom_edge = self.input_ht - 1
        right_edge = self.input_w - 1
        flag = False
        ok2export = False

        for _c in self.matte_contours:

            # As in select_and_size_objects():
            #  Exclude None elements.
            #  Exclude contours not in the specified size range.
            #  Exclude contours that have a coordinate point intersecting
            #   the img edge, that is...
            #   ...those that touch top or left edge or are background.
            #   ...those that touch bottom or right edge.
            if _c is None:
                return
            if not c_area_max > cv2.contourArea(_c) >= c_area_min:
                continue
            if {0, 1}.intersection(set(_c.ravel())):
                continue
            # Break from inner loop when either edge touch is found.
            for point in _c:
                for coord in point:
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

            cv2.drawContours(image=mask,
                             contours=[_c],
                             contourIdx=-1,
                             color=(255, 255, 255),
                             thickness=cv2.FILLED)

            # Note: this contour step provides a cleaner border around the
            #  segment that includes less of the color matte.
            cv2.drawContours(image=mask,
                             contours=[_c],
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

                if roi_idx == 1:
                    mask_img = manage.tk_image(image=result, scale_factor=3.0)
                    roi_img = manage.tk_image(image=roi, scale_factor=3.0)

                    review_window = self.preview_export(export_img=mask_img, import_img=roi_img)

                    ok2export = messagebox.askokcancel(
                        parent=review_window,
                        title='Export preview (3X zoom)',
                        message='Selected objects sample',
                        detail=f'OK: looks good, export all {self.num_obj_selected}'
                               ' selected objects.\n\n'
                               'Cancel: export nothing; try different\n'
                               'noise settings or matte color.')

                    review_window.destroy()

                    if ok2export:
                        continue
                    else:
                        return

                utils.export_each_segment(path2folder=self.input_folder_path,
                                          img2exp=result,
                                          index=roi_idx,
                                          timestamp=time_now)
            else:
                # Need to pause between messages in case there are multiple
                #  problems in the contour pointset.
                _info = (f'\n\nThere was a problem with segment # {roi_idx},\n'
                         'so it was not exported.\n\n')
                self.show_info_message(info=_info, color='vermilion')
                app.after(500)

        if ok2export:
            _info = (f'\n{self.num_obj_selected} selected objects were individually\n'
                     f' exported to the input image folder:\n'
                     f'{self.input_folder_name}, with a timestamp.\n\n')
            self.show_info_message(info=_info, color='blue')

    def report_results(self) -> None:
        """
        Write the current settings and cv metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button.
        Called from process_matte(), process_sizes(), __init__.
        Returns:
            None
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
        color = self.cbox_val['matte_color'].get()
        rgb_range = f'{const.MATTE_COLOR[color][0]}--{const.MATTE_COLOR[color][1]}'

        _num_seg: int = len(self.matte_contours)
        num_matte_segments = 0 if _num_seg == 1 else _num_seg

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
            f'{"Matte color:".ljust(space)}{color}, RGB range: {rgb_range}\n'
            f'{"Noise reduction:".ljust(space)}cv2.getStructuringElement ksize={noise_k},\n'
            f'{tab}cv2.getStructuringElement shape={morph_shape}\n'
            f'{tab}cv2.morphologyEx iterations={noise_iter}\n'
            f'{tab}cv2.morphologyEx op={morph_op},\n'
            f'{divider}\n'
            f'{"# Selected objects:".ljust(space)}{num_selected},'
            f' out of {num_matte_segments} total segments\n'
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

    def process_matte(self) -> None:
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

        _info = '\n\nFinding objects...\n\n\n'
        self.show_info_message(info=_info, color='blue')

        self.widget_control('off')
        self.time_start: float = time()

        self.matte_segmentation()
        self.select_and_size_objects(contour_pointset=self.matte_contours)

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
        self.widget_control('on')

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
            _info = ('\nUsing a pixel size greater than 1 AND\n'
                     '"None" for a size standard will give\n'
                     'wrong object sizes.\n\n')
            self.show_info_message(info=_info, color='vermilion')
        elif caller == 'circle_r':
            _info = ('\n\nNew size range for selected objects.\n'
                     'Counts may have changed.\n\n')
            self.show_info_message(info=_info, color='blue')
        elif caller in 'size_std, px_entry, custom_entry':
            _info = '\n\nNew object sizes calculated.\n\n\n'
            self.show_info_message(info=_info, color='blue')
        else:
            print('Oops, an unrecognized process_sizes() caller argument\n'
                  f'was used: caller={caller}')

        self.select_and_size_objects(contour_pointset=self.matte_contours)
        self.report_results()
        self.delay_size_std_info_msg()


class SetupApp(ViewImage):
    """
    The mainloop Class that configures windows and widgets.
    Methods:
    call_cmd
    call_start
    start_now
    setup_main_window
    setup_start_window
    configure_main_window
    add_menu_bar
    open_input
    check_for_saved_settings
    _delete_window_message
    setup_image_windows
    bind_annotation_styles
    bind_scale_adjustment
    bind_saving_images
    configure_buttons
    config_sliders
    config_comboboxes
    config_entries
    set_color_defaults
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

        # Used only in the start window.
        self.start_process_btn_txt = tk.StringVar()

    def call_cmd(self):
        """
        Groups methods that are shared by buttons, menus, and
        key bind commands in a nested Class.
        Called from setup_main_window(), add_menu_bar(),
        bind_annotation_styles(), bind_scale_adjustment(),
        configure_buttons().
        Usage example: self.call_cmd().save_results()

        Returns: Callable methods in the _Command inner class.
        """

        # Inner class concept adapted from:
        # https://stackoverflow.com/questions/719705/
        #   what-is-the-purpose-of-pythons-inner-classes/722175
        cmd_self = self
        cv_colors = list(const.COLORS_CV.keys())

        def _display_annotation_action(action: str, value: str):
            _info = (f'\n\nA new annotation style was applied.\n'
                     f'{action} was changed to {value}.\n\n')
            self.show_info_message(info=_info, color='black')

        def _apply_new_scale():
            """
            The scale_factor is applied in ProcessImage.update_image()
            Called from _Command.increase_scale, _Command.decrease_scale,
            """
            _sf = round(self.scale_factor.get(), 2)
            _info = f'\n\nA new scale factor of {_sf} was applied.\n\n\n'
            self.show_info_message(info=_info, color='black')

            for _n in self.image_names:
                self.update_image(tkimg_name=_n,
                                  cvimg_array=self.cvimg[_n])

        class _Command:
            """
            Gives command-based methods access to all module methods and
            instance variables.
            Methods:
                save_results
                new_input
                export_settings
                export_objects
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
            def save_results():
                """
                Save annotated sized image and its Report text with
                individual object sizes appended.
                Calls utils.save_report_and_img(), show_info_message().
                Called from keybinding, menu, and button commands.
                """
                _sizes = ', '.join(str(i) for i in cmd_self.sorted_size_list)
                utils.save_report_and_img(
                    path2folder=cmd_self.input_file_path,
                    img2save=cmd_self.cvimg['sized'],
                    txt2save=cmd_self.report_txt + f'\n{_sizes}',
                    caller=PROGRAM_NAME,
                )

                _info = ('\n\nSettings report and result image were saved to\n'
                         f'the input image folder: {cmd_self.input_folder_name}\n\n')
                cmd_self.show_info_message(info=_info, color='blue')

            @staticmethod
            def new_input():
                """
                Reads a new image file for preprocessing.
                Calls open_input(), show_info_message().
                Called from keybinding, menu, and button commands.

                Returns: None
                """
                if cmd_self.open_input(parent=cmd_self):
                    cmd_self.check_for_saved_settings()
                    cmd_self.update_image(tkimg_name='input',
                                          cvimg_array=cmd_self.cvimg['input'])
                else:  # User canceled input selection or closed messagebox window.
                    _info = '\n\nNo new input file was selected.\n\n\n'
                    cmd_self.show_info_message(info=_info, color='vermilion')
                    cmd_self.delay_size_std_info_msg()

                    return

                cmd_self.process_matte()

            @staticmethod
            def export_settings():
                """
                Saves a dictionary of current settings a JSON file.
                Calls utils.export_settings_to_json(), show_info_message().
                Called from menu and button commands.
                """

                settings_dict = {
                    'noise_iter': cmd_self.slider_val['noise_iter'].get(),
                    'morph_op': cmd_self.cbox_val['morph_op'].get(),
                    'morph_shape': cmd_self.cbox_val['morph_shape'].get(),
                    'circle_r_min': cmd_self.slider_val['circle_r_min'].get(),
                    'circle_r_max': cmd_self.slider_val['circle_r_max'].get(),
                    'noise_k': cmd_self.slider_val['noise_k'].get(),
                    'size_std': cmd_self.cbox_val['size_std'].get(),
                    # 'scale': cmd_self.scale_factor.get(),
                    'px_val': cmd_self.size_std['px_val'].get(),
                    'custom_val': cmd_self.size_std['custom_val'].get(),
                    'annotation_color': cmd_self.cbox_val['annotation_color'].get(),
                    'font_scale': cmd_self.font_scale,
                    'line_thickness': cmd_self.line_thickness,
                    'matte_color': cmd_self.cbox_val['matte_color'].get(),
                }

                utils.export_settings_to_json(
                    path2folder=cmd_self.input_folder_path,
                    settings2save=settings_dict,
                    called_by_cs=True)

                _info = ('\nSettings have been exported to folder:\n'
                         f'{cmd_self.input_folder_name}\n'
                         'and are available to use with "New input" or at\n'
                         'startup. Previous settings file was overwritten.\n')
                cmd_self.show_info_message(info=_info, color='blue')

            # These methods are called from the "Style" menu of add_menu_bar()
            #  and as bindings from setup_main_window() or bind_annotation_styles().
            @staticmethod
            def increase_font_size() -> None:
                """Limit upper font size scale to a 5x increase."""
                cmd_self.font_scale *= 1.1
                cmd_self.font_scale = round(min(cmd_self.font_scale, 5), 2)
                cmd_self.select_and_size_objects(contour_pointset=cmd_self.matte_contours)
                _display_annotation_action('Font scale', f'{cmd_self.font_scale}')

            @staticmethod
            def decrease_font_size() -> None:
                """Limit lower font size scale to a 1/5 decrease."""
                cmd_self.font_scale *= 0.9
                cmd_self.font_scale = round(max(cmd_self.font_scale, 0.20), 2)
                cmd_self.select_and_size_objects(contour_pointset=cmd_self.matte_contours)
                _display_annotation_action('Font scale', f'{cmd_self.font_scale}')

            @staticmethod
            def increase_line_thickness() -> None:
                """Limit upper thickness to 15."""
                cmd_self.line_thickness += 1
                cmd_self.line_thickness = min(cmd_self.line_thickness, 15)
                cmd_self.select_and_size_objects(contour_pointset=cmd_self.matte_contours)
                _display_annotation_action('Line thickness', f'{cmd_self.line_thickness}')

            @staticmethod
            def decrease_line_thickness() -> None:
                """Limit lower thickness to 1."""
                cmd_self.line_thickness -= 1
                cmd_self.line_thickness = max(cmd_self.line_thickness, 1)
                cmd_self.select_and_size_objects(contour_pointset=cmd_self.matte_contours)
                _display_annotation_action('Line thickness', f'{cmd_self.line_thickness}')

            @staticmethod
            def next_font_color() -> None:
                """Go to the next color key in const.COLORS_CV.keys."""
                current_color: str = cmd_self.cbox_val['annotation_color'].get()
                current_index = cv_colors.index(current_color)
                # Need to stop increasing idx at the end of colors list.
                if current_index == len(cv_colors) - 1:
                    next_color = cv_colors[len(cv_colors) - 1]
                else:
                    next_color = cv_colors[current_index + 1]
                cmd_self.cbox_val['annotation_color'].set(next_color)
                cmd_self.select_and_size_objects(contour_pointset=cmd_self.matte_contours)
                _display_annotation_action('Font color', f'{next_color}')

            @staticmethod
            def preceding_font_color() -> None:
                """Go to the prior color key in const.COLORS_CV.keys."""
                current_color: str = cmd_self.cbox_val['annotation_color'].get()
                current_index = cv_colors.index(current_color)
                # Need to stop decreasing idx at the beginning of colors list.
                if current_index == 0:
                    current_index = 1
                preceding_color = cv_colors[current_index - 1]
                cmd_self.cbox_val['annotation_color'].set(preceding_color)
                cmd_self.select_and_size_objects(contour_pointset=cmd_self.matte_contours)
                _display_annotation_action('Font color', f'{preceding_color}')

            @staticmethod
            def increase_scale_factor() -> None:
                """
                Limit upper factor to a 5x increase to maintain performance.
                """
                scale_factor: float = self.scale_factor.get()
                scale_factor *= 1.1
                scale_factor = round(min(scale_factor, 5), 2)
                self.scale_factor.set(scale_factor)
                _apply_new_scale()

            @staticmethod
            def decrease_scale_factor() -> None:
                """
                Limit lower factor to a 1/10 decrease to maintain readability.
                """
                scale_factor: float = self.scale_factor.get()
                scale_factor *= 0.9
                scale_factor = round(max(scale_factor, 0.10), 2)
                self.scale_factor.set(scale_factor)
                _apply_new_scale()

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
                cmd_self.set_auto_scale_factor()
                cmd_self.set_defaults()
                cmd_self.widget_control('off')  # is turned 'on' in process_matte()
                cmd_self.process_matte()

                _info = ('\nSettings have been reset to their defaults.\n'
                         'Check and adjust them if needed, then...\n'
                         'Select a "Run" button to finalize updating the\n'
                         'report and image results.\n')
                cmd_self.show_info_message(info=_info, color='blue')

        return _Command

    def call_start(self, parent) -> None:
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
        self.start_process_btn_txt.set('Processing started, wait...')
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
        self.setup_image_windows()
        self.configure_main_window()
        self.configure_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_entries()
        self.bind_annotation_styles()
        self.bind_scale_adjustment()
        self.bind_saving_images()
        if not self.use_saved_settings:
            self.set_defaults()
            # else, are using settings imported at initial open_input().
        self.grid_widgets()
        self.grid_img_labels()
        self.set_size_standard()

        # Run processing for the starting image prior to displaying images.
        # Call process_matte(), and display_windows(), in this sequence, for
        #  best performance. process_matte() is inherited from ViewImage().
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
        w_offset = int(self.screen_width * 0.65)
        self.geometry(f'+{w_offset}+0')
        self.wm_minsize(width=450, height=450)

        # Need to provide exit info msg to Terminal.
        self.protocol(name='WM_DELETE_WINDOW',
                      func=lambda: utils.quit_gui(mainloop=self))

        self.bind('<Escape>', func=lambda _: utils.quit_gui(mainloop=self))
        self.bind('<Control-q>', func=lambda _: utils.quit_gui(mainloop=self))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        c_key = 'Command' if const.MY_OS == 'dar' else 'Control'  # is 'lin' or 'win'.
        self.bind(f'<{f"{c_key}"}-m>', func=lambda _: self.process_matte())
        self.bind(f'<{f"{c_key}"}-s>', func=lambda _: self.call_cmd().save_results())
        self.bind(f'<{f"{c_key}"}-n>', func=lambda _: self.call_cmd().new_input())
        self.bind(f'<{f"{c_key}"}-d>', func=lambda _: self.call_cmd().apply_default_settings())

        self.setup_menu_bar(self)

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

        self.start_process_btn_txt.set('Process now')

        # Window basics:
        # Open with a temporary, instructional title.
        start_win = tk.Toplevel()
        start_win.title('First, select an input image file')
        start_win.wm_resizable(width=True, height=False)
        start_win.wm_minsize(width=500, height=100)
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
        start_win.bind('<Return>', func=lambda _: self.call_start(start_win))
        start_win.bind('<KP_Enter>', func=lambda _: self.call_start(start_win))

        # Window widgets:
        # Provide a placeholder window header for input file info.
        window_header = tk.Label(
            master=start_win,
            text='Image: waiting to be selected...\nSize: TBD',
            **const.LABEL_PARAMETERS)

        self.setup_menu_bar(start_win)

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
                                  values=list(const.MATTE_COLOR.keys()),
                                  textvariable=self.cbox_val['matte_color'],
                                  width=11,
                                  height=14,
                                  **const.COMBO_PARAMETERS)

        process_now_button = ttk.Button(master=start_win,
                                        textvariable=self.start_process_btn_txt,
                                        style='My.TButton',
                                        width=0,
                                        command=lambda: self.call_start(start_win))

        self.set_color_defaults()

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
            self.open_input(parent=self)
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

    def configure_main_window(self) -> None:
        """
        Settings and report window (mainloop, self) keybindings,
        configurations, and grids for contour settings and reporting frames.

        Returns:
            None
        """

        # Color-in the main (self) window and give it a yellow border;
        #  border highlightcolor changes to grey with loss of focus.
        self.config(**const.WINDOW_PARAMETERS)

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
        self.info_label.config(font=const.WIDGET_FONT,
                               width=50,  # width should fit any text expected without
                               justify='right',
                               bg=const.MASTER_BG,  # use 'pink' for development
                               fg='black')

        # Note: the main window (mainloop, self, app) is deiconified in
        #  display_windows() after all image windows so that, at startup,
        #  it stacks on top.

    def setup_menu_bar(self, parent: Union[tk.Toplevel, 'SetupApp']) -> None:
        """
        Create menu instance and add pull-down menus.
        Args:
            parent: The window object to place the menu bar.

        Returns: None
        """

        os_accelerator = 'Command' if const.MY_OS == 'dar' else 'Ctrl'

        # Unicode arrow symbols: left \u2190, right \u2192
        # Unicode arrow symbols: up \u2101, down \u2193
        if const.MY_OS == 'dar':
            color_tip = 'shift-control-↑ & shift-control-↓'
            tip_scaling_text = 'with shift-control-← & shift-control-→.'
            plus_key = '+'
            minus_key = '-'
        else:
            color_tip = ' Ctrl-↑ & Ctrl-↓'
            tip_scaling_text = 'with Ctrl-← & Ctrl-→.'
            plus_key = '(plus)'
            minus_key = '(minus)'

        menubar = tk.Menu(master=parent)
        parent.config(menu=menubar)

        file = tk.Menu(master=self.master, tearoff=0)

        # Only need "Process now", "Quit", and "About" commands for the
        # start window menu, but need all commands for the main (settings) menu.
        if isinstance(parent, tk.Toplevel):
            menubar.add_cascade(label=utils.program_name(), menu=file)
            file.add_command(label='Process now',
                             command=lambda: self.call_start(parent),
                             accelerator='Return')  # macOS doesn't recognize 'Enter'
            file.add(tk.SEPARATOR)
            file.add_command(label='Quit',
                             command=lambda: utils.quit_gui(self),
                             # macOS doesn't recognize 'Command+Q' as an accelerator
                             #   b/c cannot override that system's native Command-Q,
                             accelerator=f'{os_accelerator}+Q')

            help_menu = tk.Menu(master=parent, tearoff=0)
            menubar.add_cascade(label='Help', menu=help_menu)

            help_menu.add_command(label='About',
                                  command=lambda: utils.about_win(parent=parent))

        elif isinstance(parent, SetupApp):
            # Accelerators use key binds from setup_main_window() and
            #   bind_annotation_styles().
            menubar.add_cascade(label='File', menu=file)

            file.add_command(label='Save results',
                             font=const.MENU_FONT,
                             command=self.call_cmd().save_results,
                             accelerator=f'{os_accelerator}+S')
            file.add_command(label='Export objects individually...',
                             font=const.MENU_FONT,
                             command=self.select_and_export_objects)
            file.add_command(label='New input...',
                             font=const.MENU_FONT,
                             command=self.call_cmd().new_input,
                             accelerator=f'{os_accelerator}+N')
            file.add_command(label='Export current settings',
                             font=const.MENU_FONT,
                             command=self.call_cmd().export_settings)
            file.add(tk.SEPARATOR)
            file.add_command(label='Quit',
                             command=lambda: utils.quit_gui(self),
                             # macOS doesn't recognize 'Command+Q' as an accelerator
                             #   b/c cannot override that system's native Command-Q,
                             accelerator=f'{os_accelerator}+Q')

            style = tk.Menu(master=self.master, tearoff=0)
            menubar.add_cascade(label="Annotation styles", menu=style)
            style.add_command(label='Increase font size',
                              font=const.MENU_FONT,
                              command=self.call_cmd().increase_font_size,
                              accelerator=f'{os_accelerator}+{plus_key}')
            style.add_command(label='Decrease font size',
                              font=const.MENU_FONT,
                              command=self.call_cmd().decrease_font_size,
                              accelerator=f'{os_accelerator}+{minus_key}')
            style.add_command(label='Increase line thickness',
                              font=const.MENU_FONT,
                              command=self.call_cmd().increase_line_thickness,
                              accelerator=f'Shift+{os_accelerator}+{plus_key}')
            style.add_command(label='Decrease line thickness',
                              font=const.MENU_FONT,
                              command=self.call_cmd().decrease_line_thickness,
                              accelerator=f'Shift+{os_accelerator}+{minus_key}')
            style.add_command(label='Next color',
                              font=const.MENU_FONT,
                              command=self.call_cmd().next_font_color,
                              accelerator=f'{os_accelerator}+↑')
            style.add_command(label='Prior color',
                              font=const.MENU_FONT,
                              command=self.call_cmd().preceding_font_color,
                              accelerator=f'{os_accelerator}+↓')

            view = tk.Menu(master=self.master, tearoff=0)
            menubar.add_cascade(label="View", menu=view)
            view.add_command(label='Zoom images out',
                             command=self.call_cmd().decrease_scale_factor,
                             font=const.MENU_FONT,
                             accelerator=f'{os_accelerator}+←')
            view.add_command(label='Zoom images in',
                             font=const.MENU_FONT,
                             command=self.call_cmd().increase_scale_factor,
                             accelerator=f'{os_accelerator}+→')
            # Note that 'Update "Color matte segments"' is needed to just
            #  update the contour color and line thickness of segments image.
            #  Everything else is already up-to-date, but still need to run
            #  process_matte().
            view.add_command(label='Update "Color matte segments"',
                             font=const.MENU_FONT,
                             command=self.process_matte,
                             accelerator=f'{os_accelerator}+M')

            help_menu = tk.Menu(master=parent, tearoff=0)
            tips = tk.Menu(master=parent, tearoff=0)
            menubar.add_cascade(label='Help', menu=help_menu)

            help_menu.add_command(label='Apply default settings',
                                  font=const.MENU_FONT,
                                  command=self.call_cmd().apply_default_settings,
                                  accelerator=f'{os_accelerator}+D')

            help_menu.add_cascade(label='Tips...', menu=tips, font=const.MENU_FONT, )

            # Bullet symbol from https://coolsymbol.com/, unicode_escape: u'\u2022'
            tip_text = (
                '• Images are auto-zoomed to fit windows at startup.',
                f'     Zoom can be changed with {tip_scaling_text}',
                f'• Font and line color can be changed with {color_tip}.',
                '• Font size can be changed with Ctrl-+(plus) & -(minus).',
                '• Boldness can be changed with Shift-Ctrl-+(plus) & -(minus).',
                '• Matte color selection can affect counts and sizes',
                '      ...so can noise reduction.',
                '• "Update" button may be needed after changing settings.',
                '• Right-click to save an image at its displayed size.',
                '• Shift-Right-click to save the image at it original size.',
                '• "Export objects" saves each selected object as an image.',
                "• Noise reduction affects exported objects' edge fidelity.",
                '• More Tips are in the repository\'s README file.',
                '• Esc or Ctrl-Q from any window will exit the program.',
            )
            for _line in tip_text:
                tips.add_command(label=_line, font=const.TIPS_FONT)

            help_menu.add_command(label='About',
                                  font=const.MENU_FONT,
                                  command=lambda: utils.about_win(parent=parent))

    def open_input(self, parent: Union[tk.Toplevel, 'SetupApp']) -> bool:
        """
        Provides an open file dialog to select an initial or new input
        image file. Also sets a scale slider value for the displayed img.
        Called from setup_start_window() or "New input" button.
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
        #  called from the "New input" button in the mainloop (self) window.
        # Need to call quit_gui() without confirmation b/c a confirmation
        #  dialog answer of "No" throws an error during file input.

        try:
            if self.input_file_path:
                self.cvimg['input'] = cv2.imread(self.input_file_path)
                self.input_ht = cv2.cvtColor(src=self.cvimg['input'],
                                             code=cv2.COLOR_RGBA2GRAY).shape[0]
                self.input_w = cv2.cvtColor(src=self.cvimg['input'],
                                            code=cv2.COLOR_RGBA2GRAY).shape[1]
                self.input_file_name = Path(self.input_file_path).name
                self.input_folder_path = str(Path(self.input_file_path).parent)
                self.input_folder_name = str(Path(self.input_file_path).parts[-2])
                self.settings_file_path = Path(self.input_folder_path, const.CS_SETTINGS_FILE_NAME)
            elif parent != self:
                utils.quit_gui(mainloop=self, confirm=False)
            else:  # no input and parent is self (app).
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
                 'Esc or Ctrl-Q keys will quit the program.\n\n')
        self.show_info_message(info=_info, color='vermilion')

        self.update()

        # Give user time to read the _info before resetting it to
        #  the previous info text.
        app.after(ms=6000)
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
        # Each window contains a single Label image, so window name matches
        #  image_names tuple, and tkimg dictionary keys.

        self.tkimg_window = {_n: tk.Toplevel() for _n in self.image_names}

        self.window_title = {
            'input': 'Input image',
            'redux_mask': 'Mask with reduced noise',
            'matte_objects': f'Color matte segments',
            'sized': 'Selected & Sized objects',
        }

        # Labels to display scaled images are updated using .configure()
        #  for 'image=' in their respective processing methods via ProcessImage.update_image().
        #  Labels are gridded in their respective tkimg_window in grid_img_labels().
        self.img_label = {_n: tk.Label(self.tkimg_window[_n]) for _n in self.image_names}

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
            _toplevel.resizable(False, False)
            _toplevel.protocol(name='WM_DELETE_WINDOW', func=self._delete_window_message)
            _toplevel.columnconfigure(index=0, weight=1)
            _toplevel.columnconfigure(index=1, weight=1)
            _toplevel.rowconfigure(index=0, weight=1)
            _toplevel.title(self.window_title[_name])
            _toplevel.config(**const.WINDOW_PARAMETERS)
            _toplevel.bind('<Escape>', func=lambda _: utils.quit_gui(self))
            _toplevel.bind('<Control-q>', func=lambda _: utils.quit_gui(self))

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
        #  So, for the main (app) window, set a Scale() length  sufficient
        #  to fit everything in the Frame given current padding arguments.
        #  Keep in mind that a long input file path in the report_frame
        #   may be longer than this set scale_len in the selectors_frame.
        scale_len = int(self.screen_width * 0.20)

        # Scale() widgets for preprocessing (i.e., contrast, noise, filter,
        #  and threshold) or size max/min are called by a mouse
        #  button release. Peak-local-max and circle radius params are
        #  used in process_matte(), which is called by one of the "Run" buttons.

        self.slider['noise_k_lbl'].configure(text='Reduce noise, kernel size\n',
                                             **const.LABEL_PARAMETERS)
        self.slider['noise_k'].configure(from_=1, to=25,
                                         tickinterval=4,
                                         length=scale_len,
                                         variable=self.slider_val['noise_k'],
                                         **const.SCALE_PARAMETERS)

        self.slider['noise_iter_lbl'].configure(text='Reduce noise, iterations\n'
                                                     '(affects segment size):',
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
            else:  # is noise_k or noise_iter
                _w.bind('<ButtonRelease-1>', func=lambda _: self.process_matte())

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
        self.cbox['morph_op_lbl'].config(text='Reduce noise, morphology operator:',
                                         **const.LABEL_PARAMETERS)
        self.cbox['morph_op'].config(textvariable=self.cbox_val['morph_op'],
                                     width=18 + width_correction,
                                     values=list(const.CV_MORPH_OP.keys()),
                                     **const.COMBO_PARAMETERS)

        self.cbox['morph_shape_lbl'].config(text='... shape:',
                                            **const.LABEL_PARAMETERS)
        self.cbox['morph_shape'].config(textvariable=self.cbox_val['morph_shape'],
                                        width=16 + width_correction,
                                        values=list(const.CV_MORPH_SHAPE.keys()),
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
                                        values=list(const.MATTE_COLOR.keys()),
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
                _w.bind('<<ComboboxSelected>>', func=lambda _: self.process_matte())

    def config_entries(self) -> None:
        """
        Configure arguments and mouse button bindings for Entry widgets
        in the settings (mainloop, self) window.
        Called from __init__.

        Returns: None
        """

        self.size_std['px_entry'].config(textvariable=self.size_std['px_val'],
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

        self.cbox_val['matte_color'].set('green2')
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
        self.cbox_val['morph_shape'].set('cv2.MORPH_CROSS')  # cv2.MORPH_CROSS == 1
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
                             padx=5, sticky=tk.EW)

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
        # Note: here and throughout, use 'self' to scope the
        #  ImageTk.PhotoImage image in the Class, otherwise it will/may
        #  not display because of garbage collection.
        self.update_image(tkimg_name='input',
                          cvimg_array=self.cvimg['input'])

        self.update()

        # Now is time to show the mainloop (self) report window that
        #  was hidden in setup_main_window().
        #  Deiconifying here stacks it on top of all windows at startup.
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


if __name__ == '__main__':

    run_checks()

    try:
        print(f'{PROGRAM_NAME} has launched...')
        app = SetupApp()
        app.title(f'{PROGRAM_NAME} Report & Settings')
        # This order of events, when coordinated with the calls in
        #  start_now(), allow macOS implementation to flow well.
        app.setup_main_window()
        app.setup_start_window()

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
