#!/usr/bin/env python3
"""
A tkinter GUI, size_it_cs.py, for OpenCV processing of an image to obtain
sizes, means, and ranges of objects in a sample population. Object
segmentation is by use of a matte color screen ('cs'), such as with a
green screen. Different matte colors can be selected. Noise reduction is
interactive with live updating of resulting images.

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

Displayed imaged can be scaled with key commands to help arrange windows
on the screen.

Save settings report and the annotated image with the "Save" button.
Identified objects can be saved to individual image files.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
report window, or from command line with Ctrl-C.

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
from typing import Union

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
            'line_color': tk.StringVar(),
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

        self.matte_contours: list = []
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

        # Need (sort of) kernel to be odd, to avoid an annoying shift of
        #   the displayed image.
        _k = self.slider_val['noise_k'].get()
        noise_k = _k + 1 if _k % 2 == 0 else _k
        iteration = self.slider_val['noise_iter'].get()

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
        # self.cvimg['redux_mask'] = cv2.morphologyEx(
        self.cvimg['redux_mask'] = cv2.morphologyEx(
            src=img,
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=cv2.BORDER_DEFAULT,
        )

        self.update_image(tkimg_name='redux_mask',
                          cvimg_array=self.cvimg['redux_mask'])

        # return redux_mask

    def matte_segmentation(self) -> None:
        """
        A segmentation method for use with mattes, e.g., green screen.
        """

        # The shade of green screen cloth will need different upper and lower
        #  color space bounds.

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

        # Need a thicker line on larger images that are scaled down.
        line_thickness = self.metrics['line_thickness'] * 2

        # Need to prevent black contours because they won't show on the
        #  black mask objects.
        if self.cbox_val['line_color'].get() in 'black, dark blue':
            line_color = const.COLORS_CV['orange']
        else:
            line_color = const.COLORS_CV[self.cbox_val['line_color'].get()]

        cv2.drawContours(image=self.cvimg['matte_objects'],
                         contours=self.matte_contours,
                         contourIdx=-1,  # do all contours
                         color=line_color,
                         thickness=line_thickness,
                         lineType=cv2.LINE_AA)

        self.update_image(tkimg_name='matte_objects',
                          cvimg_array=self.cvimg['matte_objects'])

        # Now need to draw enclosing circles around watershed segments and
        #  annotate with object sizes in ViewImage.select_and_size_objects().


class ViewImage(ProcessImage):
    """
    A suite of methods to display cv segments based on selected settings
    and parameters that are in ProcessImage() methods.
    Methods:
    open_input
    check_for_saved_settings
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
    process
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

        # Screen pixel width is defined in set_auto_scale_factor().
        self.screen_width: int = 0

        # Info label is gridded in configure_main_window().
        self.info_txt = tk.StringVar()
        self.info_label = tk.Label(master=self, textvariable=self.info_txt)

        # Flag user's choice of segment export types. Defined in
        #  configure_buttons() _export_objects() Button cmd.
        self.export_segment: bool = True
        self.export_hull: bool = False

        # Defined in widget_control() to reset values that user may have
        #  tried to change during prolonged processing times.
        self.slider_values: list = []

        self.input_file_path: str = ''
        self.input_file_name: str = ''
        self.input_folder_name: str = ''
        self.input_folder_path: str = ''
        self.input_ht: int = 0
        self.input_w: int = 0

        self.settings_file_path = Path('')
        self.use_saved_settings: bool = False
        self.imported_settings: dict = {}

        self.report_txt: str = ''

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
        self.set_auto_scale_factor()
        self.configure_circle_r_sliders()
        return True

    def check_for_saved_settings(self) -> None:
        """
        Following image file import, need to check whether user wants to
        use saved settings. The JSON settings file is expected to be in
        the input image's folder. Calls import_settings().
        """
        if self.settings_file_path.exists():
            if self.first_run:
                msg = (f'Yes, use settings file in the folder: {self.input_folder_name}.\n'
                       'No, use default settings.')
            else:
                msg = (f'Yes, use settings file in folder: {self.input_folder_name}.\n'
                       'No, use current settings.')

            self.use_saved_settings = messagebox.askyesno(
                # parent=self.focus_get(),
                title=f"Use saved settings on file: {self.input_file_name}?",
                detail=msg)

        if self.use_saved_settings:
            self.import_settings()

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

        self.metrics['font_scale'] = self.imported_settings['font_scale']
        self.metrics['line_thickness'] = self.imported_settings['line_thickness']

        self.size_std['px_val'].set(self.imported_settings['px_val'])
        self.size_std['custom_val'].set(self.imported_settings['custom_val'])

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

    def _on_click_save_tkimg(self, image_name: str) -> None:
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

        utils.save_report_and_img(path2input=self.input_file_path,
                                  img2save=tkimg,
                                  txt2save=click_info,
                                  caller=image_name)

        # Provide user with a notice that a file was created and
        #  give user time to read the message before resetting it.
        _info = (f'\nThe result image, "{image_name}", was saved to\n'
                 f'the input image folder: {self.input_folder_name}\n'
                 f'with a timestamp, at a scale of {self.scale_factor.get()}.\n\n')
        self.show_info_message(info=_info, color='black')

    def _on_click_save_cvimg(self, image_name: str) -> None:
        cvimg = self.cvimg[image_name]

        click_info = (f'\nThe displayed {image_name} image was saved to\n'
                      f'the input image folder: {self.input_folder_name}\n'
                      'with a timestamp and original pixel dimensions.\n\n')

        utils.save_report_and_img(path2input=self.input_file_path,
                                  img2save=cvimg,
                                  txt2save=click_info,
                                  caller=image_name)

        # Provide user with a notice that a file was created and
        #  give user time to read the message before resetting it.
        self.show_info_message(info=click_info, color='black')

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
                              matte_segmentation().
        Returns:
            None
        """

        self.cvimg['sized'] = self.cvimg['input'].copy()

        selected_sizes: list[float] = []
        preferred_color: tuple = const.COLORS_CV[self.cbox_val['line_color'].get()]
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
        bottom_edge = self.input_ht - 1
        right_edge = self.input_w - 1

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

    def select_and_export_objects(self, contour_pointset: list) -> int:
        """
        Takes a list of contour segments, selects, masks and extracts
        each, to a bounding rectangle, for export of ROI to file.
        Calls utility_modules/utils.export_segments().
        Called from Button command in configure_buttons().

        Args:
            contour_pointset: A list of contour coordinates, e.g.,
                generated by matte_segmentation().

        Returns: Integer count of exported objects.
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
        bottom_edge = self.input_ht - 1
        right_edge = self.input_w - 1
        flag = False

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

                utils.export_segments(path2folder=self.input_folder_path,
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
        noise_iter: int = self.slider_val['noise_iter'].get()
        morph_op: str = self.cbox_val['morph_op'].get()
        morph_shape: str = self.cbox_val['morph_shape'].get()
        circle_r_min: int = self.slider_val['circle_r_min'].get()
        circle_r_max: int = self.slider_val['circle_r_max'].get()
        color = self.cbox_val['matte_color'].get()
        rgb_range = f'{const.MATTE_COLOR[color][0]}--{const.MATTE_COLOR[color][1]}'
        num_matte_segments: int = len(self.matte_contours)

        # Only odd kernel integers are used for processing.
        _nk: int = self.slider_val['noise_k'].get()
        if _nk == 0:
            noise_k = 'noise reduction not applied'
        else:
            noise_k = _nk + 1 if _nk % 2 == 0 else _nk

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
        else:
            # Need to populate main matte cbox with start window selection.
            self.cbox['matte_color'].set(self.cbox_val['matte_color'].get())

        _info = '\n\nFinding objects...\n\n\n'
        self.show_info_message(info=_info, color='blue')

        self.widget_control('off')
        self.time_start: float = time()

        self.matte_segmentation()
        self.select_and_size_objects(contour_pointset=self.matte_contours)

        # Record processing time for info_txt.
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
    setup_main_window
    setup_start_window
    start_now
    _delete_window_message
    setup_image_windows
    configure_main_window
    _settings_dict
    configure_buttons
    config_sliders
    config_comboboxes
    config_entries
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
                                  textvariable=self.cbox_val['line_color'],
                                  width=11,
                                  height=14,
                                  **const.COMBO_PARAMETERS)
        color_cbox.current(0)  # blue

        matte_label = tk.Label(master=start_win,
                               text='Matte color:',
                               **const.LABEL_PARAMETERS)
        matte_cbox = ttk.Combobox(master=start_win,
                                  values=list(const.MATTE_COLOR.keys()),
                                  textvariable=self.cbox_val['matte_color'],
                                  width=11,
                                  height=14,
                                  **const.COMBO_PARAMETERS)
        matte_cbox.current(0)  # green1

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
        tips.add_command(label='• Matte color can be changed in the main window.')
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

        matte_label.grid(row=3, column=0, **padding, sticky=tk.E)
        matte_cbox.grid(row=3, column=1, **padding, sticky=tk.W)

        # Best to use cross-platform relative padding of color msg label,
        #  which is placed to the right of the color combobox.
        start_win.update()
        color_padx = (color_cbox.winfo_reqwidth() + 10, 0)
        color_msg_lbl.grid(row=2, column=1,
                           padx=color_padx, pady=5, sticky=tk.W)

        process_now_button.grid(row=3, column=1, **padding, sticky=tk.E)

        # Gray-out widget labels until an input file is selected.
        # The settings widgets themselves will be inactive while the
        #  filedialog window is open.
        color_label.config(state=tk.DISABLED)
        color_msg_lbl.config(state=tk.DISABLED)
        matte_label.config(state=tk.DISABLED)

        # Take a break in configuring the window to grab the input.
        # For macOS: Need to have the filedialog be a child of
        #   start_win and need update() here.

        self.open_input(parent=start_win)
        self.check_for_saved_settings()
        self.update()

        # Finally, give start window its active title,...
        start_win.title('Set start parameters')

        # ...fill in window header with full input path and pixel dimensions,...
        window_header.config(text=f'Image: {self.input_file_path}\n'
                                  f'Image size, pixels (w x h): {self.input_w}x{self.input_ht}')

        # ...and make all widgets active.
        color_label.config(state=tk.NORMAL)
        color_msg_lbl.config(state=tk.NORMAL)
        matte_label.config(state=tk.NORMAL)

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
        self.set_defaults()
        self.grid_widgets()
        self.grid_img_labels()
        self.set_size_standard()

        # Run processing for the starting image prior to displaying images.
        # Call preprocess(), process(), and display_windows(), in this
        #  sequence, for best performance. preprocess() and process()
        #  are inherited from  ViewImage().
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

        # Width should fit any text expected without causing WINDOW shifting.
        self.info_label.config(font=const.WIDGET_FONT,
                               width=50, # width should fit any text expected without
                               justify='right',
                               bg=const.MASTER_BG,  # use 'pink' for development
                               fg='black')

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

        settings_dict = {
            'noise_iter': self.slider_val['noise_iter'].get(),
            'morph_op': self.cbox_val['morph_op'].get(),
            'morph_shape': self.cbox_val['morph_shape'].get(),
            'circle_r_min': self.slider_val['circle_r_min'].get(),
            'circle_r_max': self.slider_val['circle_r_max'].get(),
            'noise_k': self.slider_val['noise_k'].get(),
            'size_std': self.cbox_val['size_std'].get(),
            # 'scale': self.scale_factor.get(),
            'px_val': self.size_std['px_val'].get(),
            'custom_val': self.size_std['custom_val'].get(),
            'line_color': self.cbox_val['line_color'].get(),
            'font_scale': self.metrics['font_scale'],
            'line_thickness': self.metrics['line_thickness'],
            'matte_color': self.cbox_val['matte_color'].get(),
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

        # These inner functions are used for Button commands.

        def _save_results():
            """
            Save annotated sized image and its Report text with
            individual object sizes appended.
            """
            _sizes = ', '.join(str(i) for i in self.sorted_size_list)
            utils.save_report_and_img(
                path2input=self.input_file_path,
                img2save=self.cvimg['sized'],
                txt2save=self.report_txt + f'\n{_sizes}',
                caller=utils.program_name(),
            )

            _info = ('\n\nSettings report and result image was saved to\n'
                     f'the input image folder: {self.input_folder_name}\n\n')
            self.show_info_message(info=_info, color='blue')

        def _export_settings():
            """
            Save only the settings dictionary, as a json file. It is
            handled as a special case in utils.save_report_and_img().
            """

            utils.export_settings_to_json(
                path2folder=self.input_folder_path,
                settings2save=self._settings_dict(),
                called_by_cs=True)

            _info = ('\nSettings have been exported to folder:\n'
                     f'{self.input_folder_name}\n'
                     'and are available to use with "New input" or at\n'
                     'startup. Previous settings file was overwritten.\n')
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

            _num = self.select_and_export_objects(self.matte_contours)
            _info = (f'\n{_num} selected objects were individually\n'
                     f' exported to the input image folder:\n'
                     f'{self.input_folder_name}\n\n')
            self.show_info_message(info=_info, color='blue')

        def _new_input():
            """
            Reads a new image file for preprocessing.
            Calls open_input(), which prompts user for settings choice,
            then calls preprocess().

            Returns: None
            """
            if self.open_input(parent=self):
                self.check_for_saved_settings()
                self.update_image(tkimg_name='input',
                                  cvimg_array=self.cvimg['input'])
            else:  # User canceled input selection or closed messagebox window.
                _info = '\n\nNo new input file was selected.\n\n\n'
                self.show_info_message(info=_info, color='vermilion')
                self.delay_size_std_info_msg()

                return

            self.process()

        def _reset_to_default_settings():
            """Order of calls is important here."""
            self.slider_values.clear()
            self.metrics = manage.input_metrics(img=self.cvimg['input'])
            self.set_auto_scale_factor()
            self.configure_circle_r_sliders()
            self.set_defaults()
            self.widget_control('off')  # is turned 'on' in process()
            self.process()

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

        self.button['process_matte'].config(
            text='Process Color Matte',
            command=self.process,
            width=18,
            style='My.TButton',
        )

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
        #  used in process(), which is called by one of the "Run" buttons.

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
        # Most are bound to preprocess(); process() is called only with
        #  a "Run" Button(). To speed program responsiveness when
        #  changing the size range, only the sizing and reporting methods
        #  are called on mouse button release.
        # Note that the isinstance() condition doesn't improve performance,
        #  it just clarifies the bind intention.
        for _name, _w in self.slider.items():
            if isinstance(_w, tk.Label):
                continue
            if 'circle_r' in _name:
                _w.bind('<ButtonRelease-1>',
                        func=lambda _: self.process_sizes(caller='circle_r'))
            else:  # is noise_k or noise_iter
                _w.bind('<ButtonRelease-1>', func=lambda _: self.process())

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
                _w.bind('<<ComboboxSelected>>', func=lambda _: self.process())

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

    def bind_annotation_styles(self) -> None:
        """
        Set key bindings to change font size, color, and line thickness
        of annotations in the 'sized' cv2 image.
        Called after at startup and after any segmentation algorithm call
        from process().

        Returns: None
        """

        def _display_annotation_action(action: str, value: str):
            _info = (f'\n\nA new annotation style was applied.\n'
                     f'{action} was changed to {value}.\n\n')
            self.show_info_message(info=_info, color='black')

        def _increase_font_size() -> None:
            """Limit upper font size scale to a 5x increase."""
            font_scale: float = self.metrics['font_scale']
            font_scale *= 1.1
            font_scale = round(min(font_scale, 5), 2)
            self.metrics['font_scale'] = font_scale
            self.select_and_size_objects(contour_pointset=self.matte_contours)
            _display_annotation_action('Font scale', f'{font_scale}')

        def _decrease_font_size() -> None:
            """Limit lower font size scale to a 1/5 decrease."""
            font_scale: float = self.metrics['font_scale']
            font_scale *= 0.9
            font_scale = round(max(font_scale, 0.20), 2)
            self.metrics['font_scale'] = font_scale
            self.select_and_size_objects(contour_pointset=self.matte_contours)
            _display_annotation_action('Font scale', f'{font_scale}')

        def _increase_line_thickness() -> None:
            """Limit upper thickness to 15."""
            line_thickness: int = self.metrics['line_thickness']
            line_thickness += 1
            self.metrics['line_thickness'] = min(line_thickness, 15)
            self.select_and_size_objects(contour_pointset=self.matte_contours)
            _display_annotation_action('Line thickness', f'{line_thickness}')

        def _decrease_line_thickness() -> None:
            """Limit lower thickness to 1."""
            line_thickness: int = self.metrics['line_thickness']
            line_thickness -= 1
            self.metrics['line_thickness'] = max(line_thickness, 1)
            self.select_and_size_objects(contour_pointset=self.matte_contours)
            _display_annotation_action('Line thickness', f'{line_thickness}')

        colors = list(const.COLORS_CV.keys())

        def _next_font_color() -> None:
            current_color: str = self.cbox_val['line_color'].get()
            current_index = colors.index(current_color)
            # Need to stop increasing idx at the end of colors list.
            if current_index == len(colors) - 1:
                next_color = colors[len(colors) - 1]
            else:
                next_color = colors[current_index + 1]
            self.cbox_val['line_color'].set(next_color)
            # print('Annotation font is now:', next_color)
            self.select_and_size_objects(contour_pointset=self.matte_contours)
            _display_annotation_action('Font color', f'{next_color}')

        def _preceding_font_color() -> None:
            current_color: str = self.cbox_val['line_color'].get()
            current_index = colors.index(current_color)
            # Need to stop decreasing idx at the beginning of colors list.
            if current_index == 0:
                current_index = 1
            preceding_color = colors[current_index - 1]
            self.cbox_val['line_color'].set(preceding_color)
            print('Annotation font is now :', preceding_color)
            self.select_and_size_objects(contour_pointset=self.matte_contours)
            _display_annotation_action('Font color', f'{preceding_color}')

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

        def _apply_new_scale():
            """
            The scale_factor is applied in ProcessImage.update_image()
            """
            _sf = round(self.scale_factor.get(), 2)
            _info = f'\n\nA new scale factor of {_sf} was applied.\n\n\n'
            self.show_info_message(info=_info, color='black')

            for _n in self.image_names:
                self.update_image(tkimg_name=_n,
                                  cvimg_array=self.cvimg[_n])

        def _increase_scale_factor() -> None:
            """
            Limit upper factor to a 5x increase to maintain performance.
            """
            scale_factor: float = self.scale_factor.get()
            scale_factor *= 1.1
            scale_factor = round(min(scale_factor, 5), 2)
            self.scale_factor.set(scale_factor)
            _apply_new_scale()

        def _decrease_scale_factor() -> None:
            """
            Limit lower factor to a 1/10 decrease to maintain readability.
            """
            scale_factor: float = self.scale_factor.get()
            scale_factor *= 0.9
            scale_factor = round(max(scale_factor, 0.10), 2)
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

        if self.first_run and self.use_saved_settings:
            self.import_settings()
            return

        # Set/Reset Scale widgets.
        self.slider_val['noise_k'].set(3)
        self.slider_val['noise_iter'].set(1)
        self.slider_val['circle_r_min'].set(20)
        self.slider_val['circle_r_max'].set(600)

        # Set/Reset Combobox widgets.
        self.cbox_val['morph_op'].set('cv2.MORPH_HITMISS')  # cv2.MORPH_OPEN == 2
        self.cbox_val['morph_shape'].set('cv2.MORPH_CROSS')  # cv2.MORPH_ELLIPSE == 2
        self.cbox_val['size_std'].set('None')
        self.cbox_val['matte_color'].set('green1')
        self.cbox_val['line_color'].set('blue')

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
        #  5 because that seems to be needed to cover the combined
        #  height of the last three main window rows (2, 3, 4) with buttons.
        #  Sticky is 'east' to prevent horizontal shifting when, during
        #  segmentation processing, all buttons in col 0 are removed.
        self.info_label.grid(column=1, row=2, rowspan=5, columnspan=2,
                             padx=5, sticky=tk.E)

        # Widgets gridded in the self.selectors_frame Frame.
        # Sorted by row number:
        self.cbox['morph_op_lbl'].grid(column=0, row=2, **east_grid_params)
        self.cbox['morph_op'].grid(column=1, row=2, **west_grid_params)

        # Note: Put morph shape on same row as morph op.
        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['morph_shape'].grid(column=1, row=2, **east_grid_params)

        self.slider['noise_k_lbl'].grid(column=0, row=4, **east_grid_params)
        self.slider['noise_k'].grid(column=1, row=4, **slider_grid_params)

        self.slider['noise_iter_lbl'].grid(column=0, row=5, **east_grid_params)
        self.slider['noise_iter'].grid(column=1, row=5, **slider_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.slider['circle_r_min_lbl'].grid(column=0, row=17, **east_grid_params)
        self.slider['circle_r_min'].grid(column=1, row=17, **slider_grid_params)

        self.slider['circle_r_max_lbl'].grid(column=0, row=18, **east_grid_params)
        self.slider['circle_r_max'].grid(column=1, row=18, **slider_grid_params)

        self.size_std['px_lbl'].grid(column=0, row=19, **east_grid_params)
        self.size_std['px_entry'].grid(column=1, row=19, **west_grid_params)

        # The label widget is gridded to the left, based on this widget's width.
        self.cbox['size_std'].grid(column=1, row=19, **east_grid_params)
        self.size_std['custom_entry'].grid(column=1, row=20, **east_grid_params)

        self.cbox['matte_lbl'].grid(column=0, row=20, **east_grid_params)
        self.cbox['matte_color'].grid(column=1, row=20, **west_grid_params)

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
        export_obj_w: int = self.button['export_objects'].winfo_reqwidth()
        save_results_w: int = self.button['save_results'].winfo_reqwidth()

        self.cbox['morph_shape_lbl'].grid(column=1, row=2,
                                          padx=morph_shape_padx,
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

        # Right click will save displayed tk image to file, at current scale.
        # Shift right click saves (processed) cv image at full (original) scale.
        # macOS right mouse button has a different event ID.
        rt_click = '<Button-3>' if const.MY_OS in 'lin, win' else '<Button-2>'
        shift_rt_click = '<Shift-Button-3>' if const.MY_OS in 'lin, win' else '<Shift-Button-2>'
        for name, label in self.img_label.items():
            label.bind(rt_click,
                       lambda _, n=name: self._on_click_save_tkimg(image_name=n))
            label.bind(shift_rt_click,
                       lambda _, n=name: self._on_click_save_cvimg(image_name=n))
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