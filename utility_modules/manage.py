"""
General housekeeping utilities.
Functions:
arguments: handles command line arguments
input_metrics: reads specified input image and derives associated metrics.
scale: manages the specified scale factor for display of images.
tk_image: converts scaled cv2 image to a compatible tk.TK image format.
ttk_styles: configures platform-specific ttk.Style for Buttons and Comboboxes.
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License'

# Standard library imports.
import argparse
import sys
import tkinter
from tkinter import ttk

# Third party imports.
import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.ImageTk import PhotoImage

# Local application imports.
# noinspection PyCompatibility
from __main__ import __doc__
import utility_modules
from utility_modules import constants as const


def arguments() -> str:
    """
    Handle command line arguments.
    Returns:
        None
    """

    parser = argparse.ArgumentParser(description='Image Processing to Size Objects.')
    parser.add_argument('--about',
                        help='Provide description, version, GNU license',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    about_text = (f'{__doc__}\n'
                  f'{"Author:".ljust(13)}{utility_modules.__author__}\n'
                  f'{"Version:".ljust(13)}{utility_modules.__version__}\n'
                  f'{"Status:".ljust(13)}{utility_modules.__status__}\n'
                  f'{"URL:".ljust(13)}{utility_modules.URL}\n'
                  f'{utility_modules.__copyright__}'
                  f'{utility_modules.__license__}\n'
                  )

    if args.about:
        print('====================== ABOUT START ====================')
        print(about_text)
        print('====================== ABOUT END ====================')
        sys.exit(0)
    else:  # is called from a widget command for utils.about().
        return about_text


def input_metrics(img: np.ndarray) -> dict:
    """
    Read the image file specified in the --input command line option,
    then calculate and assign to a dictionary values that can be used
    as constants for image file path, processing, and display.

    Returns:
        Dictionary of image values and metrics; keys:'input_img',
        'gray_img', 'font_scale', 'line_thickness', 'max_circle_r'.
    """

    # Scaling factors for contours, circles, and text; empirically determined.
    size_factor = 5.5e-4
    line_thickness_factor = 1.5e-3
    # line_thickness_factor = 2e-03.

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    fig_width: int = gray_img.shape[1]

    if fig_width > const.SIZE_TO_WAIT:
        print(f'Images over {const.SIZE_TO_WAIT} pixels wide will take longer to process...'
              ' patience Grasshopper.\n  If the threshold image shows up as'
              ' black-on-white, then use the inverse option.\n'
              'If the displayed image is too large, reduce the scaling factor.')

    # Set maximum enclosing circle radius to 1/2 the shortest image dimension.
    max_circle_r: int = round(min(gray_img.shape) / 2)

    line_thickness: int = max(round(fig_width * line_thickness_factor), 1)

    # Ideas for scaling: https://stackoverflow.com/questions/52846474/
    #   how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python
    font_scale: float = max(fig_width * size_factor, 0.2)

    metrics = {
        'input_img': img,
        'gray_img': gray_img,
        'font_scale': font_scale,
        'line_thickness': line_thickness,
        'max_circle_r': max_circle_r,
    }

    return metrics


def tk_image(image: np.ndarray, scale_coef: float) -> PhotoImage:
    """
    Scales and converts cv2 images to a compatible format for display
    in tk window. Be sure that the returned image is properly scoped in
    the Class where it is called; e.g., use as self.tk_image attribute.

    Args:
        image: A cv2 numpy array of the image to scale and convert to
               a PIL ImageTk.PhotoImage.
        scale_coef: The user-selected scaling, from start parameters.

    Returns:
        Scaled PIL ImageTk.PhotoImage to display in tk.Label.
    """

    # Need to scale images for display; images for processing are left raw.

    scale_coef = 1 if scale_coef == 0 else scale_coef

    # Provide the best interpolation method for slight improvement of
    #  resized image depending on whether it is down- or up-scaled.
    interpolate = cv2.INTER_AREA if scale_coef < 0 else cv2.INTER_CUBIC

    scaled_img = cv2.resize(src=image,
                            dsize=None,
                            fx=scale_coef, fy=scale_coef,
                            interpolation=interpolate)

    # based on tutorial: https://pyimagesearch.com/2016/05/23/opencv-with-tkinter/
    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)

    scaled_img = Image.fromarray(scaled_img)
    tk_img = ImageTk.PhotoImage(scaled_img)
    # Need to prevent garbage collection to show image in tk.Label, etc.
    tk_img.image = tk_img

    return tk_img


def ttk_styles(mainloop: tkinter.Tk) -> None:
    """
    Configure platform-specific ttk.Style for Buttons and Comboboxes.
    Font and color values need to be edited as appropriate for the
    application (to avoid lengthy parameter arguments).

    Args:
         mainloop: The tk.Toplevel running as the mainloop.

    Returns:
        None
    """

    ttk.Style().theme_use('alt')

    # Use fancy buttons for Linux;
    #   standard theme for Windows and macOS, but with custom font.
    bstyle = ttk.Style()
    combo_style = ttk.Style()

    if const.MY_OS == 'lin':
        font_size = 8
    elif const.MY_OS == 'win':
        font_size = 7
    else:  # is macOS
        font_size = 11

    bstyle.configure("My.TButton", font=('TkTooltipFont', font_size))
    mainloop.option_add("*TCombobox*Font", ('TkTooltipFont', font_size))

    if const.MY_OS == 'lin':
        bstyle.map("My.TButton",
                   foreground=[('active', const.COLORS_TK['yellow'])],
                   background=[('pressed', 'gray30'),
                               ('active', const.COLORS_TK['vermilion'])],
                   )
        combo_style.map('TCombobox',
                        fieldbackground=[('readonly',
                                          const.COLORS_TK['dark blue'])],
                        selectbackground=[('readonly',
                                           const.COLORS_TK['dark blue'])],
                        selectforeround=[('readonly',
                                          const.COLORS_TK['yellow'])],
                        )
    elif const.MY_OS == 'win':
        bstyle.map("My.TButton",
                   foreground=[('active', const.COLORS_TK['yellow'])],
                   background=[('pressed', 'gray30'),
                               ('active', const.COLORS_TK['vermilion'])],
                   )
