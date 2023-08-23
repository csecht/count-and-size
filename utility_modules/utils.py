"""
General housekeeping utilities.
Functions:
check_platform - Exit if not Linux, Windows, or macOS.
valid_path_to - Get correct path to program's files.
save_settings_and_img- Save files of result image and its settings.
display_report - Place a formatted text string into a specified Frame.
count_sig_fig - Count number of significant figures in a number.
text_array - Generate an image array of text.
quit_keys -  Error-free and informative exit from the program.
no_objects_found - A simple message box when a contour pointset is empty.
"""
# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License'

# Standard library imports.
import platform
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Union

# Third party imports.
import cv2
import numpy as np
from PIL import ImageTk

# Local application imports.
from utility_modules import (constants as const)


def check_platform() -> None:
    """
    Run check for various platforms to optimize displays.
    Intended to be called at startup.
    """
    if const.MY_OS == 'dar':
        print('Developed in macOS 13; earlier versions may not work.\n')

    # Need to account for scaling in Windows10 and earlier releases.
    elif const.MY_OS == 'win':
        from ctypes import windll

        if platform.release() < '10':
            windll.user32.SetProcessDPIAware()
        else:
            windll.shcore.SetProcessDpiAwareness(2)

    print('To quit, use Esc or Ctrl-Q. From the Terminal, use Ctrl-C.')


def valid_path_to(input_path: str) -> Path:
    """
    Get correct path to program's directory/file structure
    depending on whether program invocation is a standalone app or
    the command line. Works with symlinks. Works with absolute paths
    outside of program's folder.
    Allows command line invocation using any path; does not need to be
    from parent directory.
    _MEIPASS var is used by distribution programs from
    PyInstaller --onefile; e.g. for images dir.

    :param input_path: Program's local dir/file name, as string.
    :return: Absolute path as pathlib Path object.
    """
    # Note that Path(Path(__file__).parent is the contours_modules folder.
    # Modified from: https://stackoverflow.com/questions/7674790/
    #    bundling-data-files-with-pyinstaller-onefile and PyInstaller manual.
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        base_path = getattr(sys, '_MEIPASS', Path(Path(__file__).resolve()).parent)
        valid_path = Path(base_path) / input_path
    else:
        # NOTE: this path only works for images in the program images folder.
        if 'images/' in input_path:
            valid_path = Path(Path(__file__).parent, f'../{input_path}').resolve()
        else:  # A path outside the Project was used for the input file.
            valid_path = Path(Path(f'{input_path}')).resolve()

    return valid_path


def save_settings_and_img(inputpath: str,
                          img2save,
                          txt2save: str,
                          caller: str) -> None:
    """
    Print to terminal/console and to file current settings and
    calculated image processing values. Save current result image.

    Args:
        inputpath: The input image file path, as string.
        img2save: The current resulting image array; can be a np.ndarray
            from cv2 or an ImageTk.PhotoImage from tkinter/PIL
        txt2save: The current image processing settings.
        caller: Descriptive name of the calling app to insert in the
                file names, e.g. 'sizeit'.

    Returns: None
    """

    curr_time = datetime.now().strftime('%I%M%S')
    time2print = datetime.now().strftime('%I:%M:%S%p')

    # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
    # with a possible value between 0 and 100, the default value being 95.
    # The higher value produces a better quality image file.
    #
    # For PNG file format the supported imwrite parameter is
    # cv2.IMWRITE_PNG_COMPRESSION with a possible value between 0 and 9,
    # the default being 3. The higher value does high compression of the
    # image resulting in a smaller file size but a longer compression time.

    img_ext = Path(Path(inputpath).suffix)
    img_stem = Path(Path(inputpath).stem)

    data2save = (f'\n\nTime saved: {time2print}\n'
                 f'Saved image file: {img_stem}_{caller}_{curr_time}{img_ext}\n'
                 f'Saved settings file: {img_stem}_{caller}_settings.txt\n'
                 f'{txt2save}')

    # Use this Path function for saving individual settings files:
    # Path(f'{img_stem}_clahe_settings{curr_time}.txt').write_text(data2save)
    # Use this for appending multiple settings to single file:
    with Path(f'{img_stem}_{caller}_settings.txt').open('a', encoding='utf-8') as _fp:
        _fp.write(data2save)

    # Contour images are np.ndarray direct from cv2 functions, while
    #   other images are those displayed as ImageTk.PhotoImage.
    if isinstance(img2save, np.ndarray):
        # if first_word == 'Image:':  # text is from contoured_txt
        file_name = f'{img_stem}_{caller}_{curr_time}{img_ext}'
        cv2.imwrite(file_name, img2save)
    elif isinstance(img2save, ImageTk.PhotoImage):
        # Need to get the ImageTK image into a format that can be saved to file.
        # source: https://stackoverflow.com/questions/45440746/
        #   how-to-save-pil-imagetk-photoimage-as-jpg
        imgpil = ImageTk.getimage(img2save)

        # source: https://stackoverflow.com/questions/48248405/
        #   cannot-write-mode-rgba-as-jpeg
        if imgpil.mode in ("RGBA", "P"):
            imgpil = imgpil.convert("RGB")

        img_name = Path(f'{img_stem}_{caller}_{curr_time}{img_ext}')
        imgpil.save(img_name)
    else:
        print('The specified image needs to be a np.ndarray or ImageTk.PhotoImage ')

    print(f'Result image and its settings were saved to files.'
          f'{data2save}')


def display_report(frame: tk.Frame, report: str) -> None:
    """
    Places a formatted text string into the specified Frame; allows for
    real-time updates of text and proper alignment of text in the Frame.

    Args:
        frame: The tk.Frame() in which to place the *report* text.
        report: Text string of values, data, etc. to report.

    Returns: None
    """

    max_line = len(max(report.splitlines(), key=len))

    if const.MY_OS == 'lin':
        txt_font = ('Courier', 10)
    elif const.MY_OS == 'win':
        txt_font = ('Courier', 8)
    else:  # is macOS
        txt_font = ('Courier', 10)

    # Note: 'TkFixedFont' only works when not in a tuple, so no font size.
    #  The goal is to get a suitable platform-independent mono font.
    #  font=('Courier', 10) should also work, if need to set font size.
    #  Smaller fonts are needed to shorten the window as lines & rows are added.
    #  With smaller font, need better fg font contrast, e.g. yellow, not MASTER_BG.
    reporttxt = tk.Text(frame,
                        # font='TkFixedFont',
                        font=txt_font,
                        bg=const.DARK_BG,
                        # fg=const.MASTER_BG,  # gray80 matches master self bg.
                        fg=const.COLORS_TK['yellow'],  # Matches slider labels.
                        width=max_line,
                        height=report.count('\n'),
                        relief='flat',
                        padx=8, pady=8,
                        )
    # Replace prior Text with current text;
    #   hide cursor in Text; (re-)grid in-place.
    reporttxt.delete('1.0', tk.END)
    reporttxt.insert(tk.INSERT, report)
    # Indent helps center text in the Frame.
    reporttxt.tag_config('leftmargin', lmargin1=20)
    reporttxt.tag_add('leftmargin', '1.0', tk.END)
    reporttxt.configure(state=tk.DISABLED)

    reporttxt.grid(column=0, row=0,
                   columnspan=2,
                   sticky=tk.EW)


def count_sig_fig(entry_number: Union[int, float, str]) -> int:
    """
    Determine the number of significant figures in a number.
    Be sure to verify that *entry_number* is a real number prior to calling.
    Args:
        entry_number: Any numerical representation, as string or digits.

    Returns: Integer count of significant figures in *entry_number*.

    """

    # See: https://en.wikipedia.org/wiki/Significant_figures#Significant_figures_rules_explained
    # The num_sigfig value calculated here is generally used as the
    #  'precision' parameter in to_p.to_precision() statements.
    entry_number = str(entry_number)

    # Remove non-numeric characters
    trimmed_number = ''
    trimmed_number = ''.join([trimmed_number + _c for _c in entry_number if _c.isnumeric()])

    # If scientific notation, remove the exponent value, assuming
    #  that it is only a single digit.
    if {'e', 'E'}.intersection(set(entry_number)):
        trimmed_number = trimmed_number[:-1]

    # Finally, remove leading zeros, which are not significant, and
    #  determine number of significant figures.
    return len(trimmed_number.lstrip('0'))


def quit_gui(mainloop: tk.Tk) -> None:
    """Safe and informative exit from the program.

    Args:
        mainloop: The main tk.Tk() window running the mainloop.

    Returns: None
    """
    print('\n  *** User has quit the program. ***')

    try:
        mainloop.update_idletasks()
        mainloop.after(100)
        mainloop.destroy()
        # Need explicit exit if for some reason a tk window isn't destroyed.
        sys.exit(0)
    # pylint: disable=broad-except
    except Exception as unk:
        print('An unknown error occurred:', unk)
        sys.exit(0)


def no_objects_found_msg():
    """
    Pop-up info when segments not found or their sizes out of range.
    """
    _m = ('No objects were found to size. Try changing threshold type.\n'
          'Use threshold type *_INVERSE for light-on-dark, not for'
          ' dark-on-light contrasts.\n'
          'Also, "Contour radius size" sliders may need adjusting.')
    messagebox.showinfo(detail=_m)


def wait4it(img):
    """Pop up info for larger input image files."""
    if img.shape[1] > 2000:
        msg = ('Larger images will take longer to process.\n'
               'A large number of objects will take longer to process.\n'
               'So, patience Grasshopper.\n'
               'Resist the urge to start clicking away if things seem stalled.\n'
               'If the threshold image shows up as black-on-white, then use\n'
               'the inverse option at start or in the Threshold type pull-down.')
        messagebox.showinfo(title='Wait for it...',
                            detail=msg)
