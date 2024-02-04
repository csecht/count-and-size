"""
General housekeeping utilities.
Functions:
about_win: a toplevel window for the Help>About menu selection.
check_platform - Exit if not Linux, Windows, or macOS.
valid_path_to - Get correct path to program's files.
save_report_and_img- Save files of result image and its report.
display_report - Place a formatted text string into a specified Frame.
count_sig_fig - Count number of significant figures in a number.
quit_gui -  Error-free and informative exit from the program.
no_objects_found - A simple messagebox when a contour pointset is empty.
wait4it_msg - A messagebox explaining large images can take a long time.
"""
# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License'

# Standard library imports.
import platform
import sys
import tkinter as tk
from datetime import datetime
from json import dumps
from math import floor, log10
from pathlib import Path
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Union

# Third party imports.
import cv2
import numpy as np
from PIL import ImageTk

# Local application imports.
from utility_modules import manage, constants as const

if const.MY_OS == 'win':
    from ctypes import windll


def about_win(parent: tk.Toplevel) -> None:
    """
    Basic information about the package in scrolling text in a new
    Toplevel window. Closes when the calling *parent* closes.
    Called from SetupApp window "About" button.

    Args:
        parent: The Toplevel name that is calling.
    Returns:
        None
    """
    aboutwin = tk.Toplevel(master=parent)
    aboutwin.title('About Count & Size')
    aboutwin.minsize(width=400, height=200)
    aboutwin.focus_set()
    abouttext = ScrolledText(master=aboutwin,
                             width=62,
                             bg=const.MASTER_BG,  # light gray
                             relief='groove',
                             borderwidth=8,
                             padx=30, pady=10,
                             wrap=tk.WORD,
                             font=const.WIDGET_FONT,
                             # font=cv2.FONT_HERSHEY_PLAIN,
                             )

    # The text returned from manage.arguments is that used for the --about arg.
    abouttext.insert(index=tk.INSERT,
                     chars=f'{manage.arguments()["about"]}')
    abouttext.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


def check_platform() -> None:
    """
    Run check for various platforms to optimize displays.
    Intended to be called at startup.
    """
    if const.MY_OS not in 'win, lin, dar':
        print('Only Windows, Linux, and macOS platforms are supported.\n')
        sys.exit(0)

    # Need to account for Windows scaling in different releases.
    if const.MY_OS == 'win':
        if platform.release() < '10':
            windll.user32.SetProcessDPIAware()
        else:
            windll.shcore.SetProcessDpiAwareness(2)

    # print('To quit, use Esc or Ctrl-Q. From the Terminal, use Ctrl-C.')


def program_name() -> str:
    """
    Returns the script name or, if called from a PyInstaller stand-alone,
    the executable name. Use for setting file paths and naming windows.

    :return: Context-specific name of the main program, as string.
    """
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        _name = Path(sys.executable).stem
    else:
        _name = Path(sys.modules['__main__'].__file__).stem

    return _name


def valid_path_to(input_path: str) -> Path:
    """
    Get correct path to program's directory/file structure
    depending on whether program invocation is a Pyinstaller app or
    the command line. Works with symlinks. Works with absolute paths
    outside of program's folder.
    Allows command line invocation using any path; does not need to be
    from parent directory.

    :param input_path: Program's local dir/file name, as string.
    :return: Absolute path as pathlib Path object.
    """
    # Note that Path(Path(__file__).parent is the utility_modules folder.
    # Modified from: https://stackoverflow.com/questions/7674790/
    #    bundling-data-files-with-pyinstaller-onefile and PyInstaller manual.
    if getattr(sys, 'frozen', False):  # hasattr(sys, '_MEIPASS'):
        base_path = getattr(sys, '_MEIPASS', Path(Path(__file__).resolve()).parent)
        valid_path = Path(base_path) / input_path
    else:
        valid_path = Path(f'{input_path}').resolve()

    return valid_path


def save_report_and_img(input_path: str,
                        img2save: Union[np.ndarray, ImageTk.PhotoImage],
                        txt2save: str,
                        caller: str,
                        settings2save=None,
                        ) -> None:
    """
    Write to file the current report of calculated image processing
    values. Save current result image or selected displayed image.

    Args:
        input_path: The input image file path, as string.
        img2save: The current resulting image array; can be a np.ndarray
            from cv2 or an ImageTk.PhotoImage from tkinter/PIL
        txt2save: The current image processing report.
        caller: Descriptive name of the calling script, function or
                widget to insert in the file name, e.g. 'report',
                'contrast', etc.
        settings2save: a dictionary of settings values; optional call by
                    the 'Export settings' Button() cmd; will be written
                    to a json file to save.
    Returns: None
    """

    # Only the 'Export settings' button cmd uses the settings2save parameter.
    #  In that case, save only json file argument (a settings dictionary).
    #  Note that the json.dumps() function formats single quotes to double.
    settings_path = Path(Path(input_path).parent/const.SETTINGS_FILE_NAME)
    if settings2save:
        with open(settings_path, mode='wt', encoding='utf-8') as _fp:
            _fp.write(dumps(settings2save))
        return

    curr_time = datetime.now().strftime('%Y%m%d%I%M%S')
    time2print = datetime.now().strftime('%Y/%m/%d %I:%M:%S%p')

    # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
    # with a possible value between 0 and 100, the default value being 95.
    # The higher value produces a better quality image file.
    #
    # For PNG file format the supported imwrite parameter is
    # cv2.IMWRITE_PNG_COMPRESSION with a possible value between 0 and 9,
    # the default being 3. The higher value does high compression of the
    # image resulting in a smaller file size but a longer compression time.

    img_ext = Path(input_path).suffix
    img_stem = Path(input_path).stem
    img_folder = Path(input_path).parent
    saved_report_name = f'{img_stem}_{caller}_Report.txt'
    report_file_path = Path(f'{img_folder}/{saved_report_name}')
    saved_img_name = f'{img_stem}_{caller}_{curr_time}{img_ext}'
    image_file_path = f'{img_folder}/{saved_img_name}'

    if manage.arguments()['terminal']:
        data2save = (f'\n\nTime saved: {time2print}\n'
                     f'Saved image file: {saved_img_name}\n'
                     f'Saved report file: {saved_report_name}\n'
                     f'{txt2save}')
    else:
        data2save = (f'\n\nTime saved: {time2print}\n'
                     f'Saved image file: {saved_img_name}\n'
                     f'{txt2save}')

    # Use this Path function for saving individual report files:
    #   Path(f'{img_stem}_{caller}_settings{curr_time}.txt').write_text(data2save)
    # Use this for appending multiple reports to single file:
    with open(report_file_path, mode='a', encoding='utf-8') as _fp:
        _fp.write(data2save)

    # Contour images are np.ndarray direct from cv2 functions, while
    #   other images are those displayed as ImageTk.PhotoImage.
    if isinstance(img2save, np.ndarray):
        cv2.imwrite(filename=image_file_path, img=img2save)
    elif isinstance(img2save, ImageTk.PhotoImage):
        # Need to get the ImageTK image into a format that can be saved to file.
        # source: https://stackoverflow.com/questions/45440746/
        #   how-to-save-pil-imagetk-photoimage-as-jpg
        imgpil = ImageTk.getimage(img2save)

        # source: https://stackoverflow.com/questions/48248405/
        #   cannot-write-mode-rgba-as-jpeg
        if imgpil.mode in ("RGBA", "P"):
            imgpil = imgpil.convert("RGB")

        imgpil.save(image_file_path)
    else:
        print('The specified image needs to be a np.ndarray or ImageTk.PhotoImage ')

    if manage.arguments()['terminal']:
        print(f'Result image and its report were saved to files.'
              f'{data2save}')


def export_segments(input_path: str,
                    img2exp: np.ndarray,
                    index: int,
                    timestamp: str) -> None:
    """
    Writes an image file for an individual contoured segments from a
    list of contour. File names include a timestamp and segment index
    number.
    Called from ViewImage.select_and_export_objects() from a Button() command.

    Args:
        input_path: The input image file path, as string.
        img2exp: An np.ndarray slice of a segmented (contoured) object,
                 from the input image, to be exported to file.
        index: The segmented contour index number.
        timestamp: The starting time of the calling method.

    Returns: None
    """

    export_folder = Path(input_path).parent
    export_img_name = f'seg_{timestamp}_{index}.jpg'
    export_file_path = f'{export_folder}/{export_img_name}'

    # Contour images are np.ndarray direct from cv2 functions.
    # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
    # with a possible value between 0 and 100, the default value being 95.
    # The higher value produces a better quality image file.
    if isinstance(img2exp, np.ndarray):
        cv2.imwrite(filename=export_file_path,
                    img=img2exp,
                    params=[cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        print('The specified image needs to be a np.ndarray.')


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

    reporttxt = ScrolledText(master=frame,
                             font=const.REPORT_FONT,
                             bg=const.DARK_BG,
                             fg=const.COLORS_TK['yellow'],  # Matches slider labels.
                             width=max_line,
                             height=report.count('\n'),
                             relief='flat',
                             padx=8, pady=8,
                             wrap=tk.NONE,
                             )

    # Replace prior Text with current text;
    #   hide cursor in Text; always show last line; (re-)grid in-place.
    reporttxt.delete(index1='1.0', index2=tk.END)
    reporttxt.insert(index=tk.INSERT, chars=report)
    # Indent helps center text in the Frame.
    reporttxt.tag_configure(tagName='leftmargin', lmargin1=20)
    reporttxt.tag_add('leftmargin', '1.0', tk.END)
    reporttxt.configure(state=tk.DISABLED)
    reporttxt.see(index=tk.END)

    reporttxt.grid(column=0, row=0,
                   columnspan=2,
                   sticky=tk.EW)


def count_sig_fig(entry_number: Union[int, float, str]) -> int:
    """
    Determine the number of significant figures in a number.
    Be sure to verify that *entry_number* is a real number prior to using
    it as a parameter.
    The sigfig length value returned here can be used as the 'precision'
    parameter value in to_p.to_precision() statements.

    Args:
        entry_number: Any numerical representation, as string or digits.

    Returns: Integer count of significant figures in *entry_number*.
    """

    # See: https://en.wikipedia.org/wiki/Significant_figures#Significant_figures_rules_explained
    number_str = str(entry_number).lower()

    # Grab only numeric characters from *entry_number*
    sigfig_str: str = ''
    sigfig_str = ''.join([sigfig_str + _c for _c in number_str if _c.isnumeric()])

    # If scientific notation, remove the trailing exponent value.
    #  The exponent and exp_len statements allow any size of e power.
    #  Determine only absolute value of exponent to get its string length.
    #  The 'e0' and 'e-0' conditions account for use of a leading zero in the exponent.
    if 'e' in number_str:
        abs_exp = floor(log10(float(number_str))) - 1
        if 'e-0' in number_str or abs_exp == 0:
            exp_len = len(str(abs_exp))
        elif 'e0' in number_str:
            exp_len = len(str(abs_exp)) + 1
        elif 'e-' in number_str:
            exp_len = len(str(abs_exp)) - 1
        else:  # is a plain old positive exponent
            exp_len = len(str(abs_exp))

        sigfig_str = sigfig_str[:-exp_len]

    # Finally, remove leading zeros, which are not significant, and
    #  determine number of significant figures.
    return len(sigfig_str.lstrip('0'))


def quit_gui(mainloop: tk.Tk) -> None:
    """Safe and informative exit from the program.

    Args:
        mainloop: The main tk.Tk() window running the mainloop.

    Returns: None
    """

    really_quit = messagebox.askyesno(
        title="Confirm Exit",
        detail='Are you sure you want to quit?')

    if really_quit:
        print('\n*** User has quit the program ***')
        try:
            mainloop.update()
            mainloop.after(200)
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
          'Also, "Circled radius size" sliders may need adjusting.')
    messagebox.showinfo(detail=_m)
