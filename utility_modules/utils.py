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

def check_platform() -> None:
    """
    Run check for various platforms to optimize displays.
    Intended to be called at startup.
    """
    if const.MY_OS == 'dar':
        print('Developed in macOS 13; earlier versions may not work.\n')

    # Need to account for scaling in Windows10 and earlier releases.
    elif const.MY_OS == 'win':

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
    # Note that Path(Path(__file__).parent is the utility_modules folder.
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

    if manage.arguments()['terminal']:
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
    reporttxt.delete(index1='1.0', index2=tk.END)
    reporttxt.insert(tk.INSERT, report)
    # Indent helps center text in the Frame.
    reporttxt.tag_configure(tagName='leftmargin', lmargin1=20)
    reporttxt.tag_add('leftmargin', '1.0', tk.END)
    reporttxt.configure(state=tk.DISABLED)

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

    # Remove non-numeric characters
    sigfig_str = ''
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
          'Also, "Contour radius size" sliders may need adjusting.')
    messagebox.showinfo(detail=_m)

def wait4it_msg(size_limit: int):
    """
    Pop up info for larger input image files.
    Args:
        size_limit: the pixel size of the input images longest dimension.
    """
    # It is probably desirable to have this size limit match that used for
    #  the img_size_for_msg local var in size_it watershed_segmentation().
    if size_limit > const.SIZE_TO_WAIT:
        msg = (f'Images larger than {const.SIZE_TO_WAIT} px take longer to process.\n'
               'A large number of found objects also take longer to process.\n'
               'So, patience Grasshopper.\n'
               '"OK" or Enter resumes image processing.\n'
               'If the threshold image shows up as black-on-white, then use'
               ' the INVERSE threshold type.')
        messagebox.showinfo(title='Wait for it...',
                            detail=msg)


def about_win(parent: tk.Toplevel) -> None:
    """
    Basic information about the package in scrolling text in a new
    Toplevel window. Closes when the calling *parent* closes.
    Called from Start window "About" button.

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
                             width=60,
                             bg=const.MASTER_BG,  # light gray
                             relief='groove',
                             borderwidth=8,
                             padx=30, pady=10,
                             wrap=tk.WORD,
                             # font=const.WIDGET_FONT,
                             font=cv2.FONT_HERSHEY_PLAIN,
                             )

    # The text returned from manage.arguments is that used for the --about arg.
    abouttext.insert(index=tk.INSERT,
                     chars=f'{manage.arguments()["about"]}')
    abouttext.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
