"""
General housekeeping utilities.
Functions:
about_win: a toplevel window for the Help>About menu selection.
check_platform - Exit if not Linux, Windows, or macOS.
program_name - Get the program name for file paths and window titles.
valid_path_to - Get correct path to program's files.
set_icon - Set the program icon image file.
save_report_and_img- Save files of result image and its report.
export_settings_to_json - Write selector widget values to a JSON file.
export_object_labels - Write a CSV file for object ROI bounding boxes.
export_each_segment - Write a JPEG image file for each contoured segment.
display_report - Place a formatted text string into a specified Frame.
count_sig_fig - Count number of significant figures in a number.
quit_gui -  Error-free and informative exit from the program.
no_objects_found - A simple messagebox when a contour pointset is empty.
"""
# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License'

# Standard library imports.
import csv
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


def about_win() -> None:
    """
    Basic information about the package in scrolling text in a new
    Toplevel window.
    Generally called from a "Help->About" menu.
    Calls manage.arguments() to get the about text.

    Returns:
        None
    """
    aboutwin = tk.Toplevel(bg=const.MASTER_BG)
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
                             font=const.MENU_FONT,
                             )

    # The text returned from manage.arguments is that used for the --about arg.
    abouttext.insert(index=tk.INSERT,
                     chars=f'{manage.arguments()["about"]}')
    abouttext.grid(sticky=tk.NSEW)


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
        return Path(sys.executable).stem
    return Path(sys.modules['__main__'].__file__).stem


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


def set_icon(mainloop: tk.Tk) -> None:
    """
    Set the program icon image file.  If the icon cannot be displayed,
    print a message to the console.

    Args:  mainloop: The main tk.Tk() Toplevel running the mainloop.
    """
    # The custom app icon is expected to be in the program's images folder.
    try:
        icon = tk.PhotoImage(file=valid_path_to('images/sizeit_icon_512.png'))
        mainloop.wm_iconphoto(True, icon)
    except tk.TclError as err:
        print('Cannot display program icon, so it will be blank or the tk default.\n'
              f'tk error message: {err}')
    except FileNotFoundError as fnf:
        print(f'Cannot find program icon file: {fnf}.\n'
              'The program will run without an icon image.')


def export_settings_to_json(path2folder: str,
                            settings2save: dict,
                            called_by_cs=None) -> None:
    """
    Write selector widget values to a JSON file in the input image's
    folder. The file can be used for importing the saved settings.
    Overwrites any prior values, because the same file name is used.

    Args:
        called_by_cs: Default (None) is used by size_it.py. When arg is
         present e.g, True, expected caller is size_it_cs.py.
        path2folder: Full path name of the input file folder.
        settings2save: All selector widget values, as a dictionary.
    Returns: None
    """

    if called_by_cs:
        path2folder = Path(path2folder, const.CS_SETTINGS_FILE_NAME)
    else:  # is called by size_it.py.
        path2folder = Path(path2folder, const.SETTINGS_FILE_NAME)

    with open(path2folder, mode='wt', encoding='utf-8') as fp:
        fp.write(dumps(settings2save))


def save_report_and_img(path2folder: str,
                        img2save: Union[np.ndarray, ImageTk.PhotoImage],
                        txt2save: str,
                        caller: str,
                        ) -> None:
    """
    Write to file the current report of calculated image processing
    values. Save current result image or selected displayed image.

    Args:
        path2folder: The input image file path, as string.
        img2save: The current resulting image array; can be a np.ndarray
            from cv2 or an ImageTk.PhotoImage from tkinter/PIL
        txt2save: The current image processing report.
        caller: Descriptive name of the calling script, function or
                widget to insert in the file name, e.g. 'report',
                'contrast', etc.
    Returns: None
    """
    time_now = datetime.now().strftime(const.TIME_STAMP_FORMAT)
    time2print = datetime.now().strftime(const.TIME_PRINT_FORMAT)

    # For JPEG file format the supported parameter is cv2.IMWRITE_JPEG_QUALITY
    # with a possible value between 0 and 100, the default value being 95.
    # The higher value produces a better quality image file.
    #
    # For PNG file format the supported imwrite parameter is
    # cv2.IMWRITE_PNG_COMPRESSION with a possible value between 0 and 9,
    # the default being 3. The higher value does high compression of the
    # image resulting in a smaller file size but a longer compression time.

    img_ext = Path(path2folder).suffix
    img_name = Path(path2folder).stem
    img_folder = Path(path2folder).parent
    saved_report_name = f'{img_name}_{caller}_Report.txt'
    report_file_path = Path(img_folder, saved_report_name)
    saved_img_name = f'{img_name}_{caller}_{time_now}{img_ext}'
    saved_img_path = f'{img_folder}/{saved_img_name}'

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
    #   Path(f'{img_stem}_{caller}_settings{time_now}.txt').write_text(data2save)
    # Use this for appending multiple reports to single file:
    with open(report_file_path, mode='a', encoding='utf-8') as fp:
        fp.write(data2save)

    # Contour images are np.ndarray direct from cv2 functions, while
    #   other images are those displayed as ImageTk.PhotoImage.
    if isinstance(img2save, np.ndarray):
        cv2.imwrite(filename=saved_img_path, img=img2save)
    elif isinstance(img2save, ImageTk.PhotoImage):
        # Need to get the ImageTK image into a format that can be saved to file.
        # source: https://stackoverflow.com/questions/45440746/
        #   how-to-save-pil-imagetk-photoimage-as-jpg
        imgpil = ImageTk.getimage(img2save)

        # source: https://stackoverflow.com/questions/48248405/
        #   cannot-write-mode-rgba-as-jpeg
        if imgpil.mode in ("RGBA", "P"):
            imgpil = imgpil.convert("RGB")

        imgpil.save(saved_img_path)
    else:
        print('The specified image needs to be a np.ndarray or ImageTk.PhotoImage ')

    if manage.arguments()['terminal']:
        print(f'Result image and its report were saved to files.'
              f'{data2save}')


def export_object_labels(path2input: str,
                         object_label: list[list],
                         timestamp: str) -> None:
    """
    Writes a CSV file for object ROI bounding boxes and other data
    needed for label annotation in CNN object detection training.
    Called from ViewImage.select_and_export_objects(), from a Button()
    command.

    Args:
        path2input: The full input image file path, as a string. Will be
            used to write the CSV output file.
        object_label: The list of lists of object labels, including
            input file name, width, height, label class name, bounding
            box coordinates, and object size. This list is created in
            ViewImage.select_and_export_objects() as the object_label
            instance attribute.
        timestamp: The starting time of the calling method; used in
            file naming.
    Returns: None

    """

    # Default prefix is 'seg'; is changed with Terminal --prefix argument.
    img_name = Path(path2input).name
    img_folder = Path(path2input).parent
    csv_file_name = f'labels_{img_name}_{timestamp}.csv'
    file_path = Path(img_folder, csv_file_name)

    # Headers and data are structured after those in the GitHub repository
    #  https://github.com/harshatejas/pytorch_custom_object_detection
    # header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    # Labeling app https://www.makesense.ai/, https://github.com/SkalskiP/make-sense
    #  uses these headers for its CSV file exports.
    #  Added object size here to aid editing the CSV output, if needed.
    header = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
              'image_name', 'image_width', 'image_height', 'object_size']

    with open(file_path, 'w', encoding='utf-8', newline='') as fp:
        csv_labels = csv.writer(fp)
        csv_labels.writerow(header)
        csv_labels.writerows(object_label)


def export_each_segment(path2folder: str,
                        img2exp: Union[np.ndarray, cv2.UMat],
                        index: int,
                        timestamp: str) -> None:
    """
    Writes a JPEG image file for the *img2exp* contoured segment or ROI.
    File names include a timestamp and segment index number.
    Called from ViewImage.select_and_export_objects() from a Button() command.

    Args:
        path2folder: The input image file path, as a string.
        img2exp: Either a np.ndarray or binary UMat slice of a segmented
            (contoured) object, from the input image, to export to file.
        index: The segmented contour index number; used for file naming.
        timestamp: The starting time of the calling method; used for file naming.

    Returns: None
    """

    # Default prefix is 'seg'; is changed with Terminal --prefix argument.
    export_img_name = f'{manage.arguments()["prefix"]}_{timestamp}_{index}.jpg'
    export_file_path = str(Path(path2folder, export_img_name))

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
    #  hide cursor in Text;
    #  always show start of last line with window resize;
    #  (re-)grid in-place.
    reporttxt.delete(index1='1.0', index2=tk.END)
    reporttxt.insert(index=tk.INSERT, chars=report)
    # Indent helps center text in the Frame.
    reporttxt.tag_configure(tagName='leftmargin', lmargin1=20)
    reporttxt.tag_add('leftmargin', '1.0', tk.END)
    reporttxt.configure(state=tk.DISABLED)
    reporttxt.see(index="end-1c linestart")
    reporttxt.grid(column=0, row=0, columnspan=2, sticky=tk.NSEW)


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


def quit_gui(mainloop: tk.Tk, confirm=True) -> None:
    """Safe and informative exit from the program.

    Args:
        mainloop: The main tk.Tk() Toplevel running the mainloop that
            needs to destroy() to exit the program.
        confirm: An optional boolean. When True, evokes confirmation.
            When False, quits without confirmation. Use False
            when a confirmation answer of "No" might throw an exception.

    Returns: None
    """

    def _do_quit():
        print('... user has quit the program')
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

    if confirm:
        # Need use the parent parameter to place the message window in
        #  front of the window or widget that has focus.
        try:
            really_quit = messagebox.askyesno(
                parent=mainloop.focus_get(),
                title="Confirm Exit",
                detail='Are you sure you want to quit?')
            if really_quit:
                _do_quit()
            else:
                return

        except (tk.TclError, KeyError):
            print('No need to press the quit key twice. Simply confirm or cancel.')

    else:
        _do_quit()


def no_objects_found_msg(caller: str) -> None:
    """
    Pop-up info when segments not found or their sizes out of range.

    Args:
        caller: The calling script.
    """
    if caller == 'size_it':
        _m = ('No objects were found to size.\n'
              'Try changing the threshold type.\n'
              'Use threshold type *_INVERSE for light-on-dark contrasts.\n'
              'Also, "Circled radius size" sliders may need adjusting.')
        messagebox.showinfo(detail=_m)
    else:  # is size_it_cs
        _m = ('No objects were found to size.\n'
              'Try changing matte color.\n'
              'Also, "Circled radius size" sliders may need adjusting.')
        messagebox.showinfo(detail=_m)
