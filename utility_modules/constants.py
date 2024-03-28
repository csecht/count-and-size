"""
Constants used throughout main script and modules:
ALPHA_MAX
BETA_MAX
COLORS_CV
COLORS_TK
CS_IMAGE_NAMES
CONTOUR_METHOD
CONTOUR_MODE
CS_SETTINGS_FILE_NAME
CV_BORDER
CV_FILTER
CV_MORPH_OP
CV_MORPH_SHAPE
DARK_BG
DISTANCE_TRANS_TYPE
DRAG_GRAY
FONT_TYPE
LABEL_PARAMETERS
LINE_THICKNESS_FACTOR
MASTER_BG
MATTE_COLOR_RANGE
MY_OS
PANEL_LEFT
PANEL_RIGHT
RADIO_PARAMETERS
SCALE_PARAMETERS
SETTINGS_FILE_NAME
SIZE_FACTOR
SIZE_STANDARDS
STUB_ARRAY
THRESH_TYPE
TIME_PRINT_FORMAT
TIME_STAMP_FORMAT
WIDGET_FG
WINDOW_PARAMETERS
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License'

# Standard library import
from sys import platform

# Third party import
import cv2
import numpy as np

MY_OS: str = platform[:3]

TIME_STAMP_FORMAT = '%Y%m%d%I%M%S'  # for file names, as: 20240301095308
TIME_PRINT_FORMAT = '%c'  # as: Fri Mar  1 09:53:08 2024, is locale-dependent.
# TIME_PRINT_FORMAT = '%Y-%m-%d %I:%M:%S %p'  # as: 2024-03-01 09:53:08 AM

# The stub is a white square set in a black square to make it obvious when
#  the code has a bug.
STUB_ARRAY: np.ndarray = cv2.rectangle(np.zeros(shape=(200, 200), dtype="uint8"),
                                       (50, 50), (150, 150), 255, -1)

SETTINGS_FILE_NAME = 'saved_settings.json'
CS_SETTINGS_FILE_NAME = 'saved_cs_settings.json'

# Set ranges for trackbars used to adjust contrast and brightness for
#  the cv2.convertScaleAbs method.
ALPHA_MAX = 400
BETA_MAX = 254  # Provides a range of [-127 -- 127].

# Scaling factors for contour_pointset, circles, and text; empirically determined.
#  Used in manage.py input_metrics().
SIZE_FACTOR: float = 5.5e-4
LINE_THICKNESS_FACTOR: float = 1.5e-3

# CV dict values are cv2 constants' (key) returned integers.
# Some of these dictionaries are used only to populate Combobox lists.
CV_BORDER = {
    'cv2.BORDER_REFLECT_101': 4,  # is same as cv2.BORDER_DEFAULT.
    'cv2.BORDER_REFLECT': 2,
    'cv2.BORDER_REPLICATE': 1,
    'cv2.BORDER_ISOLATED': 16,
}

THRESH_TYPE = {
    # Note: Can mimic inverse types by adjusting alpha and beta channels.
    # Note: THRESH_BINARY* is used with cv2.adaptiveThreshold, which is
    #  not implemented here.
    # 'cv2.THRESH_BINARY': 0,
    # 'cv2.THRESH_BINARY_INVERSE': 1,
    'cv2.THRESH_OTSU': 8,
    'cv2.THRESH_OTSU_INVERSE': 9,
    'cv2.THRESH_TRIANGLE': 16,
    'cv2.THRESH_TRIANGLE_INVERSE': 17,
}

CV_MORPH_OP = {
    'cv2.MORPH_OPEN': 2,
    'cv2.MORPH_CLOSE': 3,
    # 'cv2.MORPH_GRADIENT': 4,
    # 'cv2.MORPH_BLACKHAT': 6,
    'cv2.MORPH_HITMISS': 7,
}

CV_MORPH_SHAPE = {
    'cv2.MORPH_RECT': 0,  # cv2 default
    'cv2.MORPH_CROSS': 1,
    'cv2.MORPH_ELLIPSE': 2,
}

CV_FILTER = {
    'cv2.blur': 0,  # cv2 default
    'cv2.bilateralFilter': 1,
    'cv2.GaussianBlur': 2,
    'cv2.medianBlur': 3,
    # 'Convolution': None,
}

CONTOUR_MODE = {
    'cv2.RETR_EXTERNAL': 0,  # cv2 default
    'cv2.RETR_LIST': 1,
    'cv2.RETR_CCOMP': 2,
    'cv2.RETR_TREE': 3,
    'cv2.RETR_FLOODFILL': 4,
}

# from: https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
# CHAIN_APPROX_NONE
# stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
# CHAIN_APPROX_SIMPLE
# compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
# CHAIN_APPROX_TC89_L1
# applies one of the flavors of the Teh-Chin chain approximation algorithm [229]
# CHAIN_APPROX_TC89_KCOS
# applies one of the flavors of the Teh-Chin chain approximation algorithm [229]
CONTOUR_METHOD = {
    'cv2.CHAIN_APPROX_NONE': 1,
    'cv2.CHAIN_APPROX_SIMPLE': 2,
    'cv2.CHAIN_APPROX_TC89_L1': 3,
    'cv2.CHAIN_APPROX_TC89_KCOS': 4
}

DISTANCE_TRANS_TYPE = {
    'cv2.DIST_L1': 1,
    'cv2.DIST_L2': 2,
    'cv2.DIST_C': 3,
}

"""
Colorblind color pallet source:
  Wong, B. Points of view: Color blindness. Nat Methods 8, 441 (2011).
  https://doi.org/10.1038/nmeth.1618
See also: https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""
# OpenCV uses a BGR (B, G, R) color convention, instead of RGB.
COLORS_CV = {
    'blue': (178, 114, 0),
    'orange': (0, 159, 230),
    'dark blue': (112, 25, 25),
    'sky blue': (233, 180, 86),
    'blueish green': (115, 158, 0),
    'vermilion': (0, 94, 213),
    'reddish purple': (167, 121, 204),
    'yellow': (66, 228, 240),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0)
}

# Hex values source: https://www.rgbtohex.net/
COLORS_TK = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'dark blue': 'MidnightBlue',
    'sky blue': '#56B4E9',
    'blueish green': '#009E73',
    'vermilion': '#D55E00',
    'reddish purple': '#CC79A7',
    'yellow': '#F0E442',
    'black': 'black',
    'white': 'white',
    'tk_white': '',  # system's default, conditional on MY_OS
    'red': 'red1',  # not color-blind compatible
    'green': 'green1',  # not color-blind compatible
}

# Need tk to match system's default white shade.
if MY_OS == 'dar':  # macOS
    COLORS_TK['tk_white'] = 'white'
elif MY_OS == 'lin':  # Linux (Ubuntu)
    COLORS_TK['tk_white'] = 'grey85'
else:  # platform is 'win'  # Windows (10, 11?)
    COLORS_TK['tk_white'] = 'grey95'

CS_IMAGE_NAMES = ('input',
                  'redux_mask',
                  'matte_objects',
                  'sized')

# Used with size_it_cs.py, color screens (mattes).
MATTE_COLOR_RANGE = {
    'green1': (np.array([36, 25, 25]), np.array([70, 255, 255])),
    'green2': (np.array((36, 27, 27)), np.array([84, 255, 255])),
    'green3': (np.array([50, 20, 20]), np.array([100, 255, 255])),
    'green4': (np.array([52, 20, 55]), np.array([105, 255, 255])),
    'blue1': (np.array([102, 140, 100]), np.array([120, 255, 255])),
    'blue2': (np.array([80, 140, 100]), np.array([120, 255, 255])),
    'white1': (np.array([0, 0, 200]), np.array([0, 0, 255])),
    'white2': (np.array([0, 0, 200]), np.array([125, 60, 255])),
    'black1': (np.array([0, 0, 0]), np.array([255, 120, 80])),
    'black2': (np.array([0, 0, 0]), np.array([255, 120, 140])),
}

# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
# 	cv::HersheyFonts {
#   cv::FONT_HERSHEY_SIMPLEX = 0, # cv2 default
#   cv::FONT_HERSHEY_PLAIN = 1,
#   cv::FONT_HERSHEY_DUPLEX = 2,
#   cv::FONT_HERSHEY_COMPLEX = 3,
#   cv::FONT_HERSHEY_TRIPLEX = 4,
#   cv::FONT_HERSHEY_COMPLEX_SMALL = 5,
#   cv::FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
#   cv::FONT_HERSHEY_SCRIPT_COMPLEX = 7,
#   cv::FONT_ITALIC = 16
# }
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

# Fonts for various widgets. Make it os-specific instead of using
#  Tkinter's default named fonts because they can change and affect spacing.
if MY_OS == 'lin':  # Linux (Ubuntu)
    os_font = 'DejaVu Sans'
    os_mono_font = 'DejaVu Sans Mono'
elif MY_OS == 'win':  # Windows (10, 11)
    os_font = 'Segoe UI'
    os_mono_font = 'Consolas'
else:  # is 'dar', macOS
    os_font = 'SF Pro'
    os_mono_font = 'Menlo'

# Need platform-specific WIDGET_FONT size for best fit and look.
# Use platform's default mono font for REPORT_FONT.
# Need tk to match system's default white shade for COLOR_TK.
if MY_OS == 'lin':
    WIDGET_FONT = os_font, 8
    REPORT_FONT = os_mono_font, 9
    MENU_FONT = os_font, 9
    TIPS_FONT = os_font, 8
    radio_params = dict(
        fg=COLORS_TK['yellow'],
        activebackground='gray50',  # Default is 'white'.
        activeforeground=COLORS_TK['sky blue'],  # Default is 'black'.
        selectcolor=COLORS_TK['dark blue'])
    C_KEY = 'Ctrl'
    C_BIND = 'Control'

elif MY_OS == 'win':
    WIDGET_FONT = os_font, 7
    REPORT_FONT = os_mono_font, 8
    MENU_FONT = os_font, 9
    TIPS_FONT = os_font, 8
    radio_params = dict(fg='black')
    C_KEY = 'Ctrl'
    C_BIND = 'Control'

else:  # is macOS
    WIDGET_FONT = os_font, 10
    REPORT_FONT = os_mono_font, 10
    MENU_FONT = os_font, 13
    TIPS_FONT = os_font, 11
    radio_params = dict(fg='black')
    C_KEY = 'Command'
    C_BIND = 'Command'

MASTER_BG = 'gray80'
DARK_BG = 'gray20'
DRAG_GRAY = 'gray65'
WIDGET_FG = COLORS_TK['yellow']

LABEL_PARAMETERS = dict(
    font=WIDGET_FONT,
    bg=DARK_BG,
    fg=WIDGET_FG,
)

SCALE_PARAMETERS = dict(
    width=10,
    orient='horizontal',
    showvalue=False,
    sliderlength=20,
    font=WIDGET_FONT,
    bg=COLORS_TK['dark blue'],
    fg=WIDGET_FG,
    troughcolor=MASTER_BG,
)

RADIO_PARAMETERS = dict(
    font=WIDGET_FONT,
    bg='gray50',
    bd=2,
    indicatoron=False,
    **radio_params,  # are OS-specific.
)

# Color-in the main (self) window and give it a yellow border;
#  border highlightcolor changes to grey with loss of focus.
WINDOW_PARAMETERS = dict(
    bg=DARK_BG,
    # bg=COLORS_TK['sky blue'],  # for development
    highlightthickness=5,
    highlightcolor=COLORS_TK['yellow'],
    highlightbackground=DRAG_GRAY,
    padx=3, pady=3, )

# Here 'font' sets the shown value; font in the pull-down values
#   is set by option_add in ContourViewer.setup_styles()
if MY_OS == 'lin':
    COMBO_PARAMETERS = dict(
        font=WIDGET_FONT,
        foreground=COLORS_TK['yellow'],
        takefocus=False,
        state='readonly')
elif MY_OS == 'win':  # is Windows
    COMBO_PARAMETERS = dict(
        font=('TkTooltipFont', 7),  # not size 8
        takefocus=False,
        state='readonly')
else:  # is macOS
    COMBO_PARAMETERS = dict(
        font=WIDGET_FONT,
        takefocus=False,
        state='readonly')

# Grid arguments to position tk.Label images in their windows.
PANEL_LEFT = dict(
    column=0, row=0,
    padx=5, pady=5,
    sticky='w')
PANEL_RIGHT = dict(
    column=1, row=0,
    padx=5, pady=5,
    sticky='e')

# Values are in mm units.
# Value of 1.001 for 'None' is a hack to force 4 sig.fig as the default.
#  This allows the most accurate display of pixel widths at startup,
#  assuming that object sizes are limited to <10,000 pixel diameters.
SIZE_STANDARDS = {
    'None': 1.001,
    'Custom': 0,
    'Puck': 76.2,
    'Cent': 19.0,
    'Nickel': 21.2,
    'Dime': 17.9,
    'Quarter': 24.3,
    'Half Dollar': 30.6,
    'Sacagawea $': 26.5,
    'Eisenhower $': 38.1
}

# Count only physical CPU cores (exclude hyperthreads) for best performance
#   of parallel.MultiProc().
# NCPU = cpu_count(logical=False)
# The multiprocess cpu_count() will count hyperthreads as CPUs.
# NCPU = mp.cpu_count()
