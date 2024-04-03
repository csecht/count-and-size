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
# The values are used to set the corresponding cv2 constant.
# Some of these value dictionaries are used only to populate Combobox lists.
# For cv2 contour method operations, see ContourApproximationModes in:
#  https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html
CV = {
    'border': {'cv2.BORDER_REFLECT_101': 4,
               'cv2.BORDER_REFLECT': 2,
               'cv2.BORDER_REPLICATE': 1,
               'cv2.BORDER_ISOLATED': 16},
    'threshold_type': {'cv2.THRESH_OTSU': 8,
                       'cv2.THRESH_OTSU_INVERSE': 9,
                       'cv2.THRESH_TRIANGLE': 16,
                       'cv2.THRESH_TRIANGLE_INVERSE': 17},
    'morph_op': {'cv2.MORPH_OPEN': 2,
                 'cv2.MORPH_CLOSE': 3,
                 'cv2.MORPH_HITMISS': 7},
    'morph_shape': {'cv2.MORPH_RECT': 0,
                    'cv2.MORPH_CROSS': 1,
                    'cv2.MORPH_ELLIPSE': 2},
    'filter': {'cv2.blur': 0,
               'cv2.bilateralFilter': 1,
               'cv2.GaussianBlur': 2,
               'cv2.medianBlur': 3},
    'contour_mode': {'cv2.RETR_EXTERNAL': 0,
                     'cv2.RETR_LIST': 1,
                     'cv2.RETR_CCOMP': 2,
                     'cv2.RETR_TREE': 3,
                     'cv2.RETR_FLOODFILL': 4},
    'contour_method': {'cv2.CHAIN_APPROX_NONE': 1,
                       'cv2.CHAIN_APPROX_SIMPLE': 2,
                       'cv2.CHAIN_APPROX_TC89_L1': 3,
                       'cv2.CHAIN_APPROX_TC89_KCOS': 4},
    'distance_trans_type': {'cv2.DIST_L1': 1,
                            'cv2.DIST_L2': 2,
                            'cv2.DIST_C': 3},
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

# Set 'tk_white' based on the operating system's default white.
# This structure directly maps the operating system to the corresponding
#  color for 'tk_white'. This eliminates the need for a if-elif-else
#  structure. The get method for 'tk_white' is used to provide a default
#  value of 'grey95' (Windows) if the operating system is not 'dar' (macOS)
#  or 'lin' (Linux).
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
    'tk_white': {'dar': 'white', 'lin': 'grey85'}.get(MY_OS, 'grey95'),
    'red': 'red1',  # not color-blind compatible
    'green': 'green1',  # not color-blind compatible
}

CS_IMAGE_NAMES = ('input', 'redux_mask', 'matte_objects', 'sized')

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

OS_SETTINGS = {
    'lin': {
        'os_font': 'DejaVu Sans',
        'os_mono_font': 'DejaVu Sans Mono',
        'widget_font_size': (8,),
        'report_font_size': (9,),
        'menu_font_size': (9,),
        'tips_font_size': (8,),
        'radio_params': {
            'fg': COLORS_TK['yellow'],
            'activebackground': 'gray50',
            'activeforeground': COLORS_TK['sky blue'],
            'selectcolor': COLORS_TK['dark blue']
        },
        'c_key': 'Ctrl',
        'c_bind': 'Control'
    },
    'win': {
        'os_font': 'Segoe UI',
        'os_mono_font': 'Consolas',
        'widget_font_size': (7,),
        'report_font_size': (8,),
        'menu_font_size': (9,),
        'tips_font_size': (8,),
        'radio_params': {'fg': 'black'},
        'c_key': 'Ctrl',
        'c_bind': 'Control'
    },
    'dar': {
        'os_font': 'SF Pro',
        'os_mono_font': 'Menlo',
        'widget_font_size': (10,),
        'report_font_size': (10,),
        'menu_font_size': (13,),
        'tips_font_size': (11,),
        'radio_params': {'fg': 'black'},
        'c_key': 'Command',
        'c_bind': 'Command'
    }
}

# Defaults to Windows if OS is not 'lin' or 'dar'.
settings = OS_SETTINGS.get(MY_OS, OS_SETTINGS['win'])

C_KEY = settings['c_key']
C_BIND = settings['c_bind']

REPORT_FONT = settings['os_mono_font'], *settings['report_font_size']
WIDGET_FONT = settings['os_font'], *settings['widget_font_size']
MENU_FONT = settings['os_font'], *settings['menu_font_size']
TIPS_FONT = settings['os_font'], *settings['tips_font_size']

MASTER_BG = COLORS_TK['white']  # or COLORS_TK['tk_white'] for off-white.
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
    **settings['radio_params'],  # are OS-specific
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

# OS-specific combo box parameters.
# The get() method provides a default value for an unknown or untested
#  operating system.
COMBO_PARAMETERS = {
    'lin': {
        'font': WIDGET_FONT,
        'foreground': COLORS_TK['yellow'],
        'takefocus': False,
        'state': 'readonly'
    },
    'win': {
        'font': ('TkTooltipFont', 7),
        'takefocus': False,
        'state': 'readonly'
    },
    'dar': {
        'font': WIDGET_FONT,
        'takefocus': False,
        'state': 'readonly'
    }
}.get(MY_OS, {'font': 'Arial', 'takefocus': False, 'state': 'readonly'})

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
