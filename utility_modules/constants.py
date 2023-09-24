"""
Constants used throughout main script and modules:
ALPHA_MAX
BETA_MAX
COLORS_CV
COLORS_TK
CONTOUR_METHOD
CONTOUR_MODE
CV_BORDER
CV_FILTER
CV_MORPH_SHAPE
CV_MORPHOP
DARK_BG
DISTANCE_TRANS_TYPE
DRAG_GRAY
FONT_TYPE
LABEL_PARAMETERS
MASTER_BG
MY_OS
NCPU
PANEL_LEFT
PANEL_RIGHT
RADIO_PARAMETERS
SCALE_PARAMETERS
SIZE_STANDARDS
SIZE_TO_WAIT
STD_CONTOUR_COLOR
STUB_ARRAY
THRESH_TYPE
WIDGET_FG
WIN_NAME
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License'

# Standard library import
import sys
# from multiprocessing import cpu_count
from psutil import cpu_count

# Third party import
import cv2
import numpy as np

MY_OS: str = sys.platform[:3]

STUB_ARRAY: np.ndarray = np.ones(shape=(5, 5), dtype='uint8')

WIN_NAME = {
    'input': 'Input image',
    'contrast': 'Adjusted contrast <- | -> Reduced noise',
    'filter': 'Filtered <- | -> Thresholded',
    'dist_trans': 'Distances transformed <- | -> Watershed segments',
    'sized': 'Size-selected objects, circled with diameters.',
}

# Set ranges for trackbars used to adjust contrast and brightness for
#  the cv2.convertScaleAbs method.
ALPHA_MAX = 400
BETA_MAX = 254  # Provides a range of [-127 -- 127].

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

CV_MORPHOP = {
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
Hex values source: https://www.rgbtohex.net/
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
    'red': 'red1',
    'green':'green1',
}

# Need tk to match system's default white shade.
if MY_OS == 'dar':  # macOS
    COLORS_TK['tk_white'] = 'white'
elif MY_OS == 'lin':  # Linux (Ubuntu)
    COLORS_TK['tk_white'] = 'grey85'
else:  # platform is 'win' Windows (10)
    COLORS_TK['tk_white'] = 'grey95'

STD_CONTOUR_COLOR = {'green': (0, 255, 0)}

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

# Need to adjust text length across platform's for the cv2.getTextSize()
# function used in utils.text_array() module.
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

if MY_OS == 'lin':
    WIDGET_FONT = 'TkTooltipFont', 8
    radio_params = dict(
        fg=COLORS_TK['yellow'],
        activebackground='gray50',  # Default is 'white'.
        activeforeground=COLORS_TK['sky blue'],  # Default is 'black'.
        selectcolor=COLORS_TK['dark blue'])
elif MY_OS == 'win':
    WIDGET_FONT = 'TkTooltipFont', 8
    radio_params = dict(fg='black')
else:  # is macOS
    WIDGET_FONT = 'TkTooltipFont', 9
    radio_params = dict(fg='black')

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
    width=8,
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
    **radio_params  # are OS-specific.
)

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
        font=('TkTooltipFont', 7), # not size 8
        takefocus=False,
        state='readonly')
else:  # is macOS
    COMBO_PARAMETERS = dict(
        font=WIDGET_FONT,
        takefocus=False,
        state='readonly')

# Grid arguments to place Label images in image windows.
PANEL_LEFT = dict(
            column=0, row=0,
            padx=5, pady=5,
            sticky='nsew')
PANEL_RIGHT = dict(
            column=1, row=0,
            padx=5, pady=5,
            sticky='nsew')

# Values are in mm.
# Value of 0.999 for 'None' is a hack to force 3 sig.fig as the default.
#   This allows the most accurate display of pixel widths at startup.
SIZE_STANDARDS = {
    'None': 0.999,
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

# The image dimension limit to issue a warning that processing may take
#   a while to complete.
SIZE_TO_WAIT = 2000

# Count only physical CPU cores (exclude hyperthreads) for best performance
#   of parallel.MultiProc().  Leave 1 CPU open.
NCPU = cpu_count(logical=False) - 1
# The multiprocess cpu_count() will count hyperthreads as CPUs.
# NCPU = mp.cpu_count()
