"""
Constants used throughout main script and modules:
ALPHA_MAX
BETA_MAX
CBLIND_COLOR_CV
CBLIND_COLOR_TK
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
PANEL_LEFT
PANEL_RIGHT
RADIO_PARAMETERS
SCALE_PARAMETERS
SIZE_STANDARDS
STD_CONTOUR_COLOR
STUB_ARRAY
THRESH_TYPE
WIDGET_FG
WIN_NAME
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License'

# Standard library import
import sys
import numpy as np

# Third party import
import cv2

MY_OS = sys.platform[:3]

STUB_ARRAY = np.ones((5, 5), 'uint8')

# NOTE: keys here must match corresponding keys in size_it_OLD.py
#   img_window dict.
WIN_NAME = {
    'input': 'Input image',
    'contrasted': 'Adjusted contrast <- | -> Reduced noise',
    'filtered': 'Filtered <- | -> Thresholded',
    'contoured': 'Threshold <- | -> Selected Threshold contours',
    'dist_trans': 'Distances transformed <- | -> Watershed segments',
    'threshold': 'Threshold <- | -> Distances transformed',
    'ws_contours': 'Size-selected objects, circled with diameters.',
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
CBLIND_COLOR_CV = {
    'blue': (178, 114, 0),
    'orange': (0, 159, 230),
    'sky blue': (233, 180, 86),
    'blueish green': (115, 158, 0),
    'vermilion': (0, 94, 213),
    'reddish purple': (167, 121, 204),
    'yellow': (66, 228, 240),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

CBLIND_COLOR_TK = {
    'blue': '#0072B2',
    'dark blue': 'MidnightBlue',
    'orange': '#E69F00',
    'sky blue': '#56B4E9',
    'blueish green': '#009E73',
    'vermilion': '#D55E00',
    'reddish purple': '#CC79A7',
    'yellow': '#F0E442',
    'black': 'black',
    'white': 'white',
    'tk_white': '',  # system's default, conditional on MY_OS
}

# Need tk to match system's default white shade.
if MY_OS == 'dar':  # macOS
    CBLIND_COLOR_TK['tk_white'] = 'white'
elif MY_OS == 'lin':  # Linux (Ubuntu)
    CBLIND_COLOR_TK['tk_white'] = 'grey85'
else:  # platform is 'win'  # Windows (10)
    CBLIND_COLOR_TK['tk_white'] = 'grey95'

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
        fg=CBLIND_COLOR_TK['yellow'],
        activebackground='gray50',  # Default is 'white'.
        activeforeground=CBLIND_COLOR_TK['sky blue'],  # Default is 'black'.
        selectcolor=CBLIND_COLOR_TK['dark blue'])
elif MY_OS == 'win':
    WIDGET_FONT = 'TkTooltipFont', 8
    radio_params = dict(fg='black')
else:  # is macOS
    WIDGET_FONT = 'TkTooltipFont', 9
    radio_params = dict(fg='black')

MASTER_BG = 'gray80'
DARK_BG = 'gray20'
DRAG_GRAY = 'gray65'
WIDGET_FG = CBLIND_COLOR_TK['yellow']

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
    bg=CBLIND_COLOR_TK['dark blue'],
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
        foreground=CBLIND_COLOR_TK['yellow'],
        takefocus=False,
        state='readonly')
elif MY_OS == 'win':  # is Windows
    COMBO_PARAMETERS = dict(
        font=('TkTooltipFont', 7),
        takefocus=False,
        state='readonly')
else:  # is macOS
    COMBO_PARAMETERS = dict(
        font=('TkTooltipFont', 9),
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
SIZE_STANDARDS = {
    'None': 1.0,
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
