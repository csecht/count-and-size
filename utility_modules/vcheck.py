#!/usr/bin/env python3
"""
Simple check of current Python version.
Functions:
minversion() - Exit program if not minimum required version.
maxversion() - Warn if newer than tested versions.
"""
# Copyright (C) 2021 C. Echt under GNU General Public License'

import sys


def minversion(req_version: str) -> None:
    """
    Check current Python version against minimum version required.
    Exit program if current version is less than required.

    :param req_version: The required minimum major and minor version;
        example, '3.6'.
    """
    ver = tuple(map(int, req_version.split('.')))
    if sys.version_info < ver:
        print(f'Sorry, but this program requires Python {req_version} or later.\n'
              'Current Python version:'
              f' {sys.version_info.major}.{sys.version_info.minor}\n'
              'Python downloads are available from https://docs.python.org/')
        sys.exit(0)


def maxversion(req_version: str) -> None:
    """
    Check current Python version against maximum version required.
    Issue warning if current version is more than *req_version*.

    :param req_version: The required maximum major and minor version;
        example, '3.9'.
    """
    ver = tuple(map(int, req_version.split('.')))
    if sys.version_info > ver:
        print(f'NOTICE: this program has not yet been tested with'
              f' Python versions newer than {req_version}.\n'
              'Current Python version:'
              f' {sys.version_info.major}.{sys.version_info.minor}\n')
