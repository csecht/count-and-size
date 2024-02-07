#!/usr/bin/env python3
"""
Simple check of current Python version.
Functions:
minversion() - Exit program if not minimum required version.
maxversion() - Notify if running a newer than tested version.
"""
# Copyright (C) 2021-2023 C. Echt under GNU General Public License'

from packaging import version
from sys import version_info, exit as sys_exit


def minversion(req_version: str) -> None:
    """
    Check current Python version against minimum version required.
    Exit program if current version is less than required.

    :param req_version: The required minimum major and minor version;
        example, '3.6'.
    """
    curr_ver = f'{version_info.major}.{version_info.minor}'
    if version.parse(curr_ver) < version.parse(req_version):
        print(f'Sorry, but this program requires Python {req_version} or later.\n'
              f'Current Python version: {curr_ver}\n'
              'Python updates are available from https://docs.python.org/')
        sys_exit(0)


def maxversion(tested_version: str) -> None:
    """
    Check current Python version against maximum version required.
    Issue warning if current version is more than *req_version*.

    :param tested_version: The required maximum major and minor version;
        example, '3.9'.
    """

    curr_ver = f'{version_info.major}.{version_info.minor}'
    if version.parse(curr_ver) > version.parse(tested_version):
        print(f'NOTICE: this program has not yet been tested with'
              f' Python versions newer than {tested_version}.\n'
              f'Python version now running: {curr_ver}\n')
