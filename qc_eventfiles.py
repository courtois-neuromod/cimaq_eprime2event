#!/usr/bin/env python
# encoding: utf-8

import os
import sys

import argparse
import glob
from numpy import nan as NaN
import pandas as pd


def get_arguments():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Perform quality check on event files outputed by
        cimaq_convert_eprime_to_bids.py script and
        outputs report (.txt file)
        """)

    parser.add_argument(
        "-d", "--idir",
        required=True, nargs="+",
        help="Directory that contains event files (.tsv)")

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Output directory - if doesn\'t exist it will be created.")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args


def main():
    '''
    This is where the magic happens!!
    '''
    pass


if __name__ == '__main__':
    sys.exit(main())
