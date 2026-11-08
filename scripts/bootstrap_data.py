"""
Execute block-bootstrap sampling on raw H2 computed from several genomic
regions.
"""

import argparse
import h2py
import pickle


def get_args():
    parser = argparse.ArgumentParser(
        description="Execute block-bootstrap sampling on raw H2 computed "
                    "from several genomic regions.")
    parser.add_argument("-i", "--input", required=True, nargs="*",
                        help="Input .pkl file(s), each containing a dict "
                             "that maps labels to raw H2 data for one or "
                             "more genomic regions.")
    parser.add_argument("-o", "--output", required=True, nargs="*",
                        help="Output .pkl file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()


