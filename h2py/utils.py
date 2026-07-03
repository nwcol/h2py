"""
Assorted utilities.
"""

from datetime import datetime
import numpy as np
import pandas


def _read_mut_map_file(
    mut_map_file,
):
    return


def _read_rec_map_file(
    rec_map_file,
    pos_col=None,
    map_col=None,
    sep=None,
    unit="cM",
):
    """
    Load position and map coordinate arrays from a recombination map file.

    The map is assumed to be whitespace-separated. Returns map coordinates in
    Morgans.
    """
    if pos_col is None:
        pos_col = "Position(bp)"

    if map_col is None:
        map_col = "Map(cM)"

    if sep is None:
        df = pandas.read_csv(rec_map_file, sep="\\s+")

    pos = np.array(df[pos_col], dtype=np.int64)
    coords = np.array(df[map_col], dtype=np.float64)

    if unit == "cM":
        coords *= 0.01

    return pos, coords


def _map_function(r):
    """Haldane's map function: transforms rec. fraction to Morgans."""
    if np.any(r > 0.5):
        raise ValueError("r cannot exceed 0.5")
    if np.any(r < 0):
        raise ValueError("r cannot be negative")
    return -1 / 2 * np.log(1 - 2 * r)


def _inverse_map_function(d):
    """Inverse Haldane map function: transforms Morgans to rec. fraction."""
    return (1 - np.exp(2 * -d)) / 2


def timestamp():
    """Get a string representing the date and time at the moment."""
    return f"<{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}>"

