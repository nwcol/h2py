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


# -----------------------------------------------------------------------------
# Fasta files
# -----------------------------------------------------------------------------


def read_fasta_file(fasta_file):
    """
    Read a .fasta file and return sequence(s) and header label(s).

    Sequence headers should be preceded by "\n>".

    Returns
    -------
    sequences : list of str
        Sequences contained in the input file; a list is returned even if only
        one sequence was read.
    labels : list of str
        Header labels, stripped of the ">" character.
    """
    if fasta_file.endswith("gz"):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(fasta_file, "r") as fin:
        contents = fin.read()
    if fasta_file.endswith("gz"):
        contents = contents.decode()
    raw_sequences = contents.split(">")[1:]
    sequences = []
    labels = []
    for raw_seq in raw_sequences:
        split_seq = raw_seq.split("\n")
        label = split_seq[0]
        sequence = "".join(split_seq[1:])
        labels.append(label)
        sequences.append(sequence)
    return sequences, labels


def write_fasta_file(path, sequences, labels, line_width=80):
    """
    Write a fasta or multi-fasta file containing given sequence(s), label(s).

    Parameters
    ----------
    path : str
    sequences : list of str
    labels : list of str
        Labels should omit the ">" header character, which is appended here.
    line_width : int
        Maximum line width for output file (default `80`).
    """
    assert len(sequences) == len(labels)

    with open(path, "w") as fout:
        for sequence, label in zip(sequences, labels):
            label = ">" + label + "\n"
            fout.write(label)
            n_lines = len(sequence) // line_width + 1
            for i in range(n_lines):
                start = line_width * i
                end = line_width * (i + 1)
                line = sequence[start:end] + "\n"
                fout.write(line)
    return


# -----------------------------------------------------------------------------
# Misc.
# -----------------------------------------------------------------------------


def _h_names(n_pops):
    """Get a list of names for heterozygosity statistics"""
    names = []
    for ii in range(n_pops):
        for jj in range(ii, n_pops):
            names.append(f"H_{ii}_{jj}")
    return names


def _h2_names(n_pops):
    """Get a list of names for H2 statistics"""
    names = []
    for ii in range(n_pops):
        for jj in range(ii, n_pops):
            names.append(f"H2_{ii}_{jj}")
    return names


def timestamp():
    """Get a string representing the date and time at the moment."""
    return f"<{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}>"


def _phred_function(p):
    """Convert a probability to a Phred score."""
    return -10 * np.log10(p)


def _inverse_phred_function(score):
    """Convert a Phred score to a probability."""
    return 10 ** (-score / 10)

