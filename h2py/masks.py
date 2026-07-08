"""
Genetic mask class for tracking site accessibility.
"""

import gzip
import numpy as np


class GeneticMask:
    """
    Wrapper for a numpy boolean array representing site accessibility.

    Parameters
    ----------
    mask : np.ndarray
        Boolean array. True indicates accessible sites.
    offset : int, optional
        0-indexed offset from chromosome position 0.
    """

    def __init__(self, mask, offset=0):
        self.mask = np.asarray(mask, dtype=bool)
        self.offset = offset
        # Latent attributes; initialized by property calls
        self._cumsum = None
        self._complement = None

    def __str__(self):
        return (f"GeneticMask (offset {self.offset}, length {len(self)}, "
                f"{self.n_accessible_sites} accessible sites)")

    def __repr__(self):
        return f"GeneticMask({self.mask}, offset={self.offset})"

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        return self.mask[idx]

    def __slice__(self, s):
        return self.mask[s]

    @property
    def cumsum(self):
        """Cumulative sum."""
        if self._cumsum is None:
            self._cumsum = np.cumsum(self.mask)
        return self._cumsum

    @property
    def complement(self):
        """Get a boolean array with `False` at accessible positions."""
        if self._complement is None:
            self._complement = np.logical_not(self.mask)
        return self._complement

    @property
    def complete(self):
        """Get a version of the mask which starts at 0 (offset=0)"""
        implicit = np.zeros(self.offset, dtype=bool)
        return np.concatenate([implicit, self.mask])

    @property
    def n_accessible_sites(self):
        return self.cumsum[-1]

    @classmethod
    def from_intervals(
        cls,
        intervals,
        offset=0,
        chrom_end=None,
        interval=None
    ):
        """
        Initialize from an array of BED-style (0-indexed, half-open) intervals.
        """
        if chrom_end is not None and interval is not None:
            raise ValueError("pass either chrom_end or interval")

        if interval is not None:
            offset, chrom_end = interval
        else:
            if chrom_end is None:
                chrom_end = intervals[-1, 1]

        length = chrom_end - offset
        if length < 1:
            raise ValueError("arguments produce negative mask length")
        mask = np.zeros(length, dtype=bool)

        for (start, end) in intervals:
            adj_start = start - offset
            adj_end = end - offset
            if adj_start < 0:
                if adj_end < 0:
                    continue
                else:
                    adj_start = 0
            if adj_end > chrom_end:
                if adj_start >= chrom_end:
                    continue
                else:
                    adj_end = chrom_end
            mask[adj_start:adj_end] = True

        return cls(mask, offset=offset)

    @classmethod
    def from_bed_file(
        cls,
        bed_file,
        offset=0,
        chrom_end=None,
        interval=None,
        chromosome=None,
    ):
        """ """
        intervals = _read_bed_file(bed_file, chromosome=chromosome)
        return cls.from_intervals(intervals, offset=offset,
                                  chrom_end=chrom_end, interval=interval)

    def to_positions(self, index=0):
        """Get ``index``-indexed positions of accessible sites."""
        return np.where(self.mask)[0] + self.offset + index

    def to_intervals(self):
        """Get BED-style intervals"""
        extended_mask = np.concatenate([[False], self.mask, [False]])
        diffs = np.diff(np.asarray(extended_mask, dtype=np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        intervals = np.stack([starts, ends], axis=1)
        intervals += self.offset
        return intervals

    def to_bed_file(self, path, chromosome):
        """Write a BED file from this mask"""
        chromosome = str(chromosome)
        intervals = self.to_intervals()
        with open(path, "w") as fout:
            for (start, end) in intervals:
                line = "\t".join([chromosome, str(start), str(end)]) + "\n"
                fout.write(line)
        return


def parse_bed_file():
    """
    Instantiate a GeneticMask instance.
    """
    return


def _read_bed_file(bed_file, chromosome=None):
    """
    Read the intervals of a BED file and return them as an array.
    """
    if bed_file.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open

    intervals = []

    with open_func(bed_file, "rb") as fin:
        for lineb in fin:
            line = lineb.decode()
            elems = line.split()
            # Skip the header, if present
            if not elems[1].isnumeric() or not elems[2].isnumeric():
                continue
            line_chrom = elems[0]
            if chromosome is None:
                chromosome = line_chrom
            else:
                if line_chrom != chromosome:
                    continue
            start, end = int(elems[1]), int(elems[2])
            intervals.append([start, end])

    return np.asarray(intervals, dtype=np.int64)

