"""
A class for representing site accessibility on one chromosome.
"""

class GeneticMask:
    """
    Wrapper for a boolean numpy array representing site accessibility.

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

    def __len__(self):
        return len(self.mask)

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
        return

    def get_positions(self, index=0):
        """Get ``index``-indexed positions of accessible sites."""
        return np.where(self)[0] + self.offset + index

    def to_intervals(self):
        """Get BED-style intervals."""
        extended_mask = np.concatenate([[False], self.mask, [False]])
        jumps = np.diff(np.asarray(extended_mask, dtype=np.int8))
        starts = np.where(jumps == 1)[0]
        ends = np.where(jumps == -1)[0]
        intervals = np.stack([starts, ends], axis=1)
        intervals += self.offset
        return intervals

    @classmethod
    def from_intervals(cls, intervals, offset=0, chrom_end=None):

        return


def parse_bed_file():

    return


def read_bed_file():
    return


def write_bed_file():

    return


