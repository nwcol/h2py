"""
Matrix classes for representing different types of sequence data.
"""

import collections
import gzip
import numpy as np
import re

from .masks import GeneticMask
from . import simulation
from . import utils


class HaplotypeMatrix:
    """
    Matrix of haplotypes.

    Parameters
    ----------
    haplotypes : array-like, shape (n_sites, 2*n_samples)
        Array of haplotypes. These may be biallelic or multiallelic, with 0
        representing the reference allele and 1, 2, 4 representing alternate
        alleles.
    positions : array-like, shape (n_sites,)
        Array of 0-indexed haplotype site positions.
    pop_map : dict
        Mapping between population labels and indices of diploid samples.
    """

    def __init__(self, haplotypes, positions, pop_map, mask=None):
        self.haplotypes = np.asarray(haplotypes, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.pop_map = pop_map
        # Check haplotypes shape against positions, pop_map
        if self.n_sites != len(positions):
            raise ValueError("haplotypes, positions site numbers are unequal")
        if self.n_samples != sum([len(self.pop_map[p]) for p in self.pops]):
            raise ValueError("haplotypes, pop_map sample numbers are unequal")

    def __str__(self):
        return (f"HaplotypeMatrix ({self.n_sites} sites, "
                f"{self.n_samples} samples, {self.n_pops} pops)")

    def __repr__(self):
        return (f"HaplotypeMatrix({self.haplotypes}, "
                f"{self.positions}, {self.pop_map})")

    def __len__(self):
        return len(self.haplotypes)

    @property
    def shape(self):
        return self.haplotypes.shape

    @property
    def n_haplotypes(self):
        return self.haplotypes.shape[1]

    @property
    def n_samples(self):
        return int(self.n_haplotypes / 2)

    @property
    def n_sites(self):
        return self.haplotypes.shape[0]

    @property
    def pops(self):
        return [p for p in self.pop_map]

    @property
    def n_pops(self):
        return len(self.pop_map)

    def slice_haplotype(self, idx):
        """Access a single haplotype as a 1d array."""
        return self.haplotypes[:, idx]

    def slice_sample(self, idx):
        """Access the bare haplotype array for a specific sample."""
        return self.haplotypes[:, self.get_haplotype_idx(idx)]

    def slice_pop(self, pop):
        """Access the bare haplotype array for a specific population."""
        idx = [i for s in self.pop_map[pop] for i in self.get_haplotype_idx(s)]
        return self.haplotypes[:, idx]

    @staticmethod
    def get_haplotype_idx(sample_idx):
        """Get a list of indices to access the haplotypes of a sample."""
        return [2 * sample_idx, 2 * sample_idx + 1]

    @classmethod
    def from_vcf(
        cls,
        vcf_file,
        mask=None,
        bed_file=None,
        interval=None,
        chromosome=None,
        pop_file=None,
        ac_filter=True,
        filtered=False,
    ):
        """
        Load a haplotype matrix from a VCF file.
        """
        haplotypes, positions, pop_map = read_vcf_file(
            vcf_file,
            mask=mask,
            bed_file=bed_file,
            interval=interval,
            chromosome=chromosome,
            pop_file=pop_file,
            phased=True,
            ac_filter=ac_filter,
            filtered=filtered,
        )
        return cls(haplotypes, positions, pop_map)


class GenotypeMatrix():
    """
    Matrix of biallelic genotypes.

    Parameters
    ----------
    genotypes : array-like, shape (n_sites, n_samples)
        Array of biallelic genotypes. Entries 0, 1, 2 denote genotypes 0/0,
        0/1, 1/1 respectively.
    positions : array-like, shape (n_sites,)
        Array of 0-indexed site positions.
    pop_map : dict
        Mapping between population labels and lists of constituent sample
        indices.
    mask : array-like, shape (n_sites,), optional
        Boolean array
    """

    def __init__(self, genotypes, positions, pop_map, mask=None):
        self._genotypes = np.asarray(genotypes, dtype=np.float64)
        self._positions = np.asarray(positions, dtype=np.int64)
        self.pop_map = pop_map

        if mask is not None:
            self.apply_mask(mask)
        else:
            self.mask = None

        # Check genotype matrix shape against positions, pop_map
        if len(self.positions) != self.n_sites:
            raise ValueError("genotypes, positions site numbers are unequal")
        if sum([len(self.pop_map[p]) for p in self.pop_map]) != self.n_samples:
            raise ValueError("genotypes, pop_map sample numbers are unequal")

    def __str__(self):
        return (f"GenotypeMatrix ({self.n_sites} sites, "
                f"{self.n_samples} samples, {self.n_pops} pops)")

    def __repr__(self):
        return (f"GenotypeMatrix({self.genotypes}, {self.positions}, "
                f"{self.pop_map}")

    def __len__(self):
        return len(self.genotypes)

    @property
    def genotypes(self):
        """Access masked genotype matrix."""
        if self.mask is not None:
            return self._genotypes[mask]
        else:
            return self._genotypes

    @property
    def positions(self):
        """Access masked positions array."""
        if self.mask is not None:
            return self._positions[mask]
        else:
            return self._positions

    @property
    def shape(self):
        return self.genotypes.shape

    @property
    def n_samples(self):
        return self.genotypes.shape[1]

    @property
    def n_sites(self):
        return self.genotypes.shape[0]

    @property
    def pops(self):
        """Access population names."""
        return [p for p in self.pop_map]

    @property
    def n_pops(self):
        return len(self.pop_map)

    def apply_mask(self):
        pass

    def slice_sample(self, sample_idx):
        """Access the bare genotype array for a given sample index."""
        return self.genotypes[:, sample_idx]

    def slice_pop(self, pop):
        """Get the bare genotype array for a given population label."""
        return self.genotypes[:, self.pop_map[pop]]

    @classmethod
    def from_vcf(
        cls,
        vcf_file,
        mask=None,
        bed_file=None,
        interval=None,
        chromosome=None,
        pop_file=None,
        filtered=False,
    ):
        """
        Load a genotype matrix from a VCF file.
        """
        genotypes, positions, pop_map = read_vcf_file(
            vcf_file,
            mask=mask,
            bed_file=bed_file,
            interval=interval,
            chromosome=chromosome,
            pop_file=pop_file,
            filtered=filtered,
        )
        return cls(genotypes, positions, pop_map)

    @classmethod
    def from_haplotype_matrix(cls, haplotype_matrix):
        """
        Instantiate a genotype matrix by simplifying a haplotype matrix.
        """
        haplotypes = haplotype_matrix.haplotypes
        # TODO drop multiallelic sites if present!!!!
        genotypes = haplotypes[:, ::2] + haplotypes[:, 1::2]
        return cls(genotypes, haplotypes.positions, haplotypes.pop_map)


class GenotypeProbMatrix():
    """
    Matrix of biallelic genotype probabilities (VCF field ``GP``).

    Parameters
    ----------
    genotype_probs : array-like, shape (n_sites, 3 * n_samples)
        Array of genotype probabilities. Probabilities p(0/0), p(0/1), p(1/1)
        for sample ``i`` are housed in columns ``3*i, 3*i+1, 3*i+2``.
    positions : np.ndarray, shape (n_sites)
        0-indexed positions of sites in ``genotype_probs``.
    pop_map : dict
        Mapping from population labels to lists of sample indices.
    """

    def __init__(
        self,
        genotype_probs,
        positions,
        pop_map,
        mask=None
    ):
        # Check geometry
        if len(positions) != len(genotype_probs):
            raise ValueError("site number mismatch")
        if genotype_probs.shape[1] != 3 * sum([len(pop_map[x]) for x in pop_map]):
            raise ValueError("sample number mismatch")

        self._genotype_probs = np.asarray(genotype_probs, dtype=np.float64)
        self._positions = np.asarray(positions, dtype=np.int64)
        self.pop_map = pop_map
        self._mask = None
        self._non_missing = self.find_non_missing_data(genotype_probs)
        # self.apply_mask(accessible_mask) TODO

    def __str__(self):
        return (f"GenotypeProbMatrix ({self.n_sites} sites, "
                f"{self.n_samples} samples, {self.n_pops} pops)")

    def __repr__(self):
        return (f"GenotypeProbMatrix({self.genotype_probs}, "
                f"{self.positions}, {self.pop_map})")

    def __len__(self):
        """Number of sites; redundant with ``self.n_sites``."""
        return len(self.genotype_probs)

    @property
    def genotype_probs(self):
        """ """
        if self._mask is not None:
            return self._genotype_probs[self._mask]
        else:
            return self._genotype_probs

    @property
    def positions(self):
        if self._mask is not None:
            return self._positions[self._mask]
        else:
            return self._positions

    @property
    def shape(self):
        return self.genotype_probs.shape

    @property
    def n_samples(self):
        return int(self.genotype_probs.shape[1] / 3)

    @property
    def n_sites(self):
        return self.genotype_probs.shape[0]

    @property
    def pops(self):
        """Access population names from ``self.pop_map``."""
        return [p for p in self.pop_map]

    @property
    def n_pops(self):
        return len(self.pop_map)

    def apply_mask(self, accessible_mask):
        """

        """
        if accessible_mask is not None:
            self._mask = accessible_mask[self.positions]
        self._non_missing = self.find_non_missing_data(self.genotype_probs)

    def get_non_missing_mask(self, sample_idx):
        """
        Get boolean array with False where any indexed sample has missing data.
        """
        if isinstance(sample_idx, int):
            sample_idx = [sample_idx]
        return np.prod(self._non_missing[:, sample_idx], axis=1).astype(bool)

    def get_sample_positions(self, sample_idx):
        """
        Get 0-indexed positions where one or more samples have non-missing GPs.

        Parameters
        ----------
        sample_idx : int or list of int
            If more than one sample is given, return the intersection of sites
            with non-missing data across samples.
        """
        return self.positions[self.get_non_missing_mask(sample_idx)]

    def slice_sample(self, sample_idx, drop_missing_data=False):
        """
        Access the bare genotype probability array for a single sample.
        """
        idx = self.get_genotype_prob_idx(sample_idx)
        if drop_missing_data:
            mask = self.get_non_missing_mask(sample_idx)
            return self.genotype_probs[:, idx][mask]
        else:
            return self.genotype_probs[:, idx]

    def slice_samples(self, sample_idx, drop_missing_data=True):
        """Access bare genotype probability arrays for several samples."""
        idx = [self.get_genotype_prob_idx(s) for s in sample_idx]
        if drop_missing_data:
            mask = self.get_non_missing_mask(sample_idx)
            return [self.genotype_probs[:, i][mask] for i in idx]
        else:
            return [self.genotype_probs[:, idx] for i in idx]

    def slice_pop(self, pop, drop_missing_data=True):
        """Access the bare genotype probability array for a population."""
        sample_idx = self.pop_map[pop]
        idx = [i for s in sample_idx for i in self.get_genotype_prob_idx(s)]
        if drop_missing_data:
            mask = self.get_non_missing_mask(sample_idx)
            return self.genotype_probs[mask, idx]
        else:
            return self.genotype_probs[:, idx]

    @staticmethod
    def get_genotype_prob_idx(sample_idx):
        """Return the indices to genotype probabilities for a sample."""
        return [3 * sample_idx, 3 * sample_idx + 1, 3 * sample_idx + 2]

    @staticmethod
    def find_non_missing_data(data):
        """
        Get an array of sample-specific masks to exclude missing data.

        Missing genotype probability data is represented as ``[-1, -1, -1]``.

        Parameters
        ----------
        data : np.ndarray, shape (n_sites, 3 * n_samples)
            Genotype probability array.

        Returns
        -------
        np.ndarray : shape (n_sites, n_samples)
            Boolean array, with True where data is non-missing and False
            elsewhere.
        """
        n_sites, n_cols = data.shape
        n_samples = int(n_cols / 3)
        return np.sum(np.reshape(data, (n_sites, n_samples, 3)), axis=-1) > 0

    @classmethod
    def from_haplotype_matrix(cls):
        return

    @classmethod
    def from_genotype_matrix(cls):
        """
        Instantiate from a genotype matrix: place 1.0 mass on true genotypes
        """
        # TODO
        return

    @classmethod
    def from_tree_sequence(
        cls,
        ts,
        samples,
        seq_len=None,
        depth=10,
        p_err=0.01,
    ):
        """
        Use a simple simulation to generate a genotype probability matrix from
        a simulated tree sequence.

        Importantly: the "binary" mutation model must be used to simulate
        mutation with ``msprime.sim_mutations``.

        Parameters
        ----------
        # TODO write
        samples : dict
            Sample argument passed to ``msprime.sim_ancestry``, with form
            ``{"pop": sample_size,}``. Used to set up population map.
        """
        sites, _, genotype_probs = simulation.generate_genotype_probs(
            ts,
            seq_len=seq_len,
            depth=depth,
            p_err=p_err,
        )
        # Set up population mapping
        pop_map = dict()
        idx = 0
        for pop in samples:
            n_samples = samples[pop]
            pop_map[pop] = list(range(idx, idx + n_samples))
            idx += n_samples
        return cls(genotype_probs, sites, pop_map)

    @classmethod
    def from_vcf(
        cls,
        vcf_file,
        mask=None,
        bed_file=None,
        interval=None,
        chromosome=None,
        pop_file=None,
        filtered=False
    ):
        """
        Load genotype probabilities from a VCF file.
        """
        raw_probs, positions, pop_map = read_vcf_file(
            vcf_file,
            mask=mask,
            bed_file=bed_file,
            interval=interval,
            chromosome=chromosome,
            pop_file=pop_file,
            read_genotype_probs=True,
            filtered=filtered,
        )
        genotype_probs = _convert_phred_scores(raw_probs)
        return cls(genotype_probs, positions, pop_map)


# -----------------------------------------------------------------------------
# VCF reader and subordinate utilities
# -----------------------------------------------------------------------------


def read_vcf_file(
    vcf_file,
    bed_file=None,
    interval=None,
    mask=None,
    chromosome=None,
    pop_file=None,
    phased=False,
    read_genotype_probs=False,
    ac_filter=True,
    filtered=False,
    return_raw=False,
):
    """
    Read sequence data from a VCF file.

    Parameters
    ----------
    vcf_file : str, path
        Path to VCF file.
    bed_file : str, path, optional
        Path to BED file that specifies accessible intervals.
    interval : tuple, length 2, optional
        BED-style (0-indexed, half open) interval to read.
    mask : GeneticMask, optional
        Genetic mask.
    chromosome : str, optional
        If given, ignore all sites with other ``CHROM``.
    pop_file : str, path, optional
        Path to whitespace-separated population specification file. Following
        moments, this should have columns headered 'sample', 'population'.
        If not given, load every sample and assign each to a unique population.
    phased : bool, optional
        If True (default False), load haplotypes.
    read_genotype_probs : bool, optional
        If True (default False), load sample genotype probabilities ``GP`` and
        transform them from Phred scores to probabilities.
    ac_filter : bool, optional
        Allele count filter. If True (default), skip multiallelic sites.
    filtered : bool, optional
        If True (default False), skip lines where ``FILTER`` is not ``PASS``.
    return_raw : bool, optional
        If True, return the matrix as a list of lists.

    Returns
    -------
    matrix : np.ndarray, shape (n_sites, n_entries)
        Array of sequence data. For genotype data, ``n_entries`` is the number
        of samples, while for haplotypes and genotype probabilities it is twice
        or three times that number, respectively.
    positions : np.ndarray, shape (n_sites,)
        Array of 0-indexed site positions.
    pop_map : dict
        Maps population names to indices of samples in ``matrix``.
    """
    # TODO raise errors if incompatible arg combos are given

    # Load mask and population files, if given
    if mask is not None and bed_file is not None:
        raise ValueError("cannot use both `mask` and `bed_file`")

    if bed_file is not None:
        mask = GeneticMask.from_bed_file(bed_file, interval=interval)

    # Replace `GeneticMask` instance with boolean array starting at position 0
    if mask is not None:
        mask_arr = mask.complete

    if pop_file is not None:
        pop_spec = _read_pop_file(pop_file)
    else:
        pop_spec = None

    if vcf_file.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open

    # Initialize unsized output objects
    matrix = []
    positions = []

    # Indices of target entries in SAMPLE strings
    gt_idx = None
    gp_idx = None

    with open_func(vcf_file, "rb") as fin:
        for line_bytes in fin:
            line = line_bytes.decode()
            if line.startswith("#"):
                if line.startswith("#CHROM"):
                    samples = line.split()[9:]
                    if pop_spec is None:
                        pop_spec = {s: [s] for s in samples}
                    pop_idx = {p: [samples.index(s) for s in pop_spec[p]]
                               for p in pop_spec}
                    # Get indices of samples to collect
                    sample_idx = [i for x in pop_idx for i in pop_idx[x]]
                continue

            # Check whether the position is accessible
            elems = line.split()
            line_chrom = elems[0]

            if chromosome is not None:
                if line_chrom != chromosome:
                    continue
            else:
                chromosome = line_chrom

            pos1 = int(elems[1])
            pos0 = pos1 - 1

            if interval is not None:
                if pos0 < interval[0]:
                    continue
                if pos1 >= interval[1]:
                    break

            if mask is not None:
                if pos0 >= len(mask_arr):
                    break
                if mask_arr[pos0] == 0:
                    continue

            # Check whether the site passes filters
            alleles = [elems[3]] + elems[4].split(",")
            is_sn = [len(a) == 1 for a in alleles]

            if not np.all(is_sn):
                continue

            # TODO safer way to guarantee biallelic sites with GPs!
            if ac_filter:
                if len(alleles) > 2:
                    continue

            if filtered:
                filt = elems[6]
                if filtr != "PASS":
                    continue

            if read_genotype_probs:
                if gp_idx is None:
                    frmat = elems[8]
                    try:
                        gp_idx = frmat.split(":").index("GP")
                    except:
                        raise ValueError("first VCF line lacks GP")
                matrix_row = _parse_genotype_probs(elems, sample_idx, gp_idx)
            else:
                if gt_idx is None:
                    frmat = elems[8]
                    gt_idx = frmat.split(":").index("GT")
                matrix_row = _parse_haplotypes(elems, sample_idx, gt_idx)

            matrix.append(matrix_row)
            positions.append(pos0)

    # Generate mapping between pop labels and indices of diploids in ``matrix``
    pop_map = {pop: [sample_idx.index(idx) for idx in pop_idx[pop]]
        for pop in pop_idx}

    if return_raw:
        return matrix, positions

    if len(positions) == 0:
        return np.zeros((0, len(sample_idx))), np.zeros(0), pop_map

    positions = np.asarray(positions, dtype=np.int64)

    # Scan for missing data
    for ii, row in enumerate(matrix):
        for jj, elem in enumerate(row):
            if elem == ".":
                matrix[ii][jj] = "-1"

    if read_genotype_probs:
        matrix = np.asarray(matrix, dtype=np.float64)
        # matrix = _convert_phred_scores(matrix)
    else:
        matrix = np.asarray(matrix, dtype=np.int64)
        if not phased:
            if not ac_filter:
                raise ValueError("TODO post-hoc filter for biallelic sites")
            matrix = matrix[:, ::2] + matrix[:, 1::2]

    return matrix, positions, pop_map


def _parse_haplotypes(elems, sample_idx, gt_idx):
    """Extract haplotypes (as strings) from a split VCF line."""
    samples = [elems[9:][i] for i in sample_idx]
    gt_strs = [s.split(":")[gt_idx] for s in samples]
    haplotypes = [a for gt in gt_strs for a in re.split("/|\\|", gt)]
    return haplotypes


def _parse_genotype_probs(elems, sample_idx, gp_idx):
    """Extract genotype probabilities (as strings) from a split VCF line."""
    samples = [elems[9:][i] for i in sample_idx]
    # gp_strs = [s.split(":")[gp_idx] for s in samples]
    # genotype_probs = [gp for gps in gp_strs for gp in gps.split(",")]
    gp_strs = [s.split(":")[gp_idx] for s in samples]
    genotype_probs = []
    for gp in gp_strs:
        gps = gp.split(",")
        if len(gps) == 1:
            genotype_probs += ["-1", "-1", "-1"]
        else:
            genotype_probs += [p for p in gps if p != "."]
    return genotype_probs


def _read_pop_file(pop_file):
    """
    Load population specification.

    The specification file should have one whitespace-separated assignment on
    each line, e.g.,
        sample pop
        sample1 popA
        sample2 popB
        sample3 popA
    """
    samples = set()
    populations = collections.defaultdict(list)
    with open(pop_file, "r") as fin:
        header = fin.readline().strip().split()[:2]
        if header[0] != "sample" or header[1] != "pop":
            raise ValueError("pop_file format is unsupported")
        for line in fin:
            sample, population = line.strip().split()[:2]
            if sample in samples:
                raise ValueError(f"sample {sample} listed twice in pop_file")
            populations[population].append(sample)
            samples.add(sample)
    return populations


def _convert_phred_scores(phred_arr):
    """Convert an array of Phred-scaled genotype probs. to regular probs."""
    assert phred_arr.shape[1] % 3 == 0
    n_samples = int(phred_arr.shape[1] / 3)
    prob_arr = np.zeros_like(phred_arr)
    for ii in range(n_samples):
        sub_arr = phred_arr[:, 3 * ii: 3 * (ii + 1)]
        probs = np.full(sub_arr.shape, -1, dtype=np.float64)
        non_missing = np.where(np.all(sub_arr >= 0, axis=1))[0]
        ix_grid = np.ix_(non_missing, list(range(3 * ii, 3 * (ii + 1))))
        raw_probs = 10 ** (-sub_arr[non_missing] / 10)
        probs[non_missing] = raw_probs / np.sum(raw_probs, axis=1)[:, None]
        prob_arr[:, 3 * ii: 3 * (ii + 1)] = probs
    return prob_arr

