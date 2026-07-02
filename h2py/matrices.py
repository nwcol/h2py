"""
Matrix classes for storing different representations of sequence data.
"""

import collections
import gzip
import numpy as np
import re

from . import utils


class HaplotypeMatrix():
    """
    Wrapper class for potentially multiallelic haplotype matrices.
    """

    def __init__(
        self,
        haplotypes,
        positions,
        samples=None,
        populations=None
        ):
        self.haplotypes = np.asarray(haplotypes, dtype=np.int8)
        self.positions = np.asarray(positions, drype=np.int64)

        assert self.haplotypes.shape[1] % 2 == 0

        if samples is None:
            samples = list(range(self.n_samples))
        self.samples = samples

        if populations is None:
            populations = {"all": sample_names}
        self.populations = populations

    @property
    def n_samples(self):
        return int(self.haplotypes.shape[1] / 2)

    @property
    def n_haplotypes(self):
        return self.haplotypes.shape[1]

    @property
    def n_sites(self):
        return self.haplotypes.shape[0]

    @property
    def n_variant_sites(self):
        return np.sum(np.unique(self.haplotypes, axis=1) > 1)

    def slice_sample(self, sample):
        """Get the bare haplotype array for a specific sample."""
        idx = self.samples.index(sample)
        return self.haplotypes[:, 2 * idx:2 * (idx + 1)]

    def slice_population(self, population):
        """Get the bare haplotype array for a specific population."""
        samples = self.populations[population]
        idxs = []
        for sample in samples:
            idx = self.samples.index(sample)
            idxs += [2 * idx, 2 * (idx + 1)]
        return self.haplotypes[:, idxs]

    @classmethod
    def from_vcf(
        vcf_file,
        bed_file=None,
        pop_file=None,
        interval=None,
        apply_filter=False,
        ):
        """
        Load haplotypes from a VCF file.
        """
        haplotypes, positions, samples, populations = utils.read_vcf_file(
            vcf_file,
            bed_file=bed_file,
            pop_file=pop_file,
            phased=True,
            interval=interval,
            apply_filter=apply_filter,
            )
        ret = cls(
            haplotypes,
            positions,
            samples=samples,
            populations=populations,
            )
        return ret



class GenotypeMatrix():
    """
    Wrapper class for biallelic genotype matrices.

    Parameters
    ----------
    genotypes : np.ndarray
        Shape (n_variants, n_diploids). Takes values 0, 1, 2, for homozygous
        reference, heterozygous, and homozygous alternate genotypes.
    """

    def __init__(
        self,
        genotypes,
        positions,
        samples=None,
        populations=None,
        ):
        self.genotypes = np.asarray(genotypes, dtype=np.int8)
        self.positions = np.asarray(positions, dtype=np.int64)

        if samples is None:
            n_samples = genotypes.shape[1]
            samples = list(range(n_samples))
        self.samples = samples

        if populations is None:
            populations = {"all": sample_names}
        self.populations = populations

    @property
    def n_sites(self):
        return self.genotypes.shape[0]

    @property
    def n_samples(self):
        return self.genotypes.shape[1]

    def slice_sample(self, sample):
        """Get the genotype vector for a given sample."""
        idx = self.samples.index(sample)
        return self.genotypes[:, idx]

    def slice_population(self, population):
        """Get the genotype array for a given population."""
        samples = self.populations[population]
        idxs = [self.samples.index(sample) for sample in samples]
        return self.genotypes[:, idxs]

    @classmethod
    def from_vcf(
        vcf_file,
        bed_file=None,
        pop_file=None,
        interval=None,
        apply_filter=False,
        ):
        """
        Load genotypes from a VCF file.
        """
        genotypes, positions, samples, populations = utils.read_vcf_file(
            vcf_file,
            bed_file=bed_file,
            pop_file=pop_file,
            phased=False,
            interval=interval,
            apply_filter=apply_filter,
            )
        ret = cls(
            genotypes,
            positions,
            samples=samples,
            populations=populations,
            )
        return ret

    @classmethod
    def from_tree_sequence():
        pass

    @classmethod
    def from_haplotype_matrix():
        pass


class GenotypeProbMatrix():
    """
    Matrix of biallelic genotype probabilities (VCF field ``GP``).

    Parameters
    ----------
    genotype_probs : np.ndarray, shape (n_sites, 3*n_samples)
        Array of genotype probabilities. Rows contain p(0/0), p(0/1), p(1/1)
        for each sample.
    positions : np.ndarray, shape (n_sites)
        0-indexed positions of sites in ``genotype_probs``.
    pop_map : dict
        Mapping from population labels (str) to lists of indices. Indices
        access diploid samples, e.g. index ``i`` corresponds to columns sliced
        by ``3*i:3*(i+1)``.
    """

    def __init__(self, genotype_probs, positions, pop_map):
        self.genotype_probs = np.asarray(genotype_probs, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.pop_map = pop_map
        # Check matrix shape against positions and pop_map
        if len(self.genotype_probs) != len(self.positions):
            raise ValueError("genotype_probs, positions site numbers disagree")
        if sum([len(self.pop_map[x]) for x in self.pop_map]) != self.n_samples:
            raise ValueError("genotype_probs, pop_map sample numbers disagree")

    def __str__(self):
        return (f"GenotypeProbMatrix ({self.n_sites} sites, "
                f"{self.n_samples} samples, {self.n_pops} pops)")

    def __repr__(self):
        return (f"GenotypeProbMatrix({self.genotype_probs}, "
                f"{self.positions}, {self.pop_map})")

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

    def slice_sample(self, sample_idx):
        """Access the bare genotype probability array for a sample."""
        return self.genotype_probs[:, sample_idx * 3:(sample_idx + 1) *3]

    def slice_pop(self, pop):
        """Access the bare genotype probability array for a population."""
        sample_idx = self.pop_map[pop_idx]
        col_idx = [i for s in sample_idx for i in range(3 * s, 3 * (s+1))]
        return self.genotype_probs[:, col_idx]

    @classmethod
    def from_vcf(
        cls,
        vcf_file,
        bed_file=None,
        interval=None,
        chromosome=None,
        pop_file=None,
        filtered=False
    ):
        """
        Load genotype probabilities from a VCF file.
        """
        genotype_probs, positions, pop_map = read_vcf_file(
            vcf_file,
            bed_file=bed_file,
            interval=interval,
            chromosome=chromosome,
            pop_file=pop_file,
            read_genotype_probs=True,
            ac_filter=True,
            filtered=filtered,
        )
        return cls(genotype_probs, positions, pop_map)


# -----------------------------------------------------------------------------
# VCF reader and subordinate utilities
# -----------------------------------------------------------------------------


def read_vcf_file(
    vcf_file,
    bed_file=None,
    interval=None,
    chromosome=None,
    pop_file=None,
    phased=False,
    read_genotype_probs=False,
    ac_filter=True,
    filtered=False,
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
    if bed_file is not None:
        mask_regions = _read_bed_file(bed_file)
        site_mask = _regions_to_mask(mask_regions)
    else:
        site_mask = None

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

            if site_mask is not None:
                if pos0 >= len(site_mask):
                    break
                if site_mask[pos0]:
                    continue

            # Check whether the site passes filters
            alleles = [elems[3]] + elems[4].split(",")
            is_sn = [len(a) == 1 for a in alleles]

            if not np.all(is_sn):
                continue

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

    positions = np.asarray(positions, dtype=np.int64)

    if read_genotype_probs:
        matrix = np.asarray(matrix, dtype=np.float64)
        matrix = _convert_phred_scores(matrix)
    else:
        matrix = np.asarray(matrix, dtype=np.int64)
        if not phased:
            if not ac_filter:
                raise ValueError("TODO post-hoc filter for biallelic sites")
            matrix = matrix[:, ::2] + matrix[:, 1::2]

    # Generate mapping between pop labels and indices of diploids in ``matrix``
    pop_map = {pop: [sample_idx.index(idx) for idx in pop_idx[pop]]
               for pop in pop_idx}

    return matrix, positions, pop_map


def _parse_haplotypes(elems, sample_idx, gt_idx):
    """Extract allele codes (as strings) from a split VCF line."""
    samples = [elems[9:][i] for i in sample_idx]
    gt_strs = [s.split(":")[gt_idx] for s in samples]
    haplotypes = [a for gt in gt_strs for a in re.split("/|\\|", gt)]
    return haplotypes


def _parse_genotype_probs(elems, sample_idx, gp_idx):
    """Extract genotype probabilities (as strings) from a split VCF line."""
    samples = [elems[9:][i] for i in sample_idx]
    gp_strs = [s.split(":")[gp_idx] for s in samples]
    genotype_probs = [gp for gps in gp_strs for gp in gps.split(",")]
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
    populations = collections.defaultdict(list)
    with open(pop_file, "r") as fin:
        for line in fin:
            sample, population = line.strip().split()
            populations[population].append(sample)
    return populations


def _convert_phred_scores(phred_arr):
    """
    Transform an array of biallelic genotype probabilities from Phred-scaled
    probabilities to regular probabilities.
    """
    assert phred_arr.shape[1] % 3 == 0
    n_samples = int(phred_arr.shape[1] / 3)
    prob_arr = np.zeros_like(phred_arr)
    for idx in range(n_samples):
        start = 3 * idx
        end = 3 * (idx + 1)
        scores = phred_arr[:, start:end]
        raw_probs = 10 ** (-scores / 10)
        prob_arr[:, start:end] = raw_probs / np.sum(raw_probs, axis=1)[:, None]
    return prob_arr

