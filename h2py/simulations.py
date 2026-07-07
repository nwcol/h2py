"""
Simulate read sequencing to generate synthetic genotype probabilities.
"""

import numpy as np
import random
import tskit


from . import utils
from .utils import timestamp


# -----------------------------------------------------------------------------
# Simulate genotype probabilities directly from ts using a simple model
# -----------------------------------------------------------------------------


def generate_genotype_probs(
    ts,
    ref_seq=None,
    depth=5,
    p_err=0.01,
    filtered=True,
):
    """
    Generate genotype probabilities for a tree sequence using a simple model.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Mutations should be simulated with the "binary" model.
    ref_seq : str or np.ndarray, optional
        Should be composed entirely of {0, 1}.
    depth : scalar, optional
        Target mean coverage depth.
    p_err : float, optional
        Fixed sequencing error probability.
    filtered : bool, optional
        If True, remove sites where any sample lacks coverage from output.

    Returns
    -------
    sites : np.ndarray, shape (n_sites,)
    genotype_probs : np.ndarray, shape (n_sites, 3 * n_samples)
    """
    # Extract sequences from the tree sequence/
    seq_strs = ts.as_fasta(wrap_width=0).split("\n")[1::2]
    n_samples = int(len(seq_strs) / 2)
    seq_len = len(seq_strs[0])

    if ref_seq is None:
        ref_seq = np.random.randint(0, 2, size=seq_len)
    else:
        if isinstance(ref_seq, str):
            ref_seq = np.array([b for b in ref_seq], dtype=np.int64)

    sample_seqs = []
    for seq in seq_strs:
        seq = np.array([b for b in seq.replace("N", "0")], dtype=np.int64)
        # Orient to the reference sequence.
        sample_seqs.append(1 * (seq == ref_seq))
    haplotypes = np.stack(sample_seqs, axis=1)

    genotype_probs = np.zeros((seq_len, 3 * n_samples), dtype=np.float64)
    if filtered:
        sample_depths = np.zeros((seq_len, n_samples))

    for ii in range(n_samples):
        sample = haplotypes[:, 2*ii:2*(ii+1)]

        # 'cheat' by calculating priors with true data
            p_0 = np.sum(sample) / (2 * seq_len)
            p_1 = 1 - p_0
            p_het = np.sum(sample[:, 0] != sample[:, 1]) / (2 * seq_len)
            priors = np.array([p_0 - p_het / 2, p_het, p_1 - p_het / 2])

        # a different way to sample coverage depth
        k = 50 # Poisson parameter; I will later switch to gamma distr
        n_reads = int(seq_len * depth / k)
        site_depth = np.zeros(seq_len, dtype=np.int64)
        read_lens = np.random.poisson(k, size=n_reads)
        read_starts = np.random.randint(0, seq_len - read_lens, size=n_reads)
        read_ends = read_starts + read_lens
        for s, e in zip(read_starts, read_ends):
            site_depth[s:e] += 1

        # Sample coverage depth
        depths = np.random.poisson(depth, size=seq_len)
        # Draw allele samples
        f_alt = np.sum(sample, axis=1) / 2
        p_alt = f_alt + (1 - 2 * f_alt) * p_err
        n_alt = np.random.binomial(depths, p_alt)
        n_ref = depths - n_alt
        # Calculate genotype likelihoods
        genotype_liks = _get_genotype_likelihoods(n_ref, n_alt, p_err)
        # Weight genotype likelihoods by the priors
        raw_gps = genotype_liks * priors
        norm = np.sum(raw_gps, axis=1)
        genotype_probs[:, 3*ii:3*(ii+1)] = raw_gps / norm[:, None]
        if filtered:
            sample_depths[:, ii] = depths

    # Get 0-indexed positions of sites
    if filtered:
        has_coverage = sample_depths > 0
        mask = np.sum(has_coverage, axis=1) == n_samples
        sites = np.where(mask)[0]
    else:
        sites = np.arange(seq_len)

    return sites, genotype_probs


def _get_genotype_likelihoods(n_ref, n_alt, p_err):
    """
    """
    gts = (0, 1, 2)
    ref_lik = np.array([(g * p_err + (2 - g) * (1 - p_err)) / 2 for g in gts])
    alt_lik = np.array([(g * (1 - p_err) + (2 - g) * p_err) / 2 for g in gts])
    return ref_lik ** n_ref[:, None] * alt_lik ** n_alt[:, None]


def _base_likelihood(gt, b, p):
    """
    Compute the likelihood of one or more read bases given a genotype.

    Parameters
    ----------
    gt : int
        Genotype code, in {0, 1, 2}.
    b : int or np.ndarray
        Read base code(s), in {0, 1}.
    p : float or np.ndarray
        Sequencing error prob(s) for each base.
    """
    return ((1 - b) * gt * p + (1 - b) * (2 - gt) * (1 - p)
          + b * gt * (1 - p) + b * (2 - gt) * p) / 2


def _genotype_likelihood(bs, ps):
    """
    Compute biallelic genotype likelihoods across read bases.
    """
    return np.array([np.prod(base_likelihood(gt, bs, ps)) for gt in (0, 1, 2)])


def _norm_genotype_likelihood(bs, ps):
    """
    Compute normalized, Phred-scaled biallelic genotype probabilities.
    """
    gls = utils._phred_function(_genotype_likelihood(gts, bs, ps))
    return gls - np.min(gls)

