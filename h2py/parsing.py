"""
Estimation of H2 from sequence data.
"""

import numpy as np

from .masks import GeneticMask
from .matrices import HaplotypeMatrix, GenotypeMatrix, GenotypeProbMatrix
from . import utils
from .utils import timestamp


# =============================================================================
# Principal function for computing H2.
# =============================================================================


def compute_h2_stats(
    vcf_file=None,
    pop_file=None,
    pops=None,
    bed_file=None,
    chromosome=None,
    interval=None,
    rec_map_file=None,
    r_bins=None,
    bp_bins=None,
    min_bp=None,  #TODO
    mut_map_file=None,
    u_bar=None,
    use_genotypes=True,
    use_genotype_probs=False,
    report=True,
    compute_denoms=True,
    stats_to_compute=None,
    pairwise=True,
    ac_filter=True,
):
    """
    Compute H2 statistics on a chromosome or chromosome interval.

    Parameters
    ----------
    vcf_file : str, optional
        Path to VCF file.

    ac_filter : bool, optional
        Allele count filter. If True (default), ignore multiallelic sites.

    Returns
    -------
    """
    # Check arguments
    if use_genotypes and use_genotype_probs:
        raise ValueError("cannot use_genotypes and use_genotype_probs")

    if report:
        print(timestamp(), "Computing H2...")

    # Load sequence data
    if use_genotypes:
        if vcf_file is not None:
            matrix = GenotypeMatrix.from_vcf(
                vcf_file,
                bed_file=bed_file,
                chromosome=chromosome,
                interval=interval,
                pop_file=pop_file,
                ac_filter=ac_filter,
            )
        else:
            if genotype_matrix is not None:
                matrix = genotype_matrix
            else:
                raise ValueError("genotype_matrix or vcf_file required")
    elif use_genotype_probs:
        if vcf_file is not None:
            matrix = GenotypeProbMatrix.from_vcf(
                vcf_file,
                bed_file=bed_file,
                chromosome=chromosome,
                interval=interval,
                pop_file=pop_file,
                ac_filter=ac_filter,
            )
        else:
            if genotype_prob_matrix is not None:
                matrix = genotype_prob_matrix
                # TODO etc....
            else:
                raise ValueError("genotype_prob_matrix or vcf_file required")
    else:
        hm = 0.0
        matrix = hm

    # Load recombination map data and check bins
    if r_bins is not None:
        if rec_map_file is None:
            raise ValueError("rec_map_file is required")
        bins = r_bins
        coords = _get_map_coords(rec_map_file, matrix.positions)
    else:
        if bp_bins is None:
            raise ValueError("r_bins or bp_bins is required")
        bins = r_bins
        coords = matrix.positions

    # Load mutation map data
    if mut_map_file is not None:
        mut_map = _get_mut_map(mut_map_file, matrix.positions)
        if u_bar is None:
            u_bar = np.mean(mut_map)
            if report:
                print("  Using u_bar = {u_bar}")
        weights = mut_map / u_bar
    else:
        weights = None

    # Compute statistics
    if report:
        print(timestamp(), "Computing statistics...")
    sums_list = _compute_h2_sums(
        matrix,
        coords,
        bins,
        weights=weights,
        pops=pops,
        stats_to_compute=stats_to_compute,
        pairwise=pairwise,
        use_genotypes=use_genotypes,
        use_genotype_probs=use_genotype_probs
    )
    if report:
        print(timestamp(), "Computed statistics.")

    # Compute denominators
    if compute_denoms:
        if report:
            print(timestamp(), "Computing denominators...")
        denoms = compute_h2_denoms()
        if report:
            print(timestamp(), "Computed denominators.")
    else:
        denoms = None

    bin_tuples = _get_bin_tuples(bins)

    if report:
        print("  Done!")

    return {"pops": pops, "stats": stats_to_compute, "bins": bins,
            "sums": sums_list, "denoms": denoms}


def _compute_h2_sums(
    matrix,
    coords,
    bins,
    pops=None,
    stats_to_compute=None,
    pairwise=True,
    weights=None,
    use_genotypes=True,
    use_genotype_probs=False,
):
    """
    Call subordinate functions to compute H2 from preloaded data.
    """
    if pops is None:
        pops = matrix.pop_names
    n_pops = len(pops)

    if stats_to_compute is None:
        stats_to_compute = (_h2_names(n_pops), _h_names(n_pops))

    if pairwise:
        sums = _average_over_pairwise_h2(
            matrix,
            coords,
            bins,
            pops,
            stats_to_compute,
            weights=None,
            use_genotypes=True,
            use_genotype_probs=False,
        )
    else:
        raise ValueError("multi-diploid calculator under construction")

    sums_list = [s for s in sums]
    h_sums = _compute_heterozygosity(

    )
    sums_list.append(h_sums)

    return sums_list


def compute_h2_denoms(
    bed_file=None,
    rec_map_file=None,
    r_bins=None,
    bp_bins=None,
    interval=None,
    ):
    """
    Compute the denominator of the H2 statistic- the number of pairs of
    accessible sites, binned by the distances between them.

    The last element of the denominator array holds the denominator of the
    heterozygosity statistic, which is the number of accessible sites.

    Parameters
    ----------

    Returns
    -------
    denoms : np.ndarray, shape (n_bins)
        Binned counts of accessible site pairs.
    """
    positions = _get_bed_file_positions(bed_file, interval=interval)
    if rec_map_file is not None and r_bins is not None:
        coords = _assign_map_coordinates(positions, rec_map_file)
        bins = r_bins
    else:
        if bp_bins is not None:
            coords = positions
            bins = bp_bins
        else:
            raise ValueError("bins must be provided")
    h2_denoms = _compute_h2_denoms(coords, bins)
    n_sites = len(positions)
    denoms = np.append(h2_denoms, n_sites)
    return denoms


def _compute_h2_denoms(coords, bins, pos=None, min_bp=None, max_bp=None):
    """Compute binned denominator for two-locus statistics"""
    binned_denoms = np.zeros(len(bins) - 1, dtype=np.float64)
    # Indices (inclusive) of lowest-indexed sites in the zeroth bin
    lower_sites = np.maximum(np.searchsorted(coords, coords + bins[0]),
                             np.arange(1, len(coords) + 1))
    for ii, upper_edge in enumerate(bins[1:]):
        # Indices (exclusive) of highest-indexed sites in bin `ii`
        upper_sites = np.searchsorted(coords, coords + upper_edge)
        binned_denoms[ii] = np.sum(upper_sites - lower_sites)
        lower_sites = upper_sites
    return binned_denoms


# =============================================================================
# Functions for averaging/bootstrapping across genomic intervals
# =============================================================================


def get_means_across_regions(all_data):
    """
    Compute mean statistics across several genomic intervals.

    Parameters
    ----------
    all_data : dict
        Maps genomic interval labels to dicts following the output of TODO

    Returns
    -------
    means : list, length n_bins
    """
    labels = list(all_data.keys())
    numers = [0.0 * row for row in all_data[labels[0]]["sums"]]
    denoms = [0.0 for row in all_data[labels[0]]["denoms"]]
    for label in labels:
        for ii in range(len(numers)):
            numers[ii] += all_data[label]["sums"][ii]
            denoms[ii] += all_data[label]["denoms"][ii]
    means = [n / d for n, d in zip(numers, denoms)]
    return means


def get_bootstrap_replicates(all_data, n_replicates=None, n_samples=None):
    """
    Draw several bootstrap replicates from a list of sums computed on genomic
    intervals.

    Parameters
    ----------
    all_data : dict
        Maps genomic interval labels to dicts following the output of TODO
    n_replicates : int, optional
        Number of bootstrap replicates to conduct. If None, defaults to the
        length of `all_data`.
    n_samples : int, optional
        Number of samples per replicate. If None, defaults to the length of
        `all_data`.

    Returns
    -------
    sets : list
        List of bootstrap sample means.
    """
    if n_replicates is None:
        n_replicates = len(all_data)
    if n_samples is None:
        n_samples = len(all_data)

    labels = list(all_data.keys())
    all_means = []
    for ii in range(n_replicates):
        sample_data = dict()
        for jj in range(n_samples):
            label = np.random.choice(labels)
            sample_data[ii] = all_data[label]
        sample_means = get_means_across_regions(sample_data)
        all_means.append(sample_means)
    return all_means


def bootstrap_data(all_data):
    """
    Compute a variance/covariance matrix across H2 statistics for each bin,
    by bootstrapping across sums of the statistic precomputed on genomic
    intervals.
    """
    # Check to make sure the variance/covariance matrix can be computed

    labels = list(all_data.keys())
    means = get_means_across_regions(all_data)
    bootstrap_means = get_bootstrap_replicates(all_data)
    reshaped_means = [[m[i] for m in bootstrap_means]
                       for i in range(len(means))]
    varcovs = [np.cov(np.array(m).T) for m in reshaped_means]
    data = dict()
    data["pops"] = all_data[labels[0]]["pops"]
    data["stats"] = all_data[labels[0]]["stats"]
    data["bins"] = all_data[labels[0]]["bins"]
    data["means"] = means
    data["varcovs"] = varcovs
    return data


def subset_data(data, to_pops=None, r_min=None, r_max=None):
    """
    Subset a dictionary of statistics to given populations/bins.

    Parameters
    ----------

    Returns
    -------
    """
    means = data["means"]
    varcovs = data["varcovs"]

    if to_pops is not None:
        means = _subset_means(means, data["pops"], to_pops)
        varcovs = _subset_varcovs(varcovs, data["pops"], to_pops)
        pops = to_pops
        n_pops = len(pops)
        stats = [_h2_names(n_pops), _h_names(n_pops)]
    else:
        pops = data["pops"]
        stats = data["stats"]

    bins = []
    new_means = []
    new_varcovs = []
    for ii, b in enumerate(data["bins"]):
        if r_min is not None:
            if b[0] < r_min:
                continue
        if r_max is not None:
            if b[1] > r_max:
                continue
        bins.append(b)
        new_means.append(means[ii])
        new_varcovs.append(varcovs[ii])

    data_out = dict()
    data_out["pops"] = pops
    data_out["stats"] = stats
    data_out["bins"] = bins
    data_out["means"] = new_means
    data_out["varcovs"] = new_varcovs
    return data_out


def _subset_means(means, pops, to_pops):
    """Extract the subset of means that pertain to populations in `to_pops`"""
    stats = _h2_names(len(pops))
    to_indices = [pops.index(p) for p in to_pops]
    to_stats = []
    for ii, idx1 in enumerate(to_indices):
        for idx2 in to_indices[ii:]:
            idx1, idx2 = sorted([idx1, idx2])
            to_stats.append(f"H2_{idx1}_{idx2}")
    keep = np.array([stats.index(s) for s in to_stats])
    new_means = [m[keep] for m in means]
    return new_means


def _subset_varcovs(varcovs, pops, to_pops):
    """Extract a subsets of covariance matrices that correspond to `to_pops`"""
    stats = _h2_names(len(pops))
    to_indices = [pops.index(p) for p in to_pops]
    to_stats = []
    for ii, idx1 in enumerate(to_indices):
        for idx2 in to_indices[ii:]:
            idx1, idx2 = sorted([idx1, idx2])
            to_stats.append(f"H2_{idx1}_{idx2}")
    keep = np.array([stats.index(s) for s in to_stats])
    mesh = np.ix_(keep, keep)
    new_varcovs = [v[mesh] for v in varcovs]
    return new_varcovs


# -----------------------------------------------------------------------------
# 'Pairwise' H2 estimators operate on single diploids or pairs of diploids.
# -----------------------------------------------------------------------------


def _average_over_pairwise_h2(
    matrix,
    coords,
    bins,
    pops,
    stats_to_compute,
    weights=None,
    use_genotypes=True,
    use_genotype_probs=False,
):
    """
    

    Calls _call_pair_estimator
    """
    h2_stats_to_compute = stats_to_compute[0]
    n_bins = len(bins)
    n_pops = len(pops)
    n_stats = len(h2_stats_to_compute)
    result = np.zeros((n_bins, n_stats), dtype=np.float64)

    for stat_idx, stat in enumerate(h2_stats_to_compute):
        parts = stat.split("_")
        pop_idx = (int(x) for x in parts[1:])
        if pop_idx[0] == pop_idx[1]:
            sample_idx = matrix.sample_sets[pops[pop_idx[0]]]
            n_samples = len(sample_idx)
            numer = 0.0
            for i, idx1 in enumerate(n_samples):
                for idx2 in n_samples[i:]:
                    out = _call_pairwise_estimator(
                        matrix,
                        (idx1, idx2),
                        coords,
                        bins,
                        weights=weights,
                        use_genotypes=use_genotypes,
                        use_genotype_probs=use_genotype_probs
                    )
                    if idx1 == idx2:
                        numer += out
                    else:
                        numer += 4.0 * out
            n_haps = 2 * n_samples
            denom = n_haps * (n_haps - 1) / 2
        else:
            sample_idx1 = matrix.sample_sets[pop[pop_idx[0]]]
            sample_idx2 = matrix.sample_sets[pop[pop_idx[1]]]
            n_samples1 = len(sample_idx1)
            n_samples2 = len(sample_idx2)
            numer = np.zeros(n_stats, dtype=np.float64)
            for idx1 in sample_idx1:
                for idx2 in sample_idx2:
                    numer += _call_pairwise_estimator(
                        matrix,
                        (idx1, idx2),
                        coords,
                        bins,
                        weights=weights,
                        use_genotypes=use_genotypes,
                        use_genotype_probs=use_genotype_probs
                    )
            denom = n_samples1 * n_samples2
        result[:, stat_idx] = numer / denom
    return result


def _call_pairwise_estimator(
    matrix,
    samples,
    coords,
    bins,
    weights=None,
    use_genotypes=True,
    use_genotype_probs=False,
):
    """
    Compute H2 for a diploid or pair of diploids.

    Parameters
    ----------
    matrix : HaplotypeMatrix, GenotypeMatrix, or GPMatrix
    samples : 2-tuple/list or int
    coords : np.ndarray
        Shape (n_sites).
    bins : np.ndarray
        Shape (n_bins + 1). Defines bin edges, in the same unit as `coords`.
    weights : np.ndarray, optional
        Shape (n_sites). Specifies relative weights for each site.

    Returns
    -------
    np.ndarray of binned H2 sums.
    """
    sample1, sample2 = samples

    if sample1 == sample2:
        if use_genotypes:
            genotypes = matrix.get_diploid(sample1)
            result = _h2_geno_within_diploid(
                genotypes, coords, bins, weights=weights)
        elif use_genotype_probs:
            genotype_probs = matrix.get_diploid(sample1)
            result = _h2_gp_within_diploid(
                genotype_probs, coords, bins, weights=weights)
        else:
            haplotypes = matrix.get_diploid(sample1)
            result = _h2_hap_within_diploid(
                haplotypes, coords, bins, weights=weights)
    else:
        if use_genotypes:
            genotypes1 = matrix.get_diploid(sample1)
            genotypes2 = matrix.get_diploid(sample2)
            result = _h2_geno_between_diploid(
                genotypes1,
                genotypes2,
                coords,
                bins,
                weights=weights
            )
        elif use_genotype_probs:
            genotype_probs1 = matrix.get_diploid(sample1)
            genotype_probs2 = matrix.get_diploid(sample2)
            result = _h2_gp_within_diploid(
                genotype_probs1,
                genotype_probs2,
                coords,
                bins,
                weights=weights
            )
        else:
            haplotypes1 = matrix.get_diploid(sample1)
            haplotypes2 = matrix.get_diploid(sample2)
            result = _h2_hap_within_diploid(
                haplotypes1,
                haplotypes2,
                coords,
                bins,
                weights=weights
            )
    return result


# Pairwise estimators operate on bare numpy arrays.


def _h2_hap_within_diploid(haplotypes, coords, bins, weights=None):
    """Compute within-diploid H2 from haplotype (phased) data"""
    is_het = 1.0 * (haplotypes[:, 0] != haplotypes[:, 1])
    if weights is not None:
        is_het = is_het * weights
    return _compute_binned_sums(is_het, coords, bins)


def _h2_hap_between_diploid(
    haplotypes1,
    haplotypes2,
    coords,
    bins,
    weights=None
    ):
    """Compute between-diploid H2 from haplotype data"""
    sums = 0.0
    # Average across haplotype-by-haplotype pairs
    for hap1 in haplotypes1.T:
        for hap2 in haplotypes2.T:
            is_het = 1.0 * (hap1 != hap2)
            if weights is not None:
                is_het = weights * is_het
                sums += _compute_binned_sums(is_het, coords, bins)
    return sums / 4


def _h2_geno_within_diploid(genotypes, coords, bins, weights=None):
    """Compute within-diploid H2 from genotype (unphased) data."""
    is_het = 1.0 * (genotypes == 1)
    if weights is not None:
        is_het = weights * is_het
    return _compute_binned_sums(is_het, coords, bins)


def _h2_geno_between_diploid(
    genotypes1,
    genotypes2,
    coords,
    bins,
    weights=None
    ):
    """Compute between-diploid H2 from genotype data"""
    pi_12 = np.abs(genotypes1 - genotypes2) / 2
    if weights is not None:
        pi_12 = weights * pi_12
    return _compute_binned_sums(pi_12, coords, bins)


def _h2_gp_within_diploid(probs, coords, bins, weights=None):
    """Compute within or single-diploid H2 from genotype probabilities."""
    p_het = probs[:, 1]
    if weights is not None:
        p_het = weights * p_het
    return _compute_binned_sums(p_het, coords, bins)


def _h2_gp_between_diploid(
    probs1,
    probs2,
    coords,
    bins,
    weights=None
    ):
    """Compute between-diploid H2 from genotype probabilities."""
    p_aa_1, p_aA_1, p_AA_1 = probs1.T
    p_aa_2, p_aA_2, p_AA_2 = probs2.T
    pi_12 = (
        0.5 * p_aa_1 * p_aA_2
        + p_aa_1 * p_AA_2
        + 0.5 * p_aA_1 * p_aa_2
        + 0.5 * p_aA_1 * p_aA_2
        + 0.5 * p_aA_1 * p_AA_2
        + p_AA_1 * p_aa_2
        + 0.5 * p_AA_1 * p_aA_2
        )
    if weights is not None:
        pi_12 = weights * pi_12
    return _compute_binned_sums(pi_12, coords, bins)


def _compute_binned_sums(site_vals, coords, bins):
    """
    Engine for calculating binned sums of pairwise H2 estimators.

    Specifically, for each of the (n_sites choose 2) pairs of sites i, j that
    can be drawn from ``site_vals``, increment ``site_vals[i] * site_vals[j]``
    to the bin corresponding to the site distance, ``coords[j] - coords[i]``.

    Within and between-diploid estimators for H2 present a special case, where
    we can usually calculate H2 by taking products of some site value. For a
    single diploid, this value is ``1`` for heterozygous sites and ``0`` else.
    """
    binned_sums = np.zeros(len(bins) - 1, dtype=np.float64)
    lower_sites = np.maximum(np.searchsorted(coords, coords + bins[0]),
                             np.arange(1, len(coords) + 1))
    cumulative = np.concatenate(([0], np.cumsum(site_vals)))
    lower_cum_vals = cumulative[lower_sites]
    for ii, upper_edge in enumerate(bins[1:]):
        upper_sites = np.searchsorted(coords, coords + upper)
        upper_cum_vals = cumulative[upper_sites]
        bin_cum_vals = upper_cum_vals - lower_cum_vals
        # Take the product of left site values with cumulative right site
        # values, then sum across left sites to obtain the bin-wide sum
        binned_sums[ii] = np.sum(site_vals * bin_cum_vals)
        lower_sites = upper_sites
        lower_cum_vals = upper_cum_vals
    return binned_sums


# -----------------------------------------------------------------------------
# Slow counting functions for multi-sample estimation
# -----------------------------------------------------------------------------


def tally_haplotype_pairs(
    haplotypes,
    idx_i=None,
    idx_j=None,
    sample_indices=None
    ):
    """
    Compute two-locus haplotype counts for given locus pairs.

    Parameters
    ----------
    haplotypes : np.ndarray
        Shape (n_loci, n_haplotypes). Values 0, 1.

    idx_i, idx_j : list
        Length n_pairs. Lists specifying locus pairs. When not given,
        all locus pairs are included.

    sample_indices : list
        Indices of haplotypes to include.
    """
    if sample_indices is None:
        sample_indices = list(range(haplotypes.shape[1]))
    haplotypes = haplotypes[:, sample_indices]

    if idx_i is None and idx_j is None:
        n_loci = genotypes.shape[0]
        idx_i = [i for i in range(n_loci) for j in range(i + 1, n_loci)]
        idx_j = [j for i in range(n_loci) for j in range(i + 1, n_loci)]
    else:
        assert idx_i is not None and idx_j is not None

    locus_i = haplotypes[idx_i]
    locus_j = haplotypes[idx_j]

    n11 = np.sum((locus_i == 1) & (locus_j == 1), axis=1)
    n10 = np.sum((locus_i == 1) & (locus_j == 0), axis=1)
    n01 = np.sum((locus_i == 0) & (locus_j == 1), axis=1)
    n00 = np.sum((locus_i == 0) & (locus_j == 0), axis=1)

    counts = np.stack([n11, n10, n01, n00], axis=1)
    return counts


def tally_genotype_pairs(
    genotypes,
    idx_i=None,
    idx_j=None,
    sample_indices=None
    ):
    """
    Compute two-locus genotype counts for given locus pairs.

    Not very efficient.

    Parameters
    ----------
    genotypes : np.ndarray
        Shape (n_loci, n_samples). Values 0, 1, 2.

    idx_i, idx_j : list
        Length n_pairs. Lists specifying locus pairs. When not given,
        all locus pairs are included.

    sample_indices : list
        Indices of genotypes to include.
    """
    if sample_indices is None:
        sample_indices = list(range(genotypes.shape[1]))
    genotypes = genotypes[:, sample_indices]

    if idx_i is None and idx_j is None:
        n_loci = genotypes.shape[0]
        idx_i = [i for i in range(n_loci) for j in range(i + 1, n_loci)]
        idx_j = [j for i in range(n_loci) for j in range(i + 1, n_loci)]
    else:
        assert idx_i is not None and idx_j is not None

    locus_i = genotypes[idx_i]
    locus_j = genotypes[idx_j]

    n22 = np.sum((locus_i == 2) & (locus_j == 2), axis=1)
    n21 = np.sum((locus_i == 2) & (locus_j == 1), axis=1)
    n20 = np.sum((locus_i == 2) & (locus_j == 0), axis=1)
    n12 = np.sum((locus_i == 1) & (locus_j == 2), axis=1)
    n11 = np.sum((locus_i == 1) & (locus_j == 1), axis=1)
    n10 = np.sum((locus_i == 1) & (locus_j == 0), axis=1)
    n02 = np.sum((locus_i == 0) & (locus_j == 2), axis=1)
    n01 = np.sum((locus_i == 0) & (locus_j == 1), axis=1)
    n00 = np.sum((locus_i == 0) & (locus_j == 0), axis=1)

    counts = np.stack([n22, n21, n20, n12, n11, n10, n02, n01, n00], axis=1)
    return counts


def compute_expected_two_locus_genotypes(
    gprobs,
    idx_i=None,
    idx_j=None,
    sample_indices=None
    ):
    """
    Compute expected tallies of two-locus genotypes from genotype probabilities.

    Parameters
    ----------
    gprobs : np.ndarray
        Shape (n_loci, 3 * n_samples). For sample i, columns i, i + 1, i + 2 hold
        the posterior probabilities assigned to genotypes 0/0, 0/1, 1/1.
    """
    if sample_indices is None:
        sample_indices = list(range(gprobs.shape[1]))
    gprobs = gprobs[:, sample_indices]

    if idx_i is None and idx_j is None:
        n_loci = gprobs.shape[0]
        idx_i = [i for i in range(n_loci) for j in range(i + 1, n_loci)]
        idx_j = [j for i in range(n_loci) for j in range(i + 1, n_loci)]
    else:
        assert idx_i is not None and idx_j is not None

    locus_i = gprobs[idx_i]
    locus_j = gprobs[idx_j]

    n22 = np.sum(locus_i[:, 2::3] * locus_j[:, 2::3], axis=1)
    n21 = np.sum(locus_i[:, 2::3] * locus_j[:, 1::3], axis=1)
    n20 = np.sum(locus_i[:, 2::3] * locus_j[:, ::3], axis=1)
    n12 = np.sum(locus_i[:, 1::3] * locus_j[:, 2::3], axis=1)
    n11 = np.sum(locus_i[:, 1::3] * locus_j[:, 1::3], axis=1)
    n10 = np.sum(locus_i[:, 1::3] * locus_j[:, ::3], axis=1)
    n02 = np.sum(locus_i[:, ::3] * locus_j[:, 2::3], axis=1)
    n01 = np.sum(locus_i[:, ::3] * locus_j[:, 1::3], axis=1)
    n00 = np.sum(locus_i[:, ::3] * locus_j[:, ::3], axis=1)

    exp_counts = np.stack([n22, n21, n20, n12, n11, n10, n02, n01, n00], axis=1)
    return exp_counts


# -----------------------------------------------------------------------------
# Multi-sample estimators- these take precomputed genotype/haplotype counts
# -----------------------------------------------------------------------------


def _compute_multi_diploid_h2(
    matrix,
    pops,
    stats_to_compute,
    coords,
    bins,
    weights=None,
    use_genotypes=True,
    use_genotype_probs=False,
):
    """
    Compute H2 sums by counting two-locus genotypes/haplotypes and applying
    multi-sample estimators.
    """

    return


def _call_multi_diploid_estimator():

    return


def h2_haplotype_within(counts, pop_idx):
    """Calculate within-population H2 from two-locus haplotype counts"""
    start = 4 * pop_idx
    c1, c2, c3, c4 = counts[:, start:start + 4].T
    n = np.sum(counts[start:start + 4], axis=1)
    numer = c1 * c4 + c2 * c3
    denom = n * (n - 1) / 2
    stat = numer / denom
    return stat


def h2_haplotype_between(counts, pop1_idx, pop2_idx):
    """Calculate between-population H2 from two-locus haplotype counts"""
    start1 = pop1_idx * 4
    start2 = pop2_idx * 4
    c11, c12, c13, c14 = counts[:, start1:start1 + 4].T
    c21, c22, c23, c24 = counts[:, start2:start2 + 4].T
    n1 = np.sum(counts[:, start1:start1 + 4])
    n2 = np.sum(counts[:, start2:start2 + 4])
    numer = c11 * c24 + c21 * c14 + c12 * c23 + c22 * c13
    denom = n1 * n2
    stat = numer / denom
    return stat


def h2_genotype_within(counts, pop_idx):
    """Compute within-population H2 from genotype counts"""
    start = 9 * pop_idx
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = counts[:, start:start + 9].T
    n = np.sum(counts[:, start:start + 9], axis=1)
    numer = (
        g1 * g5
        + 2 * g1 * g6
        + 2 * g1 * g8
        + 4 * g1 * g9
        + g2 * g4
        + g2 * g5
        + g2 * g6
        + 2 * g2 * g7
        + 2 * g2 * g8
        + 2 * g2 * g9
        + 2 * g3 * g4
        + g3 * g5
        + 4 * g3 * g7
        + 2 * g3 * g8
        + g4 * g5
        + 2 * g4 * g6
        + g4 * g8
        + 2 * g4 * g9
        + g5 * (g5 + 1) / 2
        + g5 * g6
        + g5 * g7
        + g5 * g8
        + g5 * g9
        + 2 * g6 * g7
        + g6 * g8
        )
    denom = n * (2 * n - 1)
    stat = numer / denom
    return stat


def h2_genotype_between(counts, pop1_idx, pop2_idx):
    """Compute between-sample H2 from genotype counts"""
    start1 = 9 * pop1_idx
    start2 = 9 * pop2_idx
    g11, g12, g13, g14, g15, g16, g17, g18, g19 = counts[:, start1:start1 + 9].T
    g21, g22, g23, g24, g25, g26, g27, g28, g29 = counts[:, start2:start2 + 9].T
    n1 = np.sum(counts[:, start1:start1 + 9], axis=1)
    n2 = np.sum(counts[:, start2:start2 + 9], axis=1)
    numer = (
        (g11 + g12 / 2 + g14 / 2 + g15 / 4)
        * (g25 / 4 + g26 / 2 + g28 / 2 + g29)
        + (g15 / 4 + g16 / 2 + g18 / 2 + g19)
        * (g21 + g22 / 2 + g24 / 2 + g25 / 4)
        + (g12 / 2 + g13 + g15 / 4 + g16 / 2)
        * (g24 / 2 + g25 / 4 + g27 + g28 / 2)
        + (g14 / 2 + g15 / 4 + g17 + g18 / 2)
        * (g22 / 2 + g23 + g25 / 4 + g26 / 2)
        )
    denom = n1 * n2
    stat = numer / denom
    return stat


# -----------------------------------------------------------------------------
# Heterozygosity statistics
# -----------------------------------------------------------------------------


def _compute_heterozygosity():


    return


# -----------------------------------------------------------------------------
# Utilities
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


def _get_bin_tuples(bins):
    """Get a list of 2-tuples with bin edges from a vector of bin edges"""
    unfolded_bins = []
    for ii in range(len(bins) - 1):
        unfolded_bins.append((float(bins[ii]), float(bins[ii + 1])))
    return unfolded_bins


def _assign_map_coordinates(positions, rec_map_file):
    """Assign map coordinates to positions by loading a recombination map."""
    map_pos, map_coords = utils._read_rec_map_file(rec_map_file)
    # Assume that recombination map positions are 1-indexed
    return np.interp(positions - 1, map_pos, map_coords)

