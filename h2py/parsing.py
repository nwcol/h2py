"""
Estimation of H2 from sequence data.

Usage
-----
import h2py
intervals = [[1_000_000*i, 1_000_000*(i+1) for i in range(100)]]
sums = {i: h2py.parsing.compute_h2_stats(
              vcf_file="example.vcf.gz",
              pop_file="pops.txt",
              bed_file="example.bed",
              interval=interval[i]
           )
        for i in range(len(intervals))}
means = h2py.parsing.get_means_across_regions(sums)
means_varcovs = h2py.parsing.bootstrap_data(sums)
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
    filtered=False,
):
    """
    Compute H2 statistics on a chromosome or chromosome interval.

    Parameters
    ----------
    vcf_file : str, path
        Path to VCF file.
    pop_file : str, path, optional
        Path population specification file: a whitespace-separated file with
        two columns, headered 'sample' and 'pop', mapping VCF samples to
        population labels.
    pops : list, optional
        Populations to consider. If None (default), all populations specified
        in ``pop_file`` are used (in the order they are loaded).
    bed_file : str, path, optional
        BED file specifying accessible sites; needed to ``compute_denoms``.
    chromosome : str, optional
        Chromosome to parse (if VCF/BED files record several chromosomes).
    interval : tuple, length 2, optional
        BED-style (0-indexed, half-open) genomic interval to parse.
    rec_map_file : str, path, optional
        Should be whitespace-separated with columns 'Position(bp)', 'Map(cM)'.
    r_bins : array-like, optional
    bp_bins : array-like, optional
    min_bp : int, optional
        Minimum distance (inclusive) between sites.
    mut_map_file : str, path, optional
    u_bar : float, optional
        #TODO
    use_genotypes : bool, optional
        If True (default), treat VCF as unphased and compute statistics from
        genotypes. If False, treat VCF as phased and use haplotypes.
    use_genotype_probs : bool, optional
        If True (default False), compute stats from genotype probabilities.
    report : bool, optional
        If True (default), print verbose status messages.
    compute_denoms : bool, optional
        If True (default), calculate H2 and H denominators.
    stats_to_compute : tuple, length 2, optional
        Holds lists of H2 and H statistics to calculate, 'H2_{i}_{j}' and
        'H_{i}_{j}`. ``i`` and ``j`` index ``pops``. If None (default),
        compute all statistics for ``pops``.
    pairwise : bool, optional
        #TODO implement multi-diploid estimation
    ac_filter : bool, optional
        Allele count filter. If True (default), ignore multiallelic sites.
        #TODO should raise error if True with genotype/GP matrix.
    filtered : bool, optional
        If True (default False), skip VCF rows without ``PASS`` in ``FILTER``.

    Returns
    -------
    dict
        A dictionary with keys 'bins', 'pops', 'stats', 'sums', 'denoms'.
    """
    # Check arguments
    if use_genotypes and use_genotype_probs:
        raise ValueError("cannot use_genotypes and use_genotype_probs")

    if report:
        print(timestamp(), "Preparing data...")

    # Load sequence data
    if use_genotypes:
        if vcf_file is not None:
            matrix = GenotypeMatrix.from_vcf(
                vcf_file,
                bed_file=bed_file,
                interval=interval,
                chromosome=chromosome,
                pop_file=pop_file,
                filtered=filtered,
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
                interval=interval,
                chromosome=chromosome,
                pop_file=pop_file,
                filtered=filtered,
            )
        else:
            if genotype_prob_matrix is not None:
                matrix = genotype_prob_matrix
            else:
                raise ValueError("genotype_prob_matrix or vcf_file required")
    else:
        if vcf_file is not None:
            matrix = HaplotypeMatrix.from_vcf(
                vcf_file,
                bed_file=bed_file,
                interval=interval,
                chromosome=chromosome,
                pop_file=pop_file,
                ac_filter=ac_filter,
                filtered=filtered,
            )
        else:
            if haplotype_matrix is not None:
                matrix = haplotype_matrix
            else:
                raise ValueError("haplotype_matrix or vcf_file required")

    # Load recombination map data and check bins
    if r_bins is not None:
        if rec_map_file is None:
            raise ValueError("rec_map_file is required")
        # Transform bin units to allow direct comparison to a genetic map
        bins = utils._map_function(np.array(r_bins))
        coords = _get_map_coords(rec_map_file, matrix.positions)
    else:
        if bp_bins is None:
            raise ValueError("r_bins or bp_bins is required")
        bins = np.array(bp_bins)
        coords = matrix.positions

    # Load mutation map data
    if mut_map_file is not None:
        mut_map = _get_mut_rates(mut_map_file, matrix.positions)
        if u_bar is None:
            u_bar = np.mean(mut_map)
            if report:
                print("  Using u_bar = {u_bar:.4}")
        weights = mut_map / u_bar
    else:
        weights = None

    # Compute statistics
    if pops is None:
        pops = matrix.pops

    if stats_to_compute is None:
        n_pops = len(pops)
        stats_to_compute = (_h2_names(n_pops), _h_names(n_pops))

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
        denoms = compute_h2_denoms(
            bed_file=bed_file,
            rec_map_file=rec_map_file,
            r_bins=r_bins,
            bp_bins=bp_bins,
            interval=interval
        )
        if report:
            print(timestamp(), "Computed denominators.")
    else:
        denoms = None

    bin_tuples = _get_bin_tuples(bins)

    if report:
        print(timestamp(), "    Done!")

    return {"pops": pops, "stats": stats_to_compute, "bins": bin_tuples,
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
    h_sums = _compute_heterozygosity(matrix, pops, stats_to_compute,
                                     use_genotypes=use_genotypes,
                                     use_genotype_probs=use_genotype_probs)
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
        coords = _get_map_coords(rec_map_file, positions)
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
    For each H2 statistic, compute the average across calls to the within or
    between-diploid estimator.
    """
    h2_stats_to_compute = stats_to_compute[0]
    n_bins = len(bins) - 1
    n_pops = len(pops)
    n_stats = len(h2_stats_to_compute)
    result = np.zeros((n_bins, n_stats), dtype=np.float64)

    for stat_idx, stat in enumerate(h2_stats_to_compute):
        parts = stat.split("_")
        pop_idx = [int(x) for x in parts[1:]]
        if pop_idx[0] == pop_idx[1]:
            sample_idx = matrix.pop_map[pops[pop_idx[0]]]
            n_samples = len(sample_idx)
            numer = np.zeros(n_bins, dtype=np.float64)
            for i, idx1 in enumerate(sample_idx):
                for idx2 in sample_idx[i:]:
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
            sample_idx1 = matrix.pop_map[pops[pop_idx[0]]]
            sample_idx2 = matrix.pop_map[pops[pop_idx[1]]]
            n_samples1 = len(sample_idx1)
            n_samples2 = len(sample_idx2)
            numer = np.zeros(n_bins, dtype=np.float64)
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
    samples : 2-tuple
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
            gt = matrix.slice_sample(sample1)
            result = _h2_geno_within_diploid(gt, coords, bins, weights=weights)
        elif use_genotype_probs:
            gp = matrix.slice_sample(sample1)
            result = _h2_gp_within_diploid(gp, coords, bins, weights=weights)
        else:
            ht = matrix.slice_sample(sample1)
            result = _h2_hap_within_diploid(ht, coords, bins, weights=weights)
    else:
        if use_genotypes:
            gt1 = matrix.slice_sample(sample1)
            gt2 = matrix.slice_sample(sample2)
            result = _h2_geno_between_diploid(
                gt1, gt2, coords, bins, weights=weights)
        elif use_genotype_probs:
            gp1 = matrix.slice_sample(sample1)
            gp2 = matrix.slice_sample(sample2)
            result = _h2_gp_within_diploid(
                gp1, gp2, coords, bins, weights=weights)
        else:
            ht1 = matrix.slice_sample(sample1)
            ht2 = matrix.slice_sample(sample2)
            result = _h2_hap_within_diploid(
                ht1, ht2, coords, bins, weights=weights)

    return result


# Pairwise estimators operate on bare numpy arrays.


def _h2_hap_within_diploid(ht, coords, bins, weights=None):
    """Compute within-diploid H2 from haplotype data."""
    is_het = 1.0 * (ht[:, 0] != ht[:, 1])
    if weights is not None:
        is_het = is_het * weights
    return _compute_binned_sums(is_het, coords, bins)


def _h2_hap_between_diploid(ht1, ht2, coords, bins, weights=None):
    """Compute between-diploid H2 from haplotype data."""
    sums = 0.0
    # Average across haplotype-by-haplotype pairs
    for hap1 in ht1.T:
        for hap2 in ht2.T:
            is_het = 1.0 * (hap1 != hap2)
            if weights is not None:
                is_het = weights * is_het
                sums += _compute_binned_sums(is_het, coords, bins)
    return sums / 4


def _h2_geno_within_diploid(gt, coords, bins, weights=None):
    """Compute within-diploid H2 from genotype data."""
    is_het = 1.0 * (gt == 1)
    if weights is not None:
        is_het = weights * is_het
    return _compute_binned_sums(is_het, coords, bins)


def _h2_geno_between_diploid(gt1, gt2, coords, bins, weights=None):
    """Compute between-diploid H2 from genotype data."""
    pi_12 = np.abs(gt1 - gt2) / 2
    if weights is not None:
        pi_12 = weights * pi_12
    return _compute_binned_sums(pi_12, coords, bins)


def _h2_gp_within_diploid(gp, coords, bins, weights=None):
    """Compute within-diploid H2 from genotype probabilities."""
    p_het = gp[:, 1]
    if weights is not None:
        p_het = weights * p_het
    return _compute_binned_sums(p_het, coords, bins)


def _h2_gp_between_diploid(gp1, gp2, coords, bins, weights=None):
    """Compute between-diploid H2 from genotype probabilities."""
    p_aa_1, p_aA_1, p_AA_1 = gp1.T
    p_aa_2, p_aA_2, p_AA_2 = gp2.T
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
        upper_sites = np.searchsorted(coords, coords + upper_edge)
        upper_cum_vals = cumulative[upper_sites]
        bin_cum_vals = upper_cum_vals - lower_cum_vals
        # Take the product of left site values with cumulative right site
        # values, then sum across left sites to obtain the bin-wide sum
        binned_sums[ii] = np.sum(site_vals * bin_cum_vals)
        lower_sites = upper_sites
        lower_cum_vals = upper_cum_vals
    return binned_sums


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


def _h2_hap_within_pop(pop_counts):
    """Calculate within-population H2 from two-locus haplotype counts"""
    c1, c2, c3, c4 = pop_counts.T
    n = np.sum(pop_counts, axis=1)
    numer = c1 * c4 + c2 * c3
    denom = n * (n - 1) / 2
    return numer / denom


def _h2_hap_between_pop(pop1_counts, pop2_counts):
    """Calculate between-population H2 from two-locus haplotype counts"""
    c11, c12, c13, c14 = pop1_counts.T
    c21, c22, c23, c24 = pop2_counts.T
    n1 = np.sum(pop1_counts, axis=1)
    n2 = np.sum(pop2_counts, axis=1)
    numer = c11 * c24 + c21 * c14 + c12 * c23 + c22 * c13
    denom = n1 * n2
    return numer / denom


def _h2_geno_within_pop(pop_counts):
    """
    Compute within-population H2 from an array of two-locus genotype counts.
    """
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = pop_counts.T
    n = np.sum(pop_counts, axis=1)
    numer = (
        n1*n5
        + 2*n1*n6
        + 2*n1*n8
        + 4*n1*n9
        + n2*n4
        + n2*n5
        + n2*n6
        + 2*n2*n7
        + 2*n2*n8
        + 2*n2*n9
        + 2*n3*n4
        + n3*n5
        + 4*n3*n7
        + 2*n3*n8
        + n4*n5
        + 2*n4*n6
        + n4*n8
        + 2*n4*n9
        + 0.5*n5*(n5+1)
        + n5*n6
        + n5*n7
        + n5*n8
        + n5*n9
        + 2*n6*n7
        + n6*n8
    )
    denom = n * (2 * n - 1)
    return numer / denom


def _h2_geno_between_pop(pop1_counts, pop2_counts):
    """
    Compute between-population H2 from arrays of two-locus genotype counts.
    """
    n11, n12, n13, n14, n15, n16, n17, n18, n19 = pop1_counts.T
    n21, n22, n23, n24, n25, n26, n27, n28, n29 = pop2_counts.T
    n1 = np.sum(pop1_counts, axis=1)
    n2 = np.sum(pop2_counts, axis=1)
    numer = (
        (n11 + n12/2 + n14/2 + n15/4) * (n25/4 + n26/2 + n28/2 + n29)
        + (n15/4 + n16/2 + n18/2 + n19) * (n21 + n22/2 + n24/2 + n25/4)
        + (n12/2 + n13 + n15/4 + n16/2) * (n24/2 + n25/4 + n27 + n28/2)
        + (n14/2 + n15/4 + n17 + n18/2) * (n22/2 + n23 + n25/4 + n26/2)
    )
    denom = n1 * n2
    return numer / denom


# -----------------------------------------------------------------------------
# Heterozygosity statistics
# -----------------------------------------------------------------------------


def _compute_heterozygosity(
    matrix,
    pops,
    stats_to_compute,
    use_genotypes=True,
    use_genotype_probs=False,
):
    """Calculate H statistics. Strictly for biallelic matrices."""
    if use_genotype_probs:
        result = _compute_heterozygosity_gp(matrix, pops, stats_to_compute)
        return result

    hap_counts = []
    ref_counts = []
    alt_counts = []

    for pop in pops:
        mat = matrix.slice_pop(pop)
        n_alt = np.sum(mat, axis=1)
        if use_genotypes:
            n_hap = np.full_like(n_alt, 2 * mat.shape[1])
        else:
            n_hap = np.full_like(n_alt, mat.shape[1])
        n_ref = n_hap - n_alt
        hap_counts.append(n_hap)
        ref_counts.append(n_ref)
        alt_counts.append(n_alt)

    h_stats_to_compute = stats_to_compute[1]
    n_stats = len(h_stats_to_compute)
    result = np.zeros(n_stats, dtype=np.float64)

    for stat_idx, stat in enumerate(h_stats_to_compute):
        parts = stat.split("_")
        pop_idx = [int(x) for x in parts[1:]]
        n_hap1 = hap_counts[pop_idx[0]]
        n_ref1 = ref_counts[pop_idx[0]]
        n_alt1 = alt_counts[pop_idx[0]]
        if pop_idx[0] == pop_idx[1]:
            numer = 2 * n_alt1 * n_ref1
            denom = n_hap1 * (n_hap1 - 1)
        else:
            n_hap2 = hap_counts[pop_idx[1]]
            n_ref2 = ref_counts[pop_idx[1]]
            n_alt2 = alt_counts[pop_idx[1]]
            numer = n_ref1 * n_alt2 + n_ref2 * n_alt1
            denom = n_hap1 * n_hap2
        # Take the sum over site heterozygosities
        result[stat_idx] = np.sum(numer / denom)

    return result


def _compute_heterozygosity_gp(matrix, pops, stats_to_compute):
    """Compute H from genotype probabilities."""
    h_stats_to_compute = stats_to_compute[1]
    n_stats = len(h_stats_to_compute)
    result = np.zeros(n_stats, dtype=np.float64)
    return result


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


def _get_map_coords(rec_map_file, positions):
    """Assign map coordinates to positions by loading a recombination map."""
    map_pos, map_coords = utils._read_rec_map_file(rec_map_file)
    # Assume that recombination map positions are 1-indexed
    return np.interp(positions + 1, map_pos, map_coords)


def _get_mut_rates(mut_map_file, positions):
    """Assign mutation rates to positions."""
    mut_map = utils._read_mut_map_file(mut_map_file)
    return mut_map[positions]


def _get_bed_file_positions(bed_file, interval=None):
    """Load 0-indexed positions from a BED file."""
    mask = GeneticMask.from_bed_file(bed_file, interval=interval)
    return mask.to_positions()

