"""
Estimate H2 from sequence data.

Usage
-----
import h2py
L = 100_000_000
intervals = [[1_000_000 * i, 1_000_000 * (i + 1) for i in range(int(L / 1e6))]]
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
import warnings

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
    rec_pos_col="Position(bp)",
    rec_map_col="Map(cM)",
    r_bins=None,
    bp_bins=None,
    r=None,
    min_bp=None,
    max_bp=None,
    mut_map_file=None,
    u_bar=1,
    use_genotype_probs=False,
    use_haplotypes=False,
    report=True,
    compute_denoms=True,
    stats_to_compute=None,
    pairwise=True,
    ac_filter=True,
    filtered=False,
    genotype_matrix=None,
    genotype_prob_matrix=None,
    haplotype_matrix=None,
):
    """
    Compute H2 statistics on a chromosome or chromosome interval.

    Parameters
    ----------
    vcf_file : str, path
        Path to VCF file.
    pop_file : str, path, optional
        Path to population specification file; a whitespace-separated file with
        two columns, headered 'sample' and 'pop', which map VCF samples to
        population labels. Samples in the VCF absent from ``pop_file`` are
        ignored. If None, each VCF sample is assigned to a unique population.
    pops : list, optional
        Populations from ``pop_file`` for which to compute H2. If None
        (default), all populations specified in ``pop_file`` are parsed, in
        order of first appearance.
    bed_file : str, path, optional
        Path to BED file that specifies accessible sites; required if
        ``compute_denoms`` is True.
    chromosome : str, optional
        Chromosome to parse (if VCF or BED files record several chromosomes).
    interval : tuple, length 2, optional
        BED-style (0-indexed, half-open) genomic interval to parse. Sites
        outside the interval are ignored.
    rec_map_file : str, path, optional
        Path to recombination map file. This must be whitespace-separated with
        columns labelled 'Position(bp)', 'Map(cM)'.
    r_bins : array-like, optional
        Bin edges, defined in recombination fractions. Required if
        ``rec_map_file`` is not None.
    bp_bins : array-like, optional
        Bin edges defined in physical units (base pairs).
    min_bp : int, optional
        Minimum allowable distance (inclusive) between sites. Any site pairs
        closer together than ``min_bp`` are skipped.
    max_bp : int, optional
        Maximum allowable distance (exclusive) between sites.
    mut_map_file : str, path, optional
        Path to mutation map file for weighting site contributions to H2. This
        may be either (1) a .npy array file with site-resolution mutation
        rates, with ``np.nan`` where data is missing, or (2) a .csv or .tsv
        file (file separator defined by extension) with columns 'chromStart',
        'chromEnd', and 'mutRate' that defines mutation rates on contiguous
        intervals.
    u_bar : float, optional
        Factor by which to divide mutation rates in mutation rate-based
        scaling. If None (default), the average site rate is used. #TODO correct?
    use_genotype_probs : bool, optional
        If True (default False), compute stats from genotype probabilities.
        The default behavior is to use genotypes.
    use_haplotypes : bool, optional
        If True (default False), treat the input VCF as phased and apply
        haplotype estimators of H2.
    report : bool, optional
        If True (default), print verbose status messages.
    compute_denoms : bool, optional
        If True (default), calculate denominators for H2 and H; the numbers of
        accessible site pairs and sites, respectively.
    stats_to_compute : tuple, length 2, optional
        Holds lists of H2 and H statistics to calculate, 'H2_{i}_{j}' and
        'H_{i}_{j}`. ``i`` and ``j`` index ``pops``. If None (default),
        compute all statistics for ``pops``.
    pairwise : bool, optional
        If True, average across within and between-diploid H2 estimators. If
        False, use ``moments.LD.Parsing`` functions to count two-locus
        genotypes/haplotypes and apply multi-sample estimators to these.
        Pairwise estimators must be used if H2 is estimated from genotype
        probabilities.
    ac_filter : bool, optional
        Allele count filter. If True (default), ignore multiallelic sites.
        Affects only the haplotype estimators; estimation with genotypes or
        genotype probabilities always drops multiallelic sites.
    filtered : bool, optional
        If True (default False), skip VCF rows without 'PASS' in the 'FILTER'
        field.
    genotype_matrix : GenotypeMatrix.
        Preloaded genotype matrix to parse. Ditto for ``genotype_prob_matrix``
        and ``haplotype_matrix``. The provision of one of these overrides
        ``use_genotype_probs`` and ``use_haplotypes``.
        # TODO internal masking of preloaded file...

    Returns
    -------
    dict
        A dictionary with keys 'bins', 'pops', 'stats', 'sums', 'denoms'.
    """
    # Check arguments
    if use_genotype_probs and use_haplotypes:
        raise ValueError(
            "`use_genotype_probs`, `use_haplotypes` cannot both be True")

    if use_genotype_probs and not pairwise:
        warings.warn("`use_genotype_probs` has no multi-diploid estimators:"
                     "forcing `pairwise=True`")

    if use_genotype_probs and not compute_denoms:
        warnings.warn("`use_genotype_probs` forces denominator calculation:"
                      "ignoring `compute_denoms=False`")

    # Check matrix types and override `use_genotype_probs`, `use_haplotypes`
    preloaded = False
    if genotype_matrix is not None:
        assert isinstance(genotype_matrix, GenotypeMatrix)
        if haplotype_matrix is not None or genotype_prob_matrix is not None:
            raise ValueError("only one `matrix` instance is allowed")
        use_genotype_probs = use_haplotypes = False
        preloaded = True
    elif genotype_prob_matrix is not None:
        assert isinstance(genotype_prob_matrix, GenotypeProbMatrix)
        if haplotype_matrix is not None:
            raise ValueError("only one `matrix` instance is allowed")
        use_genotype_probs = True
        use_haplotypes = False
        preloaded = True
    elif haplotype_matrix is not None:
        assert isinstance(haplotype_matrix, HaplotypeMatrix)
        use_genotype_probs = False
        use_haplotypes = True
        preloaded = True
    else:
        if vcf_file is None:
            raise ValueError("`vcf_file` or preloaded matrix is required")

    if vcf_file is not None and preloaded:
        raise ValueError("cannot use both `vcf_file` and preloaded matrix")

    if report:
        print(timestamp(), "Preparing data ...")

    # Load BED file
    if bed_file is not None:
        mask = GeneticMask.from_bed_file(bed_file, interval=interval)
    else:
        # TODO downstream implications?
        mask = None

    # Genotype probability path
    if use_genotype_probs:
        if vcf_file is not None:
            matrix = GenotypeProbMatrix.from_vcf(
                vcf_file,
                mask=mask,
                interval=interval,
                chromosome=chromosome,
                pop_file=pop_file,
                filtered=filtered,
            )
        else:
            matrix = genotype_prob_matrix
            matrix.apply_mask(mask)
    # Haplotype path
    elif use_haplotypes:
        if vcf_file is not None:
            matrix = HaplotypeMatrix.from_vcf(
                vcf_file,
                mask=mask,
                interval=interval,
                chromosome=chromosome,
                pop_file=pop_file,
                ac_filter=ac_filter,
                filtered=filtered,
            )
        else:
            # TODO masking
            matrix = haplotype_matrix
    # Genotype path (default)
    else:
        if vcf_file is not None:
            matrix = GenotypeMatrix.from_vcf(
                vcf_file,
                mask=mask,
                interval=interval,
                chromosome=chromosome,
                pop_file=pop_file,
                filtered=filtered,
            )
        else:
            matrix = genotype_matrix
            if mask is not None:
                matrix.apply_mask(mask)
    if report:
        print(timestamp(), f"Prepared {matrix}")

    # Load recombination map data and check bins
    if r_bins is not None:
        if r is None:
            if rec_map_file is None:
                raise ValueError("rec_map_file is required")
            coords = _get_map_coords(rec_map_file, matrix.positions,
                                     rec_pos_col, rec_map_col)
        else:
            coords = r * matrix.positions
        # Transform bin units to allow direct comparison to a genetic map
        init_bins = r_bins
        bins = utils._map_function(np.array(r_bins))
        if compute_denoms and not use_genotype_probs:
            if mask is not None:
                all_pos = mask.to_positions()
            else:
                if interval is None:
                    raise ValueError(
                        "`interval` or `mask` required to compute denoms")
                all_pos = np.arange(interval[0], interval[1])
            if r is None:
                all_coords = _get_map_coords(rec_map_file, all_pos,
                                             rec_pos_col, rec_map_file)
            else:
                all_coords = r * all_pos
        if report:
            if len(all_coords) > 0:
                print(timestamp(), "Loaded map coordinates",
                    f"({all_coords[0]:.4} to {all_coords[-1]:.4} M)")
            else:
                print(timestamp(), "Loaded 0 map coordinates (empty window)")
    else:
        if bp_bins is None:
            raise ValueError("r_bins or bp_bins is required")
        init_bins = bins = np.array(bp_bins)
        coords = matrix.positions
        if compute_denoms and not use_genotype_probs:
            if mask is not None:
                all_pos = all_coords = mask.to_positions()
            else:
                if interval is None:
                    raise ValueError(
                        "`interval` or `mask` required to compute denoms")
                all_pos = all_coords = np.arange(interval[0], interval[1])
        if report:
            print(timestamp(),
                  f"Using physical positions ({coords[0]} to {coords[-1]} bp)")

    bin_tuples = _get_bin_tuples(init_bins)

    if len(all_coords) == 0:
        dummy = [0 for _ in range(len(bins))]
        return {"pops": pops, "stats": stats_to_compute, "bins": bin_tuples,
                "sums": dummy, "denoms": dummy, "u_avg": 0}

    # Prepare some specifications for estimation
    if pops is None:
        pops = matrix.pops

    if stats_to_compute is None:
        n_pops = len(pops)
        stats_to_compute = (utils._h2_names(n_pops), utils._h_names(n_pops))

    u_avg = None

    if len(matrix.positions) > 0:
        # Load mutation map data
        if mut_map_file is not None:
            mut_map = _get_mut_rates(mut_map_file, matrix.positions)
            if u_bar is None:
                u_bar = np.mean(mut_map)
                if report:
                    print(timestamp(), f"  Using u_bar = {u_bar:.4}")
            weights = u_bar / mut_map
        else:
            weights = None

        # Genotype probability pathway
        if use_genotype_probs:
            if report:
                print(timestamp(), "Computing statistics and denominators ...")
            sums_list, denoms_list = _call_genotype_prob_h2_estimators(
                matrix,
                coords,
                bins,
                pops,
                stats_to_compute,
                weights=weights,
                pos=matrix.positions,
                min_bp=min_bp,
                max_bp=max_bp
            )
            if report:
                print(timestamp(), "Computed statistics and denominators.")
        # Genotype/haplotype pathway
        else:
            if report:
                print(timestamp(), "Computing statistics ...")
            if pairwise:
                sums_list = _call_pairwise_h2_estimators(
                    matrix,
                    coords,
                    bins,
                    pops,
                    stats_to_compute,
                    weights=weights,
                    use_haplotypes=use_haplotypes,
                    pos=matrix.positions,
                    min_bp=min_bp,
                    max_bp=max_bp,
                )
            else:
                sums_list = _call_pooled_h2_estimators()
            if report:
                print(timestamp(), "Computed statistics.")
    else:
        sums_list = [0 for _ in range(len(bins))]

    if compute_denoms:
        if report:
            print(timestamp(), "Computing denominators ...")

        if mut_map_file is not None:
            full_mut_map = _get_mut_rates(mut_map_file, all_pos)
            u_avg = np.mean(full_mut_map)
        else:
            u_avg = None

        denoms = _compute_h2_denoms(
            all_coords,
            bins,
            pos=all_pos,
            min_bp=min_bp,
            max_bp=max_bp,
        )
        denoms_list = [row for row in denoms]
        if report:
            print(timestamp(), "Computed denominators.")
    else:
        denoms_list = None
        u_avg = None

    if report:
        print(timestamp(), "    Done!")

    return {"pops": pops, "stats": stats_to_compute, "bins": bin_tuples,
            "sums": sums_list, "denoms": denoms_list, "u_avg": u_avg}


def compute_h2_denoms(
    bed_file=None,
    rec_map_file=None,
    r_bins=None,
    bp_bins=None,
    interval=None,
):
    """
    Compute the denominator of the H2 statistic- the number of pairs of
    accessible sites, binned by distance.

    The last element of the denominator array holds the denominator of the
    heterozygosity statistic, the number of accessible sites.

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
        bins = utils._map_function(r_bins)
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


# =============================================================================
# Boostrap/subset functions
# =============================================================================


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
    # If only one sample is present, force covs to be 2d
    if len(means[0]) == 1:
        varcovs = [c[None, None] for c in varcovs]
    template = all_data[labels[0]]
    return {"pops": template["pops"], "stats": template["stats"],
            "bins": template["bins"], "means": means, "varcovs": varcovs}


def get_means_across_regions(all_data, compute_u_fac=False):
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

    if compute_u_fac:
        n_sites = 0
        u_tot = 0.0
        for label in labels:
            n_sites_window = all_data[label]["denoms"][-1]
            if n_sites_window is None:
                continue
            n_sites += n_sites_window
            u_tot += all_data[label]["avg_u"] * n_sites_window
        u_avg = u_tot / n_sites
        fac = 1 / u_avg ** 2
        print(timestamp(), "Computed u_fac = {u_fac:.4}")
    else:
        fac = 1.0

    numers = [0.0 * row for row in all_data[labels[0]]["sums"]]
    denoms = [0.0 * row for row in all_data[labels[0]]["denoms"]]
    for label in labels:
        for ii in range(len(numers)):
            numers[ii] += all_data[label]["sums"][ii]
            denoms[ii] += all_data[label]["denoms"][ii]
    means = [n * fac / d for n, d in zip(numers, denoms)]

    if compute_u_fac:
        return means, u_avg
    else:
        return means


def compute_avg_u(all_data):
    labels = labels = list(all_data.keys())
    n_sites = 0
    u_tot = 0.0
    for label in labels:
        n_sites_window = all_data[label]["denoms"][-1]
        if n_sites_window == 0:
            continue
        n_sites += n_sites_window
        u_tot += all_data[label]["u_avg"] * n_sites_window
    u_avg = u_tot / n_sites
    return u_avg, n_sites


def subset_data(data, to_pops=None, min_r=None, max_r=None):
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
        if min_r is not None:
            if b[0] < min_r:
                continue
        if max_r is not None:
            if b[1] > max_r:
                continue
        bins.append(b)
        new_means.append(means[ii])
        new_varcovs.append(varcovs[ii])

    return {"pops": pops, "stats": stats, "bins": bins, "means": new_means,
            "varcovs": new_varcovs}


# -----------------------------------------------------------------------------
# Bootstrap utility functions
# -----------------------------------------------------------------------------


def get_bootstrap_replicates(all_data, n_reps=None, n_samples=None):
    """
    Draw several bootstrap replicates from a list of sums computed on genomic
    intervals.

    Parameters
    ----------
    all_data : dict
        Maps genomic interval labels to dicts following the output of TODO
    n_reps : int, optional
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
    if n_reps is None:
        n_reps = len(all_data)
    if n_samples is None:
        n_samples = len(all_data)

    labels = list(all_data.keys())
    all_means = []
    for ii in range(n_reps):
        sample_data = dict()
        for jj in range(n_samples):
            label = np.random.choice(labels)
            sample_data[jj] = all_data[label]
        sample_means = get_means_across_regions(sample_data)
        all_means.append(sample_means)
    return all_means


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


def _safe_divide(numer, denom):
    """
    Perform element-wise division, skipping elements where ``denom`` is 0.
    """
    valid = denom > 0.0
    result = np.zeros(len(numer), dtype=np.float64)
    result[valid] = numer[valid] / denom[valid]
    return result


# -----------------------------------------------------------------------------
# Vectorized denominator calculation
# -----------------------------------------------------------------------------


def _compute_h2_denoms(
    coords,
    bins,
    pos=None,
    min_bp=None,
    max_bp=None
):
    """Compute binned denominators for two-locus statistics and H.

    Parameters
    ----------
    coords : np.ndarray, shape (n_pos,)
        Coordinates of accessible base pairs, in genetic map (Morgans, M) or
        physical (base pairs, bp) distances.
    bins : array-like, shape (n_bins + 1,)
        Bin edges, in genetic map units (M) or physical distances (bp)
        matching ``coords``.
    pos : np.ndarray, optional, shape (n_pos,)
        Physical positions of sites represented in ``coords``. Required to
        impose minimum/maximum physical distances.
    min_bp : int, optional
        Minimum physical distance between sites, in base pairs.
    max_bp : int, optional
        Maximum physical distance between sites, in base pars.

    Returns
    -------
    binned_denoms : np.ndarray, shape (n_bins + 1,)
        Distance-binned tallies of accessible site pairs. The last element is
        the number of accessible sites.
    """
    binned_denoms = np.zeros(len(bins), dtype=np.float64)

    # Precompute bounds due to min/max physical distances
    if min_bp is not None and min_bp > 1:
        min_idx = np.searchsorted(pos, pos + min_bp)
    else:
        min_idx = np.arange(1, len(coords) + 1)
    if max_bp is not None:
        max_idx = np.searchsorted(pos, pos + max_bp)
    else:
        max_idx = None

    # Get indices of lowest-index right loci in the zeroth bin
    lower_idx = np.searchsorted(coords, coords + bins[0])
    lower_idx = np.maximum(lower_idx, min_idx)

    for ii, upper_edge in enumerate(bins[1:]):
        # Get indices (exclusive) of highest-index right loci in bin ``ii``
        upper_idx = np.searchsorted(coords, coords + upper_edge)
        # TODO this is always not None
        if min_idx is not None:
            upper_idx = np.maximum(upper_idx, min_idx)
        if max_idx is not None:
            upper_idx = np.minimum(upper_idx, max_idx)
        binned_denoms[ii] = np.sum(upper_idx - lower_idx)
        lower_idx = upper_idx

    # The number of accessible sites, for normalizing heterozygosity
    binned_denoms[-1] = len(coords)

    return binned_denoms


# -----------------------------------------------------------------------------
# Genotype probability estimators
# -----------------------------------------------------------------------------


def _call_genotype_prob_h2_estimators(
    matrix,
    coords,
    bins,
    pops,
    stats_to_compute,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """
    Call estimator functions to compute H2 from genotype probability data.

    # TODO document
    """
    h2_stats_to_compute = stats_to_compute[0]
    n_stats = len(h2_stats_to_compute)
    n_rows = len(bins)
    sums_arr = np.zeros((n_rows, n_stats), dtype=np.float64)
    denoms_arr = np.zeros((n_rows, n_stats), dtype=np.int64)

    for stat_idx, stat in enumerate(h2_stats_to_compute):
        parts = stat.split("_")
        pop_idx = [int(x) for x in parts[1:]]
        sums_stat = np.zeros(n_rows, dtype=np.float64)
        denoms_stat = np.zeros(n_rows, dtype=np.int64)

        # Within-population
        if pop_idx[0] == pop_idx[1]:
            sample_idx = matrix.pop_map[pops[pop_idx[0]]]
            for idx in sample_idx:
                arr = matrix.slice_sample(idx)
                # Account for sample-specific missing data
                mask = matrix.get_non_missing_mask(idx)
                sums_out, denoms_out = _h2_gp_within_diploid(
                    arr[mask],
                    coords[mask],
                    bins,
                    weights=weights[mask] if weights is not None else None,
                    pos=pos[mask] if pos is not None else None,
                    min_bp=min_bp,
                    max_bp=max_bp,
                )
                sums_stat += sums_out
                denoms_stat += denoms_out

        # Between-population
        else:
            sample_idx1 = matrix.pop_map[pops[pop_idx[0]]]
            sample_idx2 = matrix.pop_map[pops[pop_idx[1]]]
            for idx1 in sample_idx1:
                arr1 = matrix.slice_sample(idx1)
                for idx2 in sample_idx2:
                    arr2 = matrix.slice_sample(idx2)
                    mask = matrix.get_non_missing_mask([idx1, idx2])
                    sums_out, denoms_out = _h2_gp_between_diploid(
                        arr1[mask],
                        arr2[mask],
                        coords[mask],
                        bins,
                        weights=weights[mask] if weights is not None else None,
                        pos=pos[mask] if pos is not None else None,
                        min_bp=min_bp,
                        max_bp=max_bp,
                    )
                    sums_stat += sums_out
                    denoms_stat += denoms_out

        sums_arr[:, stat_idx] = sums_stat
        denoms_arr[:, stat_idx] = denoms_stat

    sums_list = [row for row in sums_arr]
    denoms_list = [row for row in denoms_arr]

    return sums_list, denoms_list


def _h2_gp_within_diploid(
    arr,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """Compute within-diploid H2 from genotype probabilities."""
    p_het = arr[:, 1]
    return _bin_pair_products(p_het, coords, bins, pos=pos, weights=weights,
                              min_bp=min_bp, max_bp=max_bp)


def _h2_gp_between_diploid(
    arr1,
    arr2,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """Compute between-diploid H2 from genotype probabilities."""
    p_alt1 = 0.5 * arr1[:, 1] + arr1[:, 2]
    p_ref1 = 1 - p_alt1
    p_alt2 = 0.5 * arr2[:, 1] + arr2[:, 2]
    p_ref2 = 1 - p_alt2
    p_diff = p_alt1 * p_ref2 + p_alt2 * p_ref1
    return _bin_pair_products(p_diff, coords, bins, pos=pos, weights=weights,
                              min_bp=min_bp, max_bp=max_bp)


# -----------------------------------------------------------------------------
# Pairwise genotype/haplotype estimators
# -----------------------------------------------------------------------------


def _call_pairwise_h2_estimators(
    matrix,
    coords,
    bins,
    pops,
    stats_to_compute,
    use_haplotypes=False,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """
    Call estimator functions to estimate H2 from genotype or haplotype data.

    #TODO document me
    """
    h2_stats_to_compute = stats_to_compute[0]
    n_stats = len(h2_stats_to_compute)
    n_rows = len(bins)
    sums_arr = np.zeros((n_rows, n_stats), dtype=np.float64)

    for stat_idx, stat in enumerate(h2_stats_to_compute):
        parts = stat.split("_")
        pop_idx = [int(x) for x in parts[1:]]
        sums_stat = np.zeros(n_rows, dtype=np.float64)

        # Within-population path: average over within-diploid estimators
        if pop_idx[0] == pop_idx[1]:
            sample_idx = matrix.pop_map[pops[pop_idx[0]]]
            n_calls = len(sample_idx)

            for idx in sample_idx:
                arr = matrix.slice_sample(idx)
                if use_haplotypes:
                    sums_stat += _h2_hap_within_diploid(
                        arr,
                        coords,
                        bins,
                        weights=weights,
                        pos=pos,
                        min_bp=min_bp,
                        max_bp=max_bp
                    )
                else:
                    sums_stat += _h2_geno_within_diploid(
                        arr,
                        coords,
                        bins,
                        weights=weights,
                        pos=pos,
                        min_bp=min_bp,
                        max_bp=max_bp,
                    )
        # Between-population path: average over between-diploid estimators
        else:
            sample_idx1 = matrix.pop_map[pops[pop_idx[0]]]
            sample_idx2 = matrix.pop_map[pops[pop_idx[1]]]
            n_calls = len(sample_idx1) * len(sample_idx2)

            for idx1 in sample_idx1:
                arr1 = matrix.slice_sample(idx1)
                for idx2 in sample_idx2:
                    arr2 = matrix.slice_sample(idx2)
                    if use_haplotypes:
                        sums_stat += _h2_hap_between_diploid(
                            arr1,
                            arr2,
                            coords,
                            bins,
                            weights=weights,
                            pos=pos,
                            min_bp=min_bp,
                            max_bp=max_bp,
                        )
                    else:
                        sums_stat += _h2_geno_between_diploid(
                            arr1,
                            arr2,
                            coords,
                            bins,
                            weights=weights,
                            pos=pos,
                            min_bp=min_bp,
                            max_bp=max_bp,
                        )

        # Average across calls
        sums_arr[:, stat_idx] = sums_stat / n_calls

    # Transform the output array into a list, with one element for each bin
    sums_list = [row for row in sums_arr]

    return sums_list


# - Pairwise estimators operate on bare numpy arrays. -------------------------


def _h2_hap_within_diploid(
    arr,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """Compute within-diploid H2 from haplotype data."""
    is_het = 1.0 * (arr[:, 0] != arr[:, 1])
    return _bin_pair_products(is_het, coords, bins, weights=weights, pos=pos,
                              min_bp=min_bp, max_bp=max_bp)[0]


def _h2_hap_between_diploid(
    arr1,
    arr2,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """Compute between-diploid H2 from haplotype data."""
    sums = 0.0
    # Average across haplotype-by-haplotype pairs
    for hap1 in arr1.T:
        for hap2 in arr2.T:
            is_het = 1.0 * (hap1 != hap2)
            sums += _bin_pair_products(is_het, coords, bins, weights=weights,
                                       pos=pos, min_bp=min_bp, max_bp=max_bp)[0]
    return sums / 4


def _h2_geno_within_diploid(
    arr,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """Compute within-diploid H2 from genotype data."""
    is_het = 1.0 * (arr == 1)
    return _bin_pair_products(is_het, coords, bins, weights=weights, pos=pos,
                              min_bp=min_bp, max_bp=max_bp)[0]


def _h2_geno_between_diploid(
    arr1,
    arr2,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """Compute between-diploid H2 from genotype data."""
    p_alt1 = arr1 / 2
    p_ref1 = 1 - p_alt1
    p_alt2 = arr2 / 2
    p_ref2 = 1 - p_alt2
    p_diff = p_alt1 * p_ref2 + p_alt2 * p_ref1
    return _bin_pair_products(p_diff, coords, bins, weights=weights, pos=pos,
                              min_bp=min_bp, max_bp=max_bp)[0]


def _bin_pair_products(
    scores,
    coords,
    bins,
    weights=None,
    pos=None,
    min_bp=None,
    max_bp=None,
):
    """
    Engine for calculating aggregate H2 from precomputed site ``scores``.

    Specifically, for each unique pair of sites (i, j), compute the product
    ``scores[i] * scores[j]`` and increment it to the bin which captures
    distance ``coords[j] - coords[i]``.

    Parameters
    ----------
    scores : np.ndarray, shape (n_sites,)
        Value with which to compute H2.
        For within-diploid estimators, this is site heterozygosity.
    coords : np.ndarray, shape (n_sites,)
        Site genetic map coordinates OR physical positions.
    bins : array-like, shape (n_bins + 1,)
        Distance bins. These must share units with ``coords``.
    weights : np.ndarray, shape (n_sites,), optional
        Optional site weights.
    pos : np.ndarray, shape (n_sites,), optional
        Physical positions of sites. Required to specify min/max pair distance.
    min_bp : int, optional
        Sites separated by less than this minimum physical distance are skipped
    max_bp : int, optional
        Sites separated by more than this max. physical distance are skipped.

    Returns
    -------
    binned_prods : np.ndarray, shape (n_bins + 1,)
        Products of ``scores`` for possible site pairs, summed in bins by
        distance.
    pair_counts : np.ndarray, shape (n_bins + 1,)
        Number of site pairs observed in each bin. If ``coords`` encompasses
        all accessible sites, this is the denominator of H2.
    """
    if pos is None and (min_bp is not None or max_bp is not None):
        raise ValueError("`pos` is required to specify min/max distance")

    binned_prods = np.zeros(len(bins), dtype=np.float64)
    pair_counts = np.zeros(len(bins), dtype=np.int64)

    # Apply weights if given
    if weights is not None:
        scores = scores * weights

    # Precompute bounds due to min/max physical distances
    if min_bp is not None and min_bp > 1:
        min_idx = np.searchsorted(pos, pos + min_bp)
    else:
        min_idx = np.arange(1, len(coords) + 1)
    if max_bp is not None:
        max_idx = np.searchsorted(pos, pos + max_bp)
    else:
        max_idx = None

    # Get indices of lowest-index right loci in the zeroth bin
    lower_idx = np.maximum(np.searchsorted(coords, coords + bins[0]), min_idx)

    cum_scores = np.concatenate([[0], np.cumsum(scores)])

    for ii, upper_edge in enumerate(bins[1:]):
        upper_idx = np.searchsorted(coords, coords + upper_edge)
        if min_idx is not None:
            upper_idx = np.maximum(upper_idx, min_idx)
        if max_idx is not None:
            upper_idx = np.minimum(upper_idx, max_idx)
        score_diffs = cum_scores[upper_idx] - cum_scores[lower_idx]
        binned_prods[ii] = np.sum(scores * score_diffs)
        pair_counts[ii] = np.sum(upper_idx - lower_idx)
        lower_idx = upper_idx

    binned_prods[-1] = np.sum(scores)
    pair_counts[-1] = len(scores)

    return binned_prods, pair_counts


# -----------------------------------------------------------------------------
# Multi-sample H2 estimators. These take precomputed two-locus type counts.
# -----------------------------------------------------------------------------


def _call_pooled_h2_estimators(
    matrix,
    coords,
    bins,
    pops,
    stats_to_compute,
    weights,
    use_haplotypes=False,
    min_bp=None,
    max_bp=None,
):
    """Calculate H2 by counting types and applying multi-sample estimators."""
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
    n5 = pop_counts[:, 4]
    n = np.sum(pop_counts, axis=1)
    return n5 / n


def _h2_geno_between_pop(pop1_counts, pop2_counts):
    """
    Compute between-population H2 from arrays of two-locus genotype counts.
    """
    n11, n12, n13, n14, n15, n16, n17, n18, n19 = pop1_counts.T
    n21, n22, n23, n24, n25, n26, n27, n28, n29 = pop2_counts.T
    n1 = np.sum(pop1_counts, axis=1)
    n2 = np.sum(pop2_counts, axis=1)
    numer = ((n11 + n12/2 + n14/2 + n15/4) * (n25/4 + n26/2 + n28/2 + n29)
              + (n15/4 + n16/2 + n18/2 + n19) * (n21 + n22/2 + n24/2 + n25/4)
              + (n12/2 + n13 + n15/4 + n16/2) * (n24/2 + n25/4 + n27 + n28/2)
              + (n14/2 + n15/4 + n17 + n18/2) * (n22/2 + n23 + n25/4 + n26/2))
    denom = n1 * n2
    return numer / denom


# -----------------------------------------------------------------------------
# Heterozygosity statistics
# -----------------------------------------------------------------------------


def _compute_heterozygosity(
    matrix,
    pops,
    stats_to_compute,
    use_genotype_probs=False,
    use_haplotypes=False,
):
    """Calculate H statistics. Strictly for biallelic matrices."""
    if use_genotype_probs:
        return _compute_gp_heterozygosity(matrix, pops, stats_to_compute)

    h_stats_to_compute = stats_to_compute[1]
    n_stats = len(h_stats_to_compute)
    result = np.zeros(n_stats, dtype=np.float64)

    # Precompute haplotype/reference allele counts
    hap_counts = []
    alt_counts = []
    for pop in pops:
        mat = matrix.slice_pop(pop)
        n_alt = np.sum(mat, axis=1)
        if use_haplotypes:
            n_hap = np.full_like(n_alt, mat.shape[1])
        else:
            n_hap = np.full_like(n_alt, 2 * mat.shape[1])
        hap_counts.append(n_hap)
        alt_counts.append(n_alt)

    for stat_idx, stat in enumerate(h_stats_to_compute):
        parts = stat.split("_")
        pop_idx = [int(x) for x in parts[1:]]
        n_hap1 = hap_counts[pop_idx[0]]
        n_alt1 = alt_counts[pop_idx[0]]
        # within-population
        if pop_idx[0] == pop_idx[1]:
            numer = 2 * n_alt1 * (n_hap1 - n_alt1)
            denom = n_hap1 * (n_hap1 - 1)
        else:
            n_hap2 = hap_counts[pop_idx[1]]
            n_alt2 = alt_counts[pop_idx[1]]
            numer = n_alt2 * (n_hap1 - n_alt1) + n_alt1 * (n_hap2 - n_alt2)
            denom = n_hap1 * n_hap2
        # Take the sum over site heterozygosities
        result[stat_idx] = np.sum(numer / denom)

    return result


def _compute_gp_heterozygosity(matrix, pops, stats_to_compute):
    """Compute H from genotype probabilities."""
    h_stats_to_compute = stats_to_compute[1]
    n_stats = len(h_stats_to_compute)
    result = np.zeros(n_stats, dtype=np.float64)

    # Precompute alternate allele probabilities
    n_sites = len(matrix)
    n_pops = len(pops)
    p_alt_matrix = np.zeros((n_sites, n_pops), dtype=np.float64)
    for idx, pop in enumerate(pops):
        pop_matrix = matrix.slice_pop(pop)
        n_samples = len(matrix.pop_map[pop])
        # Get the prob. of sampling the alternate allele for each sample
        p_alts = (0.5 * pop_matrix[:, 1::3] + pop_matrix[:, 2::3])
        p_alt_matrix[:, idx] = np.sum(p_alts, axis=1) / n_samples

    for stat_idx, stat in enumerate(h_stats_to_compute):
        parts = stat.split("_")
        pop_idx = [int(x) for x in parts[1:]]

        # Within-population path. Average over probabilities of heterozygosity
        # for each diploid.
        if pop_idx[0] == pop_idx[1]:
            sample_idx = matrix.pop_map[pops[pop_idx[0]]]
            numer = 0.0
            denom = len(sample_idx)
            for idx in sample_idx:
                m = matrix.slice_sample(idx)
                p_het = m[:, 1]
                numer += np.sum(p_het)
            result[stat_idx] = numer / denom
        # Between-population path. Use precomputed alternate allele probs
        else:
            p_alt1 = p_alt_matrix[:, pop_idx[0]]
            p_ref1 = 1 - p_alt1
            p_alt2 = p_alt_matrix[:, pop_idx[1]]
            p_ref2 = 1 - p_alt2
            p_diff = p_alt1 * p_ref2 + p_alt2 * p_ref1
            result[stat_idx] = np.sum(p_diff)

    return result


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _get_bin_tuples(bins):
    """Get a list of 2-tuples with bin edges from a vector of bin edges."""
    unfolded_bins = []
    for ii in range(len(bins) - 1):
        unfolded_bins.append((float(bins[ii]), float(bins[ii + 1])))
    return unfolded_bins


def _get_map_coords(rec_map_file, positions, pos_col, map_col):
    """Assign map coordinates to positions by loading a recombination map."""
    map_pos, map_coords = utils._read_rec_map_file(
        rec_map_file, pos_col=pos_col, map_col=map_col)
    # Assume that recombination map positions are 1-indexed
    return np.interp(positions + 1, map_pos, map_coords)


def _get_mut_rates(mut_map_file, positions):
    """Assign mutation rates to positions."""
    mut_map = utils._read_mut_map_file(mut_map_file, L=positions[-1] + 1)
    ret = mut_map[positions]
    assert not np.any(np.isnan(ret))
    return ret


def _get_bed_file_positions(bed_file, interval=None):
    """Load 0-indexed positions from a BED file."""
    mask = GeneticMask.from_bed_file(bed_file, interval=interval)
    return mask.to_positions()

