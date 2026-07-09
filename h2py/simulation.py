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


def generate_genotype_probs(ts, ref_seq=None, depth=5, p_err=0.01):
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
    sample_depths = np.zeros((seq_len, n_samples))

    for ii in range(n_samples):
        sample = haplotypes[:, 2*ii:2*(ii+1)]

        # 'cheat' by calculating priors with true data
        p_0 = np.sum(sample) / (2 * seq_len)
        p_1 = 1 - p_0
        p_het = np.sum(sample[:, 0] != sample[:, 1]) / (2 * seq_len)
        priors = np.array([p_0 - p_het / 2, p_het, p_1 - p_het / 2])

        priors = np.ones(3) / 3



        """
        goal: maximize P(R|f0,f1,theta) and use these to estimate prior
        """

        priors = np.array([f_0 - theta/2, theta, f_1 - theta/2])

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
        sample_depths[:, ii] = depths

    # Mark missing data
    missing = np.repeat(sample_depths == 0, 3, axis=1)
    genotype_probs[missing] = -1

    # Drop sites without coverage in any sample
    depth_sum = np.sum(sample_depths, axis=1)
    mask = depth_sum > 0
    genotype_probs = genotype_probs[mask]
    sites = np.where(mask)[0]

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


# -----------------------------------------------------------------------------
# Produce synthetic SAM files using an underlying biallelic substition model
# -----------------------------------------------------------------------------


# TODO handle multiple samples


def generate_sam_file(
    ts,
    path,
    ref_seq,
    ref_name=None,
    sample_name=None,
    chrom=None,
    ref_div=1e-4,
    depth=10,
    read_shape=10,
    read_scale=5,
    mean_base_qual=30,
    std_base_qual=10,
    mean_map_qual=60,
    std_map_qual=10,
    report=100000,
):
    """
    Generate a SAM file from a tree sequence using a simple sequencing read
    sampling model.

    ``ts`` must result from simulation with the 'binary' mutation model.

    Parameters
    ----------
    ref_seq : str, list, or array-like
        Sequence of nucleotide code characters to represent the reference
        genome. The ancestral sequence is sampled from ``ref_seq``.
    """
    # Check arguments
    if ref_name is None:
        ref_name = "ref"

    if sample_name is None:
        sample_name = "sample"

    if chrom is None:
        chrom = "chrom"

    # Set up reference, ancestral, and sampled sequences
    ref_seq = np.array([b for b in ref_seq])
    anc_seq = _sample_ancestral_sequence(ref_seq, ref_div)
    # TODO proper subsetting to desired sample.
    # Currently assume one diploid.
    raw_seq_strs = ts.as_fasta(wrap_width=0).split("\n")[1::2]
    raw_seq_arrs = [np.array([b if b != "N" else 0 for b in seq], dtype=np.int8)
                    for seq in raw_seq_strs]
    sample_seqs = [_sample_derived_sequence(seq, anc_seq)
                   for seq in raw_seq_arrs]

    # Calculate the approximate number of reads required to reach `depth`
    seq_len = len(ref_seq)
    mean_read_len = read_shape * read_scale
    n_reads = int(seq_len * depth / mean_read_len)

    # Define the columns to be filled
    sam_fields = [
        "QNAME",
        "FLAG",
        "RNAME",
        "POS",
        "MAPQ",
        "CIGAR",
        "RNEXT",
        "PNEXT",
        "TLEN",
        "SEQ",
        "QUAL",
        "RG",
        "NM",
    ]

    # Define the name of the readgroup
    readgroup = "SimReadGroup1"
    readgroup_symbol = f"RG:Z:{readgroup}"

    # Presample read positions and lengths
    read_lens = np.random.gamma(read_shape, scale=read_scale,
                                size=n_reads).astype(np.int64)
    read_lens[read_lens < 30] = 30
    read_starts = np.random.randint(0, seq_len - read_lens)
    read_seqs = np.random.randint(0, 2, size=n_reads)
    read_strands = np.random.choice([-1, 1], size=n_reads)

    map_scores = np.random.normal(mean_map_qual, std_map_qual,
                                  size=n_reads).astype(np.int64)
    map_scores[map_scores < 0] = 0
    map_scores[map_scores > 255] = 255

    # Sample base quality scores in blocks for efficiency
    def get_qual_block(size=1000000):
        qual_block = np.random.normal(mean_base_qual, std_base_qual,
                                      size=size).astype(np.int64)
        qual_block[qual_block < 0] = 0
        qual_block[qual_block > 40] = 40
        err_prob = utils._inverse_phred_function(qual_block)
        draws = np.random.rand(size)
        err_block = draws < err_prob
        return qual_block, err_block

    qual_block, err_block = get_qual_block()
    block_idx = 0

    coverage = 0

    # Write the output file
    with open(path, "w") as fout:
        # Write the header
        header = _get_sam_file_header(readgroup, sample_name, chrom, ref_name,
                                      seq_len)
        fout.write(header)

        for ii in range(n_reads):
            read_name = f"{sample_name}:{ii}"
            # Grab cached read characteristics
            read_start = read_starts[ii]
            read_len = read_lens[ii]
            read_end = read_start + read_len
            read_strand = read_strands[ii]
            seq_idx = read_seqs[ii]
            read_seq = sample_seqs[seq_idx][read_start:read_end]
            map_qual = map_scores[ii]

            # Get base quality scores and simulate sequencing error
            if block_idx + read_len > len(qual_block):
                qual_block, err_block = get_qual_block()
                block_idx = 0

            read_quals = qual_block[block_idx:block_idx + read_len]
            read_errs = err_block[block_idx:block_idx + read_len]
            block_idx += read_len
            base_qual_string = _get_base_qual_string(read_quals)
            read_seq = _sample_sequencing_errors(read_seq, read_errs)

            # Generate some other strings for output
            read_str = "".join(read_seq)
            cigar_str = _get_cigar_string(read_len)
            ref_dist = _get_ref_distance(read_seq, ref_seq[read_start:read_end])
            ref_dist_symbol = f"NM:i:{ref_dist}"

            flag = 0
            if read_strand == -1:
                flag += 16

            record = {
                "QNAME": read_name,
                "FLAG": flag,
                "RNAME": ref_name,
                "POS": read_start + 1,  # Switch to 1-indexing
                "MAPQ": map_qual,
                "CIGAR": cigar_str,
                "RNEXT": "*",
                "PNEXT": "0",
                "TLEN": read_strand * read_len,
                "SEQ": read_str,
                "QUAL": base_qual_string,
                "RG": readgroup_symbol,
                "NM": ref_dist_symbol
            }
            line = "\t".join([str(record[x]) for x in sam_fields]) + "\n"
            fout.write(line)

            coverage += read_len
            if report is not None and report > 0:
                if ii % report == 0:
                    depth_now = coverage / seq_len
                    print(timestamp(), f"Wrote read {ii}; depth {depth_now:3}")
    if report:
        depth_now = coverage / seq_len
        print(timestamp(), f"Wrote read {ii}; depth {depth_now:3}")
        print(timestamp(), f"Finished writing {path}")
    return


def _sample_ancestral_sequence(ref_seq, ref_div):
    """Generate a sequence diverged from ``ref_seq`` by ``ref_div``."""
    seq_len = len(ref_seq)
    # Copy `ref_seq`
    anc_seq = np.array(ref_seq)
    is_diff = np.random.choice([True, False], size=seq_len,
                               p=[ref_div, 1 - ref_div])
    for idx in np.where(is_diff)[0]:
        choices = [b for b in "ACTG" if b != ref_seq[idx]]
        anc_seq[idx] = np.random.choice(choices)
    return anc_seq


def _sample_derived_sequence(subs, anc_seq):
    """
    """
    derived_seq = np.array(anc_seq)
    is_sub = np.where(subs == 1)[0]
    for idx in is_sub:
        choices = [b for b in "ACTG" if b != anc_seq[idx]]
        derived_seq[idx] = np.random.choice(choices)
    return derived_seq


def _get_sam_file_header(
    readgroup,
    sample_name,
    chrom,
    ref_name,
    ref_len
):
    """
    Create a multi-line SAM file header and return it as a single string.

    Parameters
    ----------
    readgroup : str
    sample_name : str
    chrom : str
        May be of the form `chr1` or `1`.
    ref_name : str
        Name of the reference sequence, e.g. its filename.
    ref_len : int
        Length of the reference sequence.

    Returns
    -------
    header : str
        Header with file information. Rows are:
        @HD File-level metadata
        @SQ Reference sequence information
        @RG Readgroup information
        @PG Program information
    """
    sam_version = 1.6
    header_elems = {
        "@HD": {
            "VN": sam_version,
            "SO": "unsorted",
            },
        "@SQ": {
            "SN": ref_name,
            "LN": ref_len,
            },
        "@RG": {
            "ID": readgroup,
            "SM": sample_name,
            },
        "@PG": {
            "ID": "h2py",
            "PN": "h2py",
            },
        }
    header_rows = [
        row + "\t" + "\t".join([field + ":" + str(header_elems[row][field])
        for field in header_elems[row]]) for row in header_elems]
    header = "\n".join(header_rows) + "\n"
    return header


def _get_cigar_string(read_len):
    return f"{read_len}M"


def _get_ref_distance(read_seq, ref_segment):
    return np.sum(read_seq != ref_segment)


def _get_base_qual_string(scores):
    """Convert a list or array of `scores` to a Phred code string."""
    # Round scores down
    symbols = np.array(["!", '"', "#", "$", "%", "&", "'", "(", ")", "*",
                        "+", ",", "-", ".", "/", "0", "1", "2", "3", "4",
                        "5", "6", "7", "8", "9", ":", ";", "<", "=", ">",
                        "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I"])
    codes = "".join(symbols[scores])
    return codes


def _sample_sequencing_errors(read_seq, err_block):
    
    read_seq = np.array(read_seq)
    for idx in np.where(err_block)[0]:
        choices = [b for b in "ACTG" if b != read_seq[idx]]
        read_seq[idx] = np.random.choice(choices)
    return read_seq


# -----------------------------------------------------------------------------
# TODO read simulation with realistic nucleotide model... as before
# -----------------------------------------------------------------------------





