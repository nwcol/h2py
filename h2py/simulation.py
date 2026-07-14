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
    seq_len=None,
    ref_div=1e-4,
    depth=10,
    p_err=0.001
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

    Returns
    -------
    sites : np.ndarray, shape (n_sites,)
    genotype_probs : np.ndarray, shape (n_sites, 3 * n_samples)
    """
    # Extract sequences from the tree sequence
    seq_strs = ts.as_fasta(wrap_width=0).split("\n")[1::2]
    sample_seqs = _get_sample_sequences(seq_strs, ref_div=ref_div)
    haplotypes = np.stack(sample_seqs, axis=1)
    n_samples = int(haplotypes.shape[1] / 2)
    seq_len = len(seq_strs[0])

    genotypes = np.zeros((seq_len, n_samples), dtype=np.int64)
    genotype_probs = np.zeros((seq_len, 3 * n_samples), dtype=np.float64)
    sample_depths = np.zeros((seq_len, n_samples))

    for ii in range(n_samples):
        sample = haplotypes[:, 2*ii:2*(ii+1)]

        # 'cheat' by calculating priors with true data
        genotypes_ii = np.sum(sample, axis=1)
        genotypes[:, ii] = genotypes_ii
        # priors = np.array([
        #     np.count_nonzero(genotypes == 0),
        #     np.count_nonzero(genotypes == 1),
        #     np.count_nonzero(genotypes == 2)]) / len(genotypes)

        # p_alt = np.sum(sample) / (2 * seq_len)
        # p_ref = 1 - p_alt
        # p_het = np.sum(sample[:, 0] != sample[:, 1]) / seq_len
        # priors = np.array([p_0 - p_het / 2, p_het, p_1 - p_het / 2])

        # priors = np.ones(3) / 3

        # priors = np.array([
        #     p_ref * (np.exp(-p_het) + p_ref * (1 - np.exp(-p_het))),
        #     2 * p_ref * p_alt * (1 - np.exp(-p_het)),
        #     p_alt * (np.exp(-p_het) + p_alt * (1 - np.exp(-p_het))),
        # ])

        # Sample coverage depth
        depths = np.random.poisson(depth, size=seq_len)
        # Draw allele samples
        f_alt = genotypes_ii / 2
        p_alt = f_alt + (1 - 2 * f_alt) * p_err
        n_alt = np.random.binomial(depths, p_alt)
        n_ref = depths - n_alt
        # Calculate genotype likelihoods
        genotype_liks = _compute_genotype_likelihood(n_ref, n_alt, p_err)

        # priors
        # p_1 = np.sum(genotypes) / (2 * seq_len)
        # p_0 = 1 - p_1
        # theta = np.sum(genotypes == 1) / seq_len
        # priors = np.array([p_0 * (1-theta) + p_0 ** 2 * theta,
        #                    2 * p_0 * p_1 * theta,
        #                    p_1 * (1-theta) + p_1 ** 2 * theta])

        # Weight genotype likelihoods by the priors
        raw_gps = genotype_liks
        norm = np.sum(raw_gps, axis=1)
        genotype_probs[:, 3*ii:3*(ii+1)] = raw_gps / norm[:, None]
        sample_depths[:, ii] = depths

    # Mark missing data
    missing = np.repeat(sample_depths == 0, 3, axis=1)
    genotype_probs[missing] = -1

    # Drop sites without coverage in any sample
    depth_sum = np.sum(sample_depths, axis=1)
    mask = depth_sum > 0
    genotypes = genotypes[mask]
    genotype_probs = genotype_probs[mask]
    sites = np.where(mask)[0]

    return sites, genotypes, genotype_probs


def _compute_genotype_likelihood(n_ref, n_alt, p_err):
    """
    Compute likelihoods for genotypes 0/0, 0/1, 1/1 from arrays of reference/
    alternate read counts and a fixed error rate.
    """
    return np.stack([(1 - p_err) ** n_ref * p_err ** n_alt,
                     0.5 ** (n_ref + n_alt),
                     p_err ** n_ref * (1 - p_err) ** n_alt], axis=1)


def _get_sample_sequences(fasta_strs, ref_div=1e-3):
    """
    fasta_strs looks like ['001NNN01NN', ...]
    """
    seq_len = len(fasta_strs[0])
    # Generate a reference sequence in ancestral/derived states
    ref_seq = np.random.choice([1, 0], size=seq_len, p=[ref_div, 1 - ref_div])
    seqs = []
    for fasta_str in fasta_strs:
        # Get the sample sequence as an array of ancestral/derived states
        sample_seq = np.array([b for b in fasta_str.replace("N", "0")],
                              dtype=np.int64)
        # Tranform into reference/alternate states by comparison to `ref_seq`
        _sample_seq = np.array([0 if sample_seq[i] == ref_seq[i] else 1
                                for i in range(seq_len)], dtype=np.int64)
        seqs.append(_sample_seq)

    return seqs


# -----------------------------------------------------------------------------
# Produce synthetic SAM files using an underlying biallelic substition model
# -----------------------------------------------------------------------------


def generate_sam_files(
    ts,
    samples,
    path_fmt,
    depth=10,
    chrom=None,
    ref_name=None,
    ref_seq=None,
    ref_div=1e-4,
    read_shape=10,
    read_scale=5,
    qual_mean=30,
    qual_std=5,
    map_mean=60,
    map_std=10,
    report=100000,
):
    """
    Generate SAM files for one or more samples using a simple read sequencing
    model.

    Expects ``samples`` to correspond in order and length to the number of
    samples in ``ts``.

    Parameters
    ----------
    path_fmt : str
        Output file path pattern: 'path/to/desired/output_{sample}.vcf'. Must
        include '{sample}'.
    """
    if chrom is None:
        chrom = "chr0"

    if ref_name is None:
        ref_name = "ref"

    # Generate a reference sequence if none was given
    seq_len = int(ts.sequence_length)
    if ref_seq is not None:
        assert len(ref_seq) == seq_len
        ref_seq = np.array([b for b in ref_seq])
    else:
        ref_seq = np.random.choice(["A", "T", "C", "G"], size=seq_len)

    # Generate an ancestral sequence with specified divergence to the reference
    anc_seq = _sample_ancestral_sequence(ref_seq, ref_div)

    # Split up sampled sequences into arrays and "polarize"
    n_samples = len(samples)
    raw_seqs = ts.as_fasta(wrap_width=0).split("\n")[1::2]
    assert len(raw_seqs) / 2 == n_samples
    sample_seqs = []
    for idx in range(n_samples):
        this_sample = [raw_seqs[2 * idx], raw_seqs[2 * idx + 1]]
        this_sample = [np.array([b if b != "N" else 0 for b in seq],
                                dtype=np.int64) for seq in this_sample]
        this_sample = [_construct_derived_sequence(seq, anc_seq)
                       for seq in this_sample]
        sample_seqs.append(this_sample)

    for ii, sample in enumerate(samples):
        path = path_fmt.format(sample=sample)
        generate_sam_file(
            sample_seqs[ii],
            path,
            ref_seq,
            ref_name=ref_name,
            sample_name=sample,
            chrom=chrom,
            depth=depth,
            read_shape=read_shape,
            read_scale=read_scale,
            qual_mean=qual_mean,
            qual_std=qual_std,
            map_mean=map_mean,
            map_std=map_std,
            report=report
        )

    return ref_seq


def generate_sam_file(
    sample_seqs,
    path,
    ref_seq,
    ref_name=None,
    sample_name=None,
    chrom=None,
    depth=10,
    read_shape=10,
    read_scale=5,
    qual_mean=30,
    qual_std=10,
    map_mean=60,
    map_std=10,
    report=100000,
):
    """
    Generate a SAM file from a tree sequence using a simple sequencing read
    sampling model.

    ``ts`` must result from simulation with the 'binary' mutation model.

    """
    # Check arguments
    if ref_name is None:
        ref_name = "ref"

    if sample_name is None:
        sample_name = "sample"

    if chrom is None:
        chrom = "chrom"

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

    map_scores = np.random.normal(map_mean, map_std,
                                  size=n_reads).astype(np.int64)
    map_scores[map_scores < 0] = 0
    map_scores[map_scores > 255] = 255

    # Sample base quality scores in blocks for efficiency
    def get_qual_block(size=1000000):
        qual_block = np.random.normal(qual_mean, qual_std,
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


def _construct_derived_sequence(subs, anc_seq):
    """
    """
    derived_seq = np.array(anc_seq, copy=True)
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





