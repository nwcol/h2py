"""
Simulation tools, primarily for generating genotype probabilities from
simulated sequences.
"""

import numpy as np
import random
import tskit


from . import utils
from .utils import timestamp


# -----------------------------------------------------------------------------
# Tools to create synthetic VCF files
# -----------------------------------------------------------------------------


def sample_mutations(anc_seq, ts):
    """
    """
    sampled_nodes = [x.id for x in ts.nodes() if x.individual >= 0]
    site_dict = dict()

    for mut in ts.mutations():
        node = mut.node
        site = mut.site
        pos = int(ts.sites_position[site])
        site_tree = ts.at(pos)
        #
        children = [n for n in site_tree.nodes(node) if n in sampled_nodes]
        # True if there is another already-handled mutation at this position
        if pos in site_dict:
            variants = site_dict[pos]
            anc_allele = variants[children[0]]
        else:
            anc_allele = anc_seq[pos]
            variants = [anc_allele for _ in sampled_nodes]
        # Sample derived allele
        choices = [b for b in "ACGT" if b != anc_allele]
        derived_allele = random.choice(choices)
        for child in children:
            variants[child] = derived_allele
        site_dict[pos] = variants
    return site_dict


def generate_sequence(anc_seq, site_dict, idx):
    """
    """
    new_seq = [b for b in anc_seq]
    for pos in site_dict:
        new_seq[pos] = site_dict[pos][idx]
    return "".join(new_seq)


def write_vcf(
    path,
    ref_seq,
    site_dict,
    sample_ids,
    chrom="0"
):
    """
    Write a phased VCF file (.vcf or .vcf.gz) describing simulated variation.

    # TODO: fixed differences from the reference!
    """
    seq_len = len(ref_seq)

    if path.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(path, "w") as fout:
        # write header
        header_lines = [
            "##fileformat=VCFv4.1\n",
            f"##contig=<ID={chrom},length={len(ref_seq)}>\n",
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n',
        ]
        cols = [
            "#CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO",
            "FORMAT"] + sample_ids
        header_lines.append("\t".join(cols) + "\n")
        for line in header_lines:
            fout.write(line)

        for pos in site_dict:
            # Note that `pos` is 0-indexed~
            ref = ref_seq[pos]
            variants = site_dict[pos]
            allele_set = set(variants)
            # Make sure `ref` is a the first element in this list
            alts = [a for a in allele_set if a != ref]
            alt = ",".join(alts)
            ordered_alleles = [ref] + alts
            codes = [ordered_alleles.index(a) for a in variants]
            genotypes = [f"{x}|{y}" for x, y in zip(codes[::2], codes[1::2])]
            line_elems = [
                chrom,
                str(pos + 1),
                ".",
                ref,
                alt,
                ".",
                ".",
                ".",
                "GT"] + genotypes
            line = "\t".join(line_elems) + "\n"
            fout.write(line)
    return


# -----------------------------------------------------------------------------
# Tools to simulate the sampling of sequencing reads
# -----------------------------------------------------------------------------


def generate_sam_file(
    path,
    ref_seq,
    sample_seqs,
    ref_name,
    sample_name,
    chrom=None,
    depth=10,
    read_model=None,
    map_model=None,
    qual_model=None,
    report=True,
):
    """
    read model:
    """
    ref_seq = np.array([b for b in ref_seq])
    sample_seqs = [np.array([b for b in sample_seqs[0]]),
                   np.array([b for b in sample_seqs[1]])]


    # set up models
    mean_read_len = 50

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

    for sample_seq in sample_seqs:
        assert len(sample_seq) == len(ref_seq)

    seq_len = len(ref_seq)
    n_reads = int((seq_len * depth) / mean_read_len)


    # Sample read positions and lengths
    read_lens = np.random.gamma(5, scale=10,  size=n_reads).astype(np.int64)
    read_lens[read_lens < 10] = 10
    read_starts = np.random.randint(0, seq_len, size=n_reads)
    read_seqs = np.random.randint(0, 2, size=n_reads)
    read_strands = np.random.choice([-1, 1], size=n_reads)

    map_scores = np.random.normal(60, 10, size=n_reads).astype(np.int64)
    map_scores[map_scores < 0] = 0
    map_scores[map_scores > 255] = 255

    # Sample quality scores in blocks for efficiency
    def get_qual_block():
        qual_block = np.random.normal(30, 10, size=1000000)
        qual_block[qual_block < 0] = 0
        qual_block[qual_block > 40] = 40
        return qual_block

    qual_block = get_qual_block()
    block_idx = 0

    coverage = 0

    # Write the output file
    with open(path, "w") as fout:
        # Write the header
        header = get_sam_header(readgroup, sample_name, chrom, ref_name, seq_len)
        fout.write(header)

        for ii in range(n_reads):
            qname = f"{sample_name}:{n_reads}"
            # Grab cached read characteristics
            read_len = read_lens[ii]
            tlen = read_len * read_strands[ii]
            read_start = read_starts[ii]
            pos = read_start + 1
            read_seq = sample_seqs[read_seqs[ii]][read_start:read_start+read_len]

            if block_idx + read_len > len(qual_block):
                qual_block = get_qual_block()
                block_idx = 0
            read_quals = qual_block[block_idx:block_idx + read_len]
            block_idx += read_len
            qual_codes = encode_quality_scores(read_quals)
            # err_read = sample_errors(read_seq, read_quals)
            read = "".join(read_seq)
            mapq = map_scores[ii]

            # Build CIGAR string using `ref`
            cigar = get_cigar_string(ref_seq, read_start, read)

            # Find flag
            flag = 0
            if tlen < 0:
                flag += 16

            # Find distance to the reference sequence
            ref_dist = get_ref_distance(ref_seq, read_start, read)
            ref_dist_symbol = f"NM:i:{ref_dist}"

            record = {
                "QNAME": qname,
                "FLAG": flag,
                "RNAME": ref_name,
                "POS": pos,
                "MAPQ": mapq,
                "CIGAR": cigar,
                "RNEXT": "*",
                "PNEXT": "0",
                "TLEN": tlen,
                "SEQ": read,
                "QUAL": qual_codes,
                "RG": readgroup_symbol,
                "NM": ref_dist_symbol
                }
            line = "\t".join([str(record[x]) for x in sam_fields]) + "\n"
            fout.write(line)

            coverage += read_len
            if report:
                if ii % 1000 == 0:
                    depth_now = coverage / seq_len
                    print(timestamp(), f"Wrote read {ii}; depth {depth_now:3}")
    if report:
        depth_now = coverage / seq_len
        print(timestamp(), f"Wrote read {ii}; depth {depth_now:3}")
        print(timestamp(), f"Finished writing {path}")
    return


def get_sam_header(
    readgroup,
    sample_name,
    chrom,
    ref_name,
    ref_len):
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
        Header with minimal information. Rows are:
        @HD File-level metadata
        @SQ Reference sequence information
        @RG Readgroup information
        @PG Profram information
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
            "ID": "sample_reads.py",
            "PN": "sample_reads.py",
            },
        }
    header_rows = [
        row + "\t" + "\t".join([field + ":" + str(header_elems[row][field])
        for field in header_elems[row]]) for row in header_elems]
    header = "\n".join(header_rows) + "\n"
    return header


def get_cigar_string(ref_seq, pos, read):

    return f"{len(read)}M"


def simulate_seq_error(read):
    """
    """
    # simulate read sequencing error
    quals = np.random.normal(mean, 5, len(read)).astype(int)
    quals[quals < 0] = 0
    quals[quals > 40] = 40

    err_probs = utils.inverse_phred_function(quals)
    has_err = np.random.rand(len(read)) < err_probs
    err_read = []
    for i in range(len(read)):
        if has_err[i]:
            base = read[i]
            other_bases = [x for x in ("ACTG") if x != base]
            err_base = np.random.choice(other_bases)
            err_read.append(err_base)
        else:
            err_read.append(read[i])
    err_read = np.array(err_read)
    return err_read, quals


def get_ref_distance(ref_seq, pos, read):
    """
    Find the edit distance between a reference and sample sequence.
    """
    ref_segment = ref_seq[pos:pos + len(read)]
    return np.sum(ref_segment != read)


def encode_quality_scores(scores):
    """Convert a list or array of `scores` to a Phred code string."""
    # Round scores down
    idx = np.asarray(scores, dtype=np.int64)
    symbols = np.array(["!", '"', "#", "$", "%", "&", "'", "(", ")", "*",
                        "+", ",", "-", ".", "/", "0", "1", "2", "3", "4",
                        "5", "6", "7", "8", "9", ":", ";", "<", "=", ">",
                        "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I"])
    codes = "".join(symbols[idx])
    return codes

