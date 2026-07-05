"""
Simulation tools, primarily for generating genotype probabilities from
simulated sequences.
"""

import numpy as np
import tskit


# -----------------------------------------------------------------------------
# Tools to create synthetic VCF files
# -----------------------------------------------------------------------------


def sample_mutations(anc_seq, ts):
    """
    """
    sampled_nodes = [x.id for x in ts.nodes() if x.individual >= 0]
    working = dict()

    for mut in ts.mutations():
        node = mut.node()
        site = mut.site()
        pos = int(ts.sites_position[site])
        site_tree = ts.at(pos)
        #
        children = [n for n in site_tree.nodes(n) if n in sampled_nodes]
        # True if there is another already-handled mutation at this position
        if pos in working:
            alleles = working[pos]
            variants = alleles[children[0]]
        else:
            anc_allele = anc_seq[pos]
            variants = [anc_allele for _ in sampled_nodes]
        # Sample derived allele
        choices = [b for b in "ACGT" if b != anc_allele]
        derived_allele = np.random.choice(choices)
        for child in children:
            variants[child] = derived_allele
        working[pos] = site_variants

    # Get array of strings
    haplotypes = [[working[x] for x in working.keys()]]
    sites = np.array([x for x in working.keys()], dtype=np.int64)
    return sites, haplotypes


def write_vcf(fname, ref_seq, anc_seq, variants, sample_ids, chrom="0"):
    """
    Write a phased VCF file (.vcf or .vcf.gz) describing simulated variation.

    Parameters
    ----------
    fname : str
        Output filename.
    ref_seq : str
        String defining the reference sequence.
    anc_seq : str
    variants : dict
        A dictionary that maps positions (0-indexed) to sorted allelic states.
    sample_ids : list of str
        Sample identifiers.
    chrom : str
        Chromosome identifier.
    """
    seq_len = len(ref_seq)

    if fname.endswith(".gz"):
        open_func = gzip.open
    else:
        open_func = open
    with open_func(fname, "w") as fout:
        # write header
        header_elems = [
            "#CHROM",
            "POS",
            "ID",
            "REF",
            "ALT",
            "QUAL",
            "FILTER",
            "INFO",
            "FORMAT"] + sample_ids
        header = "\t".join(header_elems) + "\n"
        fout.write(header)

        for pos0 in range(seq_len):
            if pos0 in variants:
                pos1 = pos0 + 1
                ref = ref_seq[pos0]
                site_variants = variants[pos0]
                unique_alleles = set(site_variants)
                alt_alleles = [a for a in unique_alleles if a != ref]
                alt = ",".join(alt_alleles)
                all_alleles = [ref] + alt_alleles
                codes = [all_alleles.index(a) for a in site_variants]
            else:
                if ref_seq[pos0] != anc_seq[pos0]:
                    pos1 = pos0 + 1
                    ref = ref_seq[pos0]
                    alt = anc_seq[pos0]
                    codes = [1 for _ in range(2 * len(sample_ids))]
                else:
                    continue
            genotypes = [
                f"{x}|{y}" for x, y in zip(codes[::2], codes[1::2])]
            line_elems = [
                chrom,
                str(pos1),
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


def sample_sequencing_reads(
    ref_fname,
    sample_fname,
    out_fname,
    depth=10,
    read_model=None,
    map_model=None,
    qual_model=None,
    pmd_model=None,
    paired=False,
    verbose=0):
    """
    Simulate sequencing reads from a diploid sequence and save them in a .sam
    file.

    Parameters
    ----------
    ref_fname : str
        Expected header format: ">chrN", where N is the chromosome number.

    sample_fname : str
        Expected header formats: ">samplename:chrN:seqI", where samplename is
        the label assigned to the sample, N is the chromosome number, and I
        specifies the homolog number (0 or 1).

    out_fname : str

    Returns
    -------
    None
    """
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

    # Set up distributions
    read_distr = set_up_distribution(read_model)
    map_distr = set_up_distribution(map_model)
    qual_distr = set_up_distribution(qual_model)

    # Load data
    ref_seqs, chroms = lib.read_fasta(ref_fname)
    ref_seq = ref_seqs[0]
    chrom = chroms[0]
    ref_name = chrom
    ref_len = len(ref_seq)
    sample_seqs, labels = lib.read_fasta(sample_fname)
    sample_name = labels[0].split(":")[0]

    # ref_seq = np.array([x for x in ref_seq])
    # seqs = [np.array([x for x in seq]) for seq in seqs]

    for sample_seq in sample_seqs:
        assert len(sample_seq) == len(ref_seq)

    # Initialize array for counting
    coverage = np.zeros(len(ref_seq))
    n_reads = 0

    with open(out_fname, "w") as fout:
        # Write the header
        header = get_header(
            readgroup,
            sample_name,
            chrom,
            ref_name,
            ref_len)
        fout.write(header)

        while np.mean(coverage) < depth:
            qname = f"{sample_name}:{n_reads}"

            # Sample one homolog
            sample_seq = sample_seqs[np.random.randint(2)]

            # Sample a read
            pos0, raw_read, strand = sample_read(sample_seq, read_distr)
            pos = pos0 + 1
            if strand == 1:
                tlen = len(raw_read)
            else:
                tlen = -len(raw_read)


            # Simulate sequencing error
            err_read, quals = simulate_seq_error(raw_read, qual_distr)
            read = "".join(err_read)

            # Get quality scores for each base
            qual_codes = encode_quality_scores(quals)

            # Sample map quality
            mapq = map_distr(1)[0]

            # Build CIGAR string using `ref`
            cigar = get_cigar_string(ref_seq, pos0, read)

            # Find flag
            flag = 0
            if tlen < 0:
                flag += 16

            # Find distance to the reference sequence
            ref_dist = get_ref_distance(ref_seq, pos0, read)
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

            # Increment coverage array
            coverage[pos:pos + len(read)] += 1
            n_reads += 1

            if verbose:
                if n_reads % verbose == 0:
                    print(f"sampled read {n_reads}; coverage {np.mean(coverage)}")
    return








def get_header(
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


def simulate_pmd(read):
    """
    """
    # Simulate post-mortem damage to the DNA molecule
    return read


def get_cigar_string(ref_seq, pos, read):

    return f"{len(read)}M"

def simulate_seq_error(read, distr_func):
    """
    """
    # simulate read sequencing error
    #quals = np.random.normal(mean, 5, len(read)).astype(int)
    #quals[quals < 0] = 0
    #quals[quals > 40] = 40

    # Sample quality scores
    quals = distr_func(len(read))
    err_probs = inv_phred_func(quals)
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


def get_ref_distance(ref_seq, pos0, read):
    """
    Find the edit distance between a reference and sample sequence.
    """
    ref_segment = ref_seq[pos0:pos0 + len(read)]
    return sum([x != y for x, y in zip(ref_segment, read)])




def sample_read(sample_seq, distr_func):
    """
    Sample a read with a gamma-distributed length and returns the position of
    the read and its sequence.

    At present, samples only from the positive strand. This may change in the
    future.
    """
    read_len = int(distr_func(1)[0])
    strand = 1
    pos0 = np.random.randint(0, len(sample_seq) - read_len + 1)
    start = pos0
    end = pos0 + read_len
    read = sample_seq[start:end]
    return pos0, read, strand




def get_reverse_complement(seq):
    """
    Take the reverse complement of a sequence.

    Parameters
    ----------
    seq : string, np.ndarray, or list
        Sequence to reverse-complement.

    Returns
    -------
    reverse_complement : string
        Reverse complement of `seq`.
    """
    mapping = {
        "A": "T",
        "T": "A",
        "G": "C",
        "C": "G"}
    reverse_complement = "".join([mapping[x] for x in seq[::-1]])
    return reverse_complement


def encode_quality_scores(scores):
    """Convert a list or array of `scores` to a Phred code string."""
    # Round scores down
    scores = [int(x) for x in scores]
    mapping = {
        0: "!",
        1: '"',
        2: "#",
        3: "$",
        4: "%",
        5: "&",
        6: "'",
        7: "(",
        8: ")",
        9: "*",
        10: "+",
        11: ",",
        12: "-",
        13: ".",
        14: "/",
        15: "0",
        16: "1",
        17: "2",
        18: "3",
        19: "4",
        20: "5",
        21: "6",
        22: "7",
        23: "8",
        24: "9",
        25: ":",
        26: ";",
        27: "<",
        28: "=",
        29: ">",
        30: "?",
        31: "@",
        32: "A",
        33: "B",
        34: "C",
        35: "D",
        36: "E",
        37: "F",
        38: "G",
        39: "H",
        40: "I"}
    codes = "".join([mapping[x] for x in scores])
    return codes



