"""
Estimate H2 from a VCF file.

Example
-------
$ python estimate_h2.py -v variants.vcf -r recmap.txt --rec-map-col Map(cM) \
    --rec-pos-col Pos(bp) -b accessible.bed -L 2.45e8 --interval-size 5e6 \
    --bins 5e-6,5e-3,12 --pops pops.txt -o raw_h2.pkl
"""

import argparse
import h2py
import numpy as np
import pickle


def get_args():
    parser = argparse.ArgumentParser(
        description="Estimate H2 from a VCF file.")
    parser.add_argument("-v", "--vcf", required=True,
                        help="Path to VCF file (.vcf or .vcf.gz).")
    parser.add_argument("-p", "--pops", default=None,
                        help="Path to whitespace-separated population file; "
                             "each line should have a unique SAMPLE POP "
                             "pair. If not given, each VCF sample is assigned "
                             "to a population named after its sample ID.")
    parser.add_argument("-b", "--bed", default=None,
                        help="Path to BED file with accessible regions (.bed).")
    parser.add_argument("--chrom", required=True,
                        help="VCF CHROM to parse; used for labelling intervals.")
    parser.add_argument("-L", type=float, default=None,
                        help="Sequence length for configuring intervals. If "
                             "none is given, the whole chromosome is parsed "
                             "in one interval.")
    parser.add_argument("--interval-size", type=float, default=None,
                        help="Interval length (default none; required if -L "
                             "is given).")
    parser.add_argument("-r", "--rec", required=True,
                        help="Path to whitespace-separated recombination map "
                             "file OR uniform recombination rate, e.g. 1e-8.")
    parser.add_argument("--rec-map-col", default="Map(cM)",
                        help="Map column in recombination map file "
                             "(default 'Map(cM)').")
    parser.add_argument("--rec-pos-col", default="Position(bp)",
                        help="Position column in recombination map file "
                             "(default 'Position(bp)').")
    parser.add_argument("--r-bins", default="1e-6,1e-2,16",
                        help="Recombination bin specification, in format "
                             "'lower,upper,nbins', e.g. 1e-6,1e-2,16 (default).")
    parser.add_argument("--bp-bins", default=None,
                        help="Physical distance bin specification, in format "
                             "'lower,upper,nbins' (default none, overrides "
                             "--r-bins).")
    parser.add_argument("--min-bp", type=float, default=None,
                        help="Minimum physical distance between site pairs "
                             "counted in estimation (default none)")
    parser.add_argument("--max-bp", type=float, default=None,
                        help="Maximum physical distance between site pairs "
                             "counted in estimation (default none).")
    parser.add_argument("-u", "--mut", default=None,
                        help="Mutation map for local mutation rate adjustment "
                             "(.npy, .csv, .tsv, default none).")
    parser.add_argument("--u-bar", type=float, default=None,
                        help="Genome-wide mutation rate for normalizing "
                             "local mutation rate adjustment (default none).")
    parser.add_argument("--filtered", action="store_true",
                        help="Skip VCF rows without 'PASS' in the 'FILTER' "
                             "column.")
    parser.add_argument("--phased", action="store_true",
                        help="Treat sequences as phased.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output file (.pkl).")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Parse recombination map argument
    r = None
    rec_file = None
    try:
        r = float(args.rec)
    except:
        rec_file = args.rec

    # Set up bins. --bp-bins overrides the default --r-bins parameter.
    if args.bp_bins is not None:
        bin_spec = args.bp_bins.split(",")
        bins_start, bins_end = float(bin_spec[0]), float(bin_spec[1])
        n_bins = int(bin_spec[2])
        bp_bins = np.logspace(
            np.log10(bins_start), np.log10(bins_end), n_bins + 1)
    else:
        bin_spec = args.r_bins.split(",")
        bins_start, bins_end = float(bin_spec[0]), float(bin_spec[1])
        n_bins = int(bin_spec[2])
        r_bins = np.logspace(
            np.log10(bins_start), np.log10(bins_end), n_bins + 1)

    # Set up intervals
    if args.L is not None:
        assert args.interval_size is not None
        L = int(args.L)
        w = int(args.interval_size)
        intervals = [(i * w, (i + 1) * w) for i in range(int(L / w) + 1)]
    else:
        intervals = [None]

    # Loop over intervals and compute raw statistics in each.
    raw_sums = dict()
    for ii, interval in enumerate(intervals):
        if interval is not None:
            label = args.chrom + "_" + str(interval)
        else:
            label = args.chrom
        raw_sums[label] = h2py.parsing.compute_h2_stats(
            vcf_file=args.vcf,
            pop_file=args.pops,
            bed_file=args.bed,
            chromosome=args.chrom,
            interval=interval,
            rec_map_file=rec_file,
            rec_pos_col=args.rec_pos_col,
            rec_map_col=args.rec_map_col,
            r_bins=r_bins,
            bp_bins=bp_bins,
            r=r,
            min_bp=int(args.min_bp),
            max_bp=int(args.max_bp),
            mut_map_file=args.mut,
            u_bar=args.u_bar,
            use_haplotypes=args.phased,
            filtered=args.filtered
        )

    # Write output
    with open(args.output, "wb") as fout:
        pickle.dump(raw_sums, fout)

