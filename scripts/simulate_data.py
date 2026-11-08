"""
Simulate --n-reps genomic windows and compute H2 across them.
"""

import argparse
import demes
import h2py
from h2py.utils import timestamp
import io
import msprime
import numpy as np
import pickle


def get_args():
    parser = argparse.ArgumentParser(
        description="Simulate --n-reps genomic windows with uniform "
                    "recombination/mutation, and compute H2 across them.")
    parser.add_argument("-g", "--graph", required=True,
                        help="Demes model file (.yaml).")
    parser.add_argument("-p", "--pops", required=True,
                        help="Population file (whitespace-separated file with "
                             "one sample/population pair per line.")
    parser.add_argument("-o", "--out", required=True,
                        help="Output file for raw H2 statistics (.pkl).")
    # Optional arguments
    parser.add_argument("-u", type=float, default=1.5e-8,
                        help="Mutation rate (default 1.5e-8).")
    parser.add_argument("-r", type=float, default=1e-8,
                        help="Recombination rate (default 1e-8).")
    parser.add_argument("-L", type=float, default=1e6,
                        help="Window sequence length (default 1e6).")
    parser.add_argument("-n", "--n-reps", type=int, default=100,
                        help="Number of windows to simulate (default 100).")
    parser.add_argument("--bins", default=None,
                        help="Bin configuration, expressed as lower,upper,n "
                             "(default 1e-6,1e-2,16).")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Prep inputs
    pop_spec = h2py.matrices._read_pop_file(args.pops)
    sample_spec = {pop: len(pop_spec[pop]) for pop in pop_spec}
    samples = [s for pop in pop_spec for s in pop_spec[pop]]
    graph = demes.load(args.graph)
    demog = msprime.Demography.from_demes(graph)
    # Simulate sequence
    print(timestamp(), f"simulating {args.n_reps} {args.L} "
          f"bp windows under {args.graph}")
    print(timestamp(), f"sampled populations: {samples}")
    tree_seqs = msprime.sim_ancestry(
        sample_spec,
        demography=demog,
        sequence_length=args.L,
        num_replicates=args.n_reps,
        recombination_rate=args.r,
        record_provenance=False
    )
    # Set up bins
    if args.bins is None:
        bins_start, bins_end, n_bins = 1e-6, 1e-2, 16
    else:
        bin_spec = args.bins.split(",")
        bins_start, bins_end =float(bin_spec[0]), float(bin_spec[1])
        n_bins = int(bin_spec[2])
    bins = np.logspace(np.log10(bin_start), np.log10(bin_end), n_bins + 1)
    raw_sums = dict()
    # Iterate over tree sequences to simulate mutation and parse H2
    for ii, ts in enumerate(tree_seqs):
        mts = msprime.sim_mutations(ts, rate=args.u)
        data = io.StringIO(mts.as_vcf(contig_id=ii, individual_names=samples,
                           position_transform=lambda x: 1 + np.array(x)))
        raw_sums[ii] = h2py.parsing.compute_h2_stats(
            vcf_file=data,
            pop_file=args.pops,
            interval=[0, args.L],
            r=args.r,
            r_bins=r_bins,
            report=True,
        )
    boot = h2py.parsing.bootstrap_data(raw_sums)
    # Write to disk
    with open(args.out, "wb") as fout:
        pickle.dump(boot, fout)
    print(timestamp(), "Simulation and H2 estimation complete!")

