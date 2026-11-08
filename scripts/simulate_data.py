"""
Simulate --n-reps genomic windows and compute H_2 from them.
"""

import argparse
import demes
import h2py
from h2py.utils import timestamp
import moments
import msprime
import numpy as np
import pandas
import pickle
import sys


def get_args():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("-g", "--graph", required=True)
    parser.add_argument("-p", "--pops", required=True)
    parser.add_argument("-o", "--out", required=True)
    # Optional
    parser.add_argument("-u", type=float, default=1.3e-8)
    parser.add_argument("-r", type=float, default=1e-8)
    parser.add_argument("-L", type=float, default=1e6)
    parser.add_argument("-n", "--n-reps", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Prep inputs
    pop_spec = h2py.matrices._read_pop_file(args.pops)
    samples = {pop: args.n_samples for pop in pop_spec}
    graph = demes.load(args.graph)
    demog = msprime.Demography.from_demes(graph)
    # Simulate sequence
    print(timestamp(), f"simulating {args.n_reps} {args.L} bp windows under {args.graph}")
    print(timestamp(), f"sampled populations: {samples}")
    tree_seqs = msprime.sim_ancestry(
        samples,
        demography=demog,
        sequence_length=args.L,
        num_replicates=args.n_reps,
        recombination_rate=args.r,
        record_provenance=False
    )
    filenames = []
    for ii, ts in enumerate(tree_seqs):
        mts = msprime.sim_mutations(ts, rate=args.u)
        filename = f"sim_window_{ii}.vcf"
        with open(filename, "w+") as fout:
            mts.write_vcf(
                fout,
                contig_id=ii,
                position_transform=lambda x: 1 + np.array(x),
                individual_names=list(samples.keys())
            )
        filenames.append(filename)
    # Compute H_2 from sequence.
    r_bins = np.logspace(-6, -2, 17)
    raw = dict()
    for ii, filename in enumerate(filenames):
        raw[ii] = h2py.parsing.compute_h2_stats(
            vcf_file=filename,
            pop_file=args.pops,
            interval=[0, args.L],
            r=args.r,
            r_bins=r_bins,
            report=True,
        )
    boot = h2py.parsing.bootstrap_data(raw)
    # Write to disk
    with open(args.out, "wb") as fout:
        pickle.dump(boot, fout)
    print(timestamp(), "genome simulation (H_2) complete!")

