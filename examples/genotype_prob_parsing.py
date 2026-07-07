"""
simulates genotype probs in a slightly sloppy way
"""

import demes
import msprime
import numpy as np
import os

import h2py


if not os.path.isdir("data/"):
    os.makedirs("data/")

# Simulation parameters
L = 1_000_000
n_reps = 100
n_samples = 1  # per population
u = 1.5e-8
r = 1e-8
r_bins = np.logspace(-6, -2, 17)


# File names
pop_file = "data/populations.txt"
rec_map_file = "data/rec_map.txt"
bed_file = "data/coverage.bed"


def demographic_model():
    b = demes.Builder()
    b.add_deme("anc", epochs=[dict(start_size=1e4, end_time=3e3)])
    b.add_deme("pop0", ancestors=["anc"], epochs=[dict(start_size=1e4)])
    b.add_deme("pop1", ancestors=["anc"], epochs=[dict(start_size=1e4)])
    # b.add_migration(demes=["pop0", "pop1"], rate=1e-4)
    g = b.resolve()
    return g


def run_sim(g):
    demog = msprime.Demography.from_demes(g)
    samples = {"pop0": n_samples, "pop1": n_samples}
    ts = msprime.sim_ancestry(
        samples,
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
    )
    ts = msprime.sim_mutations(ts, rate=u, model="binary")
    gp = h2py.GenotypeProbMatrix.from_tree_sequence(ts, samples, depth=30)
    sums = h2py.parsing.compute_h2_stats(
        genotype_prob_matrix=gp,
        use_genotype_probs=True,
        pop_file=pop_file,
        bed_file=bed_file,
        rec_map_file=rec_map_file,
        r_bins=r_bins,
        report=True,
    )
    return sums


def write_pop_file():
    with open(pop_file, "w") as fout:
        fout.write("sample\tpop\n")
        for pidx in range(2):
            for sidx in range(n_samples):
                fout.write(f"tsk_{pidx * n_samples + sidx}\tpop{pidx}\n")


def write_rec_map_file():
    with open(rec_map_file, "w") as fout:
        fout.write("chrom\tPosition(bp)\tMap(cM)\n")
        fout.write("none\t0\t0\n")
        fout.write(f"none\t{L}\t{100*r*L}\n")


def write_bed_file():
    with open(bed_file, "w") as fout:
        fout.write(f"none\t0\t{L}\n")


if __name__ == "__main__":
    write_pop_file()
    write_bed_file()
    write_rec_map_file()
    g = demographic_model()
    sums = {i: run_sim(g) for i in range(n_reps)}
    boot_data = h2py.parsing.bootstrap_data(sums)
    model = h2py.H2stats.from_demes(g, sampled_demes=["pop0", "pop1"], u=u,
                                      r_bins=r_bins)
    h2py.plotting.plot_h2_curves_comp(
        model,
        boot_data["means"],
        boot_data["varcovs"],
        r_bins=boot_data["bins"])

    print(boot_data["means"][-1])
    print(model.data[-1])

