"""
Simulate 500Mb of sequence under a two-population isolation with migration
model and fit five parameters to the resulting data.
"""

import demes
import msprime
import numpy as np
import os
import pickle

import h2py
from h2py.utils import timestamp


if not os.path.isdir("data/"):
    os.makedirs("data/")


# Simulation parameters
L = 5_000_000
n_reps = 100
u = 1.5e-8
r = 1e-8
r_bins = np.logspace(-6, -2, 17)


graph_file_content = """time_units: generations
demes:
- name: anc
  epochs:
  - {start_size: 1e4, end_time: 5e3}
- name: pop0
  ancestors: [anc]
  epochs:
  - {start_size: 2e4, end_time: 0}
- name: pop1
  ancestors: [anc]
  epochs:
  - {start_size: 5e3, end_time: 0}
migrations:
- {demes: [pop0, pop1], rate: 1e-4}
"""


options_file_content = """parameters:
- name: N_anc
  lower_bound: 10
  upper_bound: 1e5
  values:
    - demes:
        anc:
          epochs:
            0: start_size
- name: T_split
  lower_bound: 100
  upper_bound: 1e5
  values:
    - demes:
        anc:
          epochs:
            0: end_time
- name: N_pop0
  lower_bound: 10
  upper_bound: 1e5
  values:
    - demes:
        pop0:
          epochs:
            0: start_size
- name: N_pop1
  lower_bound: 10
  upper_bound: 1e5
  values:
    - demes:
        pop1:
          epochs:
            0: start_size
- name: m_0_1
  lower_bound: 1e-8
  upper_bound: 1e-2
  values:
    - migrations:
        0: rate
"""


samples = {"pop0": 1, "pop1": 1}
pop_file_content = """sample pop
tsk_0 pop0
tsk_1 pop1
"""
pops = list(samples.keys())


rec_map_file_content = f"""chrom\tPosition(bp)\tMap(cM)
0\t1\t0
0\t{L+1}\t{100*L*r}
"""


def run_msprime(graph, out_file):
    demog = msprime.Demography.from_demes(graph)
    ts = msprime.sim_ancestry(
        samples,
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
    )
    ts = msprime.sim_mutations(ts, rate=u)
    with open(out_file, "w+") as f:
        ts.write_vcf(f, position_transform = lambda x: 1 + np.array(x))
    return


def compute_stats(in_file):
    sums = h2py.parsing.compute_h2_stats(
        vcf_file=in_file,
        pop_file=pop_file,
        interval=[0, L],
        rec_map_file=rec_map_file,
        r_bins=r_bins,
        report=False,
    )
    print(timestamp(), f"Parsed {in_file}")
    return sums


if __name__ == "__main__":
    prefix = "data/inference_example"
    graph_file = f"{prefix}_graph.yaml"
    options_file = f"{prefix}_params.yaml"
    rec_map_file = f"{prefix}_map.txt"
    pop_file = f"{prefix}_pops.txt"

    with open(graph_file, "w") as f:
        f.write(graph_file_content)
    with open(options_file, "w") as f:
        f.write(options_file_content)
    with open(rec_map_file, "w") as f:
        f.write(rec_map_file_content)
    with open(pop_file, "w") as f:
        f.write(pop_file_content)

    stats_file = f"{prefix}_stats.pkl"
    # Only run simulations if the output file does not exit
    if True: #not os.path.isfile(stats_file):
        graph = demes.load(graph_file)
        out_files = [f"{prefix}_{i}.vcf" for i in range(n_reps)]
        for out_file in out_files:
            run_msprime(graph, out_file)
        sums = {x: compute_stats(x) for x in out_files}
        boot_data = h2py.parsing.bootstrap_data(sums)
        with open(stats_file, "wb") as f:
            pickle.dump(boot_data, f)
    else:
        with open(stats_file, "rb") as f:
            boot_data = pickle.load(f)

    #model = h2py.H2stats.from_demes(
    #    graph_file,
    #    sampled_demes=list(samples.keys()),
    #    u=u,
    #    r_bins=r_bins,
    #    phased=False
    #)
    #h2py.plotting.plot_h2_curves_comp(
    #    model,
    #    boot_data["means"],
    #    boot_data["varcovs"],
    #    r_bins=boot_data["bins"]
    #)

    fit_graph_file = f"{prefix}_inferred_graph.yaml"
    model_fit = h2py.inference.optimize(
        graph_file,
        options_file,
        boot_data["means"],
        boot_data["varcovs"],
        pops=list(samples.keys()),
        r_bins=r_bins,
        u=u,
        report=10,
        max_iter=100,
        output=fit_graph_file,
        overwrite=True,
        perturb=0.5,
    )

    # Plot fitted model
    fit_model = h2py.H2stats.from_demes(
        fit_graph_file,
        sampled_demes=list(samples.keys()),
        u=u,
        r_bins=r_bins,
        phased=False
    )
    h2py.plotting.plot_h2_curves_comp(
        fit_model,
        boot_data["means"],
        boot_data["varcovs"],
        r_bins=boot_data["bins"]
    )

    boot_reps = h2py.parsing.get_bootstrap_replicates(sums)

    uncertssss = h2py.uncerts.compute_uncerts(
        fit_graph_file,
        options_file,
        boot_data["means"],
        boot_data["varcovs"],
        boot_means=boot_reps,
        pops=pops,
        r_bins=r_bins,
        u=u,
        method="FIM",
    )

    uncertssss = h2py.uncerts.compute_uncerts(
        fit_graph_file,
        options_file,
        boot_data["means"],
        boot_data["varcovs"],
        boot_means=boot_reps,
        pops=pops,
        r_bins=r_bins,
        u=u,
        method="GIM",
    )




