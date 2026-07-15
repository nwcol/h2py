"""
Apply an LRT adjustment to an IM model.
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
L = 1_000_000
n_reps = 200
u = 1.5e-8
r = 1e-8
r_bins = np.logspace(-6, -2, 17)


null_graph_file = "data/IM_null_model.yaml"
null_graph_file_content = """time_units: generations
demes:
- name: anc
  epochs:
  - {start_size: 1e4, end_time: 8e3}
- name: pop0
  ancestors: [anc]
  epochs:
  - {start_size: 2e4, end_time: 0}
- name: pop1
  ancestors: [anc]
  epochs:
  - {start_size: 5e3, end_time: 0}
"""


graph_file = "data/IM_model.yaml"
graph_file_content = """time_units: generations
demes:
- name: anc
  epochs:
  - {start_size: 1e4, end_time: 8e3}
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


null_options_file = "data/IM_null_model_options.yaml"
null_options_file_content = """parameters:
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
"""


options_file = "data/IM_model_options.yaml"
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
pop_file = "data/IM_populations.txt"
pop_file_content = """sample pop
tsk_0 pop0
tsk_1 pop1
"""
pops = list(samples.keys())


rec_map_file = "data/rec_map_1Mb_1cM.txt"
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
    # Write files
    with open(null_graph_file, "w") as f:
        f.write(null_graph_file_content)
    with open(null_options_file, "w") as f:
        f.write(null_options_file_content)

    with open(graph_file, "w") as f:
        f.write(graph_file_content)
    with open(options_file, "w") as f:
        f.write(options_file_content)

    with open(rec_map_file, "w") as f:
        f.write(rec_map_file_content)
    with open(pop_file, "w") as f:
        f.write(pop_file_content)

    # Run simulations
    graph = demes.load(graph_file)
    out_files = [f"data/IM_sim_{i}.vcf" for i in range(n_reps)]
    for out_file in out_files:
        run_msprime(graph, out_file)

    # Compute statistics and run the bootstrap
    sums = {x: compute_stats(x) for x in out_files}
    boot_data = h2py.parsing.bootstrap_data(sums)
    boot_means = h2py.parsing.get_bootstrap_replicates(sums)

    mle_graph_file = "data/IM_full_MLE.yaml"
    mle_null_graph_file = "data/IM_null_MLE.yaml"

    # Fit the null model
    ll_null = h2py.inference.optimize(
        null_graph_file,
        null_options_file,
        boot_data["means"],
        boot_data["varcovs"],
        pops=pops,
        r_bins=r_bins,
        u=u,
        report=30,
        max_iter=100,
        output=mle_null_graph_file,
        overwrite=True,
        perturb=0.5,
    )[-1]
    fit_model = h2py.H2stats.from_demes(
        mle_null_graph_file,
        sampled_demes=pops,
        u=u,
        r_bins=r_bins,
    )
    h2py.plotting.plot_h2_curves_comp(
        fit_model,
        boot_data["means"],
        boot_data["varcovs"],
        r_bins=boot_data["bins"]
    )

    # Fit the full model
    ll_full = h2py.inference.optimize(
        graph_file,
        options_file,
        boot_data["means"],
        boot_data["varcovs"],
        pops=pops,
        r_bins=r_bins,
        u=u,
        report=30,
        max_iter=500,
        output=mle_graph_file,
        overwrite=True,
        perturb=0.1,
    )[-1]
    fit_model = h2py.H2stats.from_demes(
        mle_graph_file,
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

    # Calculate the LRT adjustment
    fac = h2py.uncerts.compute_lrt_adjustment(
        null_graph_file,
        graph_file,
        options_file,
        boot_data["means"],
        boot_data["varcovs"],
        boot_means,
        nested_params=["m_0_1"],
        nested_values=[0.0],
        pops=pops,
        r_bins=r_bins,
        u=u,
    )
    score = 2 * (ll_full - ll_null)
    print(f"Naive LRT score: {score:.3}")
    print(f"Adjustment factor: {fac:.3}")
    print(f"Adjusted LRT score: {score * fac:.3}")





