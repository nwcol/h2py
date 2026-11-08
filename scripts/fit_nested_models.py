"""
Fit nested null/alternative models; initiate alternative model parameters at
the best-fit values of the null model.

Use the --nested argument to provide initial values for alternative model
parameters that are not present in the null model. For example, if the
alternative model includes an instantaneous admixture pulse, with time
parameter Tpulse and proportion Fpulse, we can set initial values with
    --nested Tpulse 1e5 Fpulse 0.1
"""

import argparse
import demes
import h2py
from h2py.utils import timestamp
import moments
from moments.Demes.Inference import (
    _update_builder,
    _get_demes_dict,
    _get_params_dict,
    _set_up_params_and_bounds,
)
import numpy as np
import pickle


def get_args():
    parser = argparse.ArgumentParser(
        description="Fit nested null/alternative models; initiate alternative "
                    "model parameters at best-fit values of the null model.")
    parser.add_argument("-d", "--data", required=True,
                        help="File with bootstrapped statistics (.pkl).")
    parser.add_argument("-g0", "--graph0", required=True,
                        help="Demes file with null model (.yaml).")
    parser.add_argument("-p0", "--params0", required=True,
                        help="Options file for null model (.yaml).")
    parser.add_argument("-g1", "--graph1", required=True,
                        help="Demes file with alternative model (.yaml).")
    parser.add_argument("-p1", "--params1", required=True,
                        help="Options file for alternative model (.yaml).")
    parser.add_argument("--nested", nargs="*",
                        help="Nested parameter-value pairs (see docstring).")
    parser.add_argument("-o", "--outpref", required=True,
                        help="Prefix for output files: {outpref}_0.yaml, "
                             "{outpref}_1.yaml")
    # Optional arguments
    parser.add_argument("-u", type=float, default=1.5e-8,
                        help="Mutation rate parameter.")
    parser.add_argument("--max-iter", type=int, default=10000,
                        help="Maximum function calls allowed in each stage of"
                             "optimization.")
    parser.add_argument("--perturb", type=float, default=0,
                        help="If > 0, perturb initial parameters for the "
                             "null model randomly/uniformly by as much as"
                              "`perturb`-fold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Load input data
    with open(args.data, "rb") as fin:
        data = pickle.load(fin)
    # Subset to population labels present in both data and model
    graph0 = demes.load(args.graph0)
    deme_names = [d.name for d in graph0.demes]
    pops = [p for p in data["pops"] if p in deme_names]
    data = h2py.parsing.subset_data(data, to_pops=pops)

    # Set up a function for fitting models
    def fit_model(graph, params, perturb, out):
        # Invariant arguments
        kwargs = {
            "pops": data["pops"],
            "r_bins": data["bins"],
            "u": args.u,
            "use_log": True,
            "max_iter": args.max_iter,
            "report": 100,
            "output": out,
            "overwrite": True
        }
        # Stage 1: LBFGSB
        pnames, ps, ll = h2py.inference.optimize(
            graph,
            params,
            data["means"],
            data["varcovs"],
            method="lbfgsb",
            perturb=perturb,
            **kwargs
        )
        # Stage 2: powell
        pnames, ps, ll = h2py.inference.optimize(
            graph,
            params,
            data["means"],
            data["varcovs"],
            method="powell",
            **kwargs
        )
        return pnames, ps, -ll

    # Fit the null model to convergence
    pnames0, ps0, ll0 = fit_model(args.graph0, args.params0, args.perturb,
                                  f"{args.outpref}_0.yaml")
    print(timestamp(), f"Finished fitting model {args.graph0}!")
    # Get default values for parameters absent from the null model
    nest_params = {key: float(val)
                   for key, val in zip(args.nested[::2], args.nested[1::2])}
    print(timestamp(), f"nesting: {nest_params}")
    builder1 = _get_demes_dict(args.graph1)
    options1 = _get_params_dict(args.params1)
    pnames1, ps1 = _set_up_params_and_bounds(options1, builder1)[:2]
    # Set up the alternative model at MLE parameters of the null model
    ps1_work = []
    for pname1 in pnames1:
        if pname1 in pnames0:
            idx = pnames0.index(pname1)
            ps1_work.append(ps0[idx])
        else:
            assert pname1 in nest_params
            val = nest_params[pname1]
            ps1_work.append(val)
    builder1 = _update_builder(builder1, options1, ps1_work)
    work_graph1 = demes.Graph.fromdict(builder1)
    # Write the constructed alternative model to disk
    model1_start = f"{args.outpref}_1_init.yaml"
    demes.dump(work_graph1, model1_start)
    # Fit the null model to convergence
    perturb1 = 0
    pnames1, pvals1, ll1 = fit_model(model1_start, args.params1, perturb1,
                                     f"{args.outpref}_1.yaml")
    print(timestamp(), f"Finished fitting model {args.graph1}!")
    print(timestamp(), f"Likelihood difference (LL1 - LL0) = {ll1 - ll0:.4}")

