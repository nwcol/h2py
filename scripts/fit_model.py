"""
Fit a demographic model to bootstrapped H2 data.
"""

import argparse
import demes
import h2py
from h2py.utils import timestamp
import numpy as np
import pickle


def get_args():
    parser = argparse.ArgumentParser(
        description="Fit a demographic model to bootstrapped H2 data.")
    parser.add_argument("-d", "--data", required=True,
                        help="File with bootstrapped statistics (.pkl).")
    parser.add_argument("-g", "--graph", required=True,
                        help="Demes model file (.yaml).")
    parser.add_argument("-p", "--params", required=True,
                        help="Model options file (.yaml).")
    parser.add_argument("-o", "--output", requirerd=True,
                        help="Output fitted model file (.yaml)")
    # Optional arguments
    parser.add_argument("-u", type=float, default=1.5e-8,
                        help="Mutation rate parameter.")
    parser.add_argument("--max-iter", type=int, default=10000,
                        help="Maximum function calls allowed in each stage of"
                             "optimization.")
    parser.add_argument("--perturb", type=float, default=0,
                        help="If > 0, perturb initial parameters randomly/"
                             "uniformly by as much as `perturb`-fold.")
    parser.add_argument("--report", type=int, default=10,
                        help="If > 0, print optimization progress reports "
                             "every `report` function calls.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Load input data
    with open(args.data, "rb") as fin:
        data = pickle.load(fin)
    # Subset to population labels present in both data and model
    graph = demes.load(args.graph)
    deme_names = [d.name for d in graph0.demes]
    pops = [p for p in data["pops"] if p in deme_names]
    data = h2py.parsing.subset_data(data, to_pops=pops)
    # Set up invariant optimization arguments
    kwargs = {
        "pops": data["pops"],
        "r_bins": data["bins"],
        "u": args.u,
        "use_log": True,
        "max_iter": args.max_iter,
        "report": args.report,
        "output": args.output,
        "overwrite": True
    }
    # Optimize parameters in two stages, using scipy.optimize `LBFGSB` and
    # `Powell` algorithms in succession.
    h2py.inference.optimize(
        args.graph,
        args.params,
        data["means"],
        data["varcovs"],
        method="lbfgsb",
        perturb=args.perturb,
        **kwargs
    )
    *_, ll = h2py.inference.optimize(
        args.output,
        args.params,
        data["means"],
        data["varcovs"],
        method="powell",
        **kwargs
    )
    print(timestamp(),
          f"Finished fitting {args.graph}. Likelihood LL = {ll:.4}")

