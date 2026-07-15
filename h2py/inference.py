"""
Infer parameters using maximum likelihood, after ``moments.Demes``.

Usage
-----

TODO
"""

import collections
import demes
import moments
from moments.Demes.Inference import (
    _get_demes_dict,
    _get_params_dict,
    _set_up_params_and_bounds,
    _set_up_constraints,
    _update_builder,
)
import numpy as np
import os
import pandas
import pickle
import random
import scipy

from . import utils, parsing, stats
from .stats import H2stats, _LDstats
from .utils import timestamp


OUT_OF_BOUNDS = 1e10
_counter = 0


# =============================================================================
# Principal inference function
# =============================================================================


def optimize(
    graph_file,
    options_file,
    means,
    varcovs,
    pops=None,
    r_bins=None,
    u=None,
    fit_u=False,
    phased=False,
    method="fmin",
    max_iter=1000,
    use_log=True,
    report=False,
    overwrite=False,
    output=None,
    perturb=False,
):
    """
    Fit the parameters of a demographic model to observed H2 using ML.

    Mostly mimics the functionality of ``moments.Demes.Inference.optimize_LD``.

    Parameters
    ----------
    graph_file : str, path
        Demes YAML file.
    options_file : str, path
        Moments.Demes-style parameter options file. TODO link to docs
    means : list
        Arrays with mean observed statistics.
    varcovs : list
        Covariance matrices generated with the bootstrap.
    pops : list
        List of population labels corresponding to data. Each element should
        match the name of a deme in ``graph_file``.
    r_bins : np.ndarray
        Array of recombination bin edges corresponding to data.
    u : float
        Mutation rate parameter.
    fit_u : bool, optional
        If True, fit the mutation rate as a free parameter.
    phased : bool, optional
        If True, compute phased expected H2. Should be used only when observed
        H2 was computed from phased data with the haplotype setting.
    method : str, optional
        SciPy optimization function to use. Must be "fmin", "powell", or
        "lbfgsb".
    max_iter : int, optional
        Maximum number of optimization iterations.
    use_log : bool, optional
        If True, optimize over the base-10 logarithm of parameters.
    report : int, optional
        If > 0, print status messages every ``report`` objective function calls.
    output : str, path, optional
        Save a graph with optimized parameters to this path.
    overwrite : bool, optional
        If True, overwrite any existing file at ``output``.
    perturb : float, optional
        If a positive float, perturb parameters by as much as ``|p * perturb|``
        from the initial guess given in ``graph_file``.

    Returns
    -------
    param_names : list
    params_opt : np.ndarray
    ll_opt : float
        Log-likelihood achieved in optimization.
    """
    # Reset function call counter
    global _counter
    _counter = 0

    # Check for required keyword arguments
    if pops is None:
        raise ValueError("pops is required")
    if r_bins is None:
        raise ValueError("r_bins is required")
    if u is None:
        raise ValueError("u is required")

    valid_methods = ["fmin", "powell", "lbfgsb"]
    if method not in valid_methods:
        raise ValueError("invalid method given")

    # Check for model/data consistency

    # Build the demography
    builder = _get_demes_dict(graph_file)
    options = _get_params_dict(options_file)
    param_names, params_0, lower_bounds, upper_bounds = \
        _set_up_params_and_bounds(options, builder)
    constraints = _set_up_constraints(options, param_names)

    # Set up mutation rate parameter
    if fit_u:
        param_names.append("u")
        params_0 = np.append(params_0, u)
        lower_bounds = np.append(lower_bounds, 1e-9)
        upper_bounds = np.append(upper_bounds, 1e-7)

    if perturb > 0:
        params_0 = _perturb_params(
            params_0,
            perturb,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            constraints=constraints
        )

    if use_log:
        params_0 = np.log10(params_0) + 1

    # Retrieve deme names and assign sample times
    deme_names = [d["name"] for d in builder["demes"]]
    sample_times = []
    for pop in pops:
        idx = deme_names.index(pop)
        end_time = builder["demes"][idx]["epochs"][-1]["end_time"]
        sample_times.append(end_time)
    sampled_demes = pops

    # Set up arguments
    args = (
        builder,
        options,
        means,
        varcovs,
        sampled_demes,
        sample_times,
        u,
        fit_u,
        r_bins,
        phased,
        use_log,
        lower_bounds,
        upper_bounds,
        constraints,
        report,
    )

    # Print a status message
    if report > 0:
        print(timestamp(), f"Fitting to observed H2 for {pops}")
        namestr = "".join([f"{n:>10}" for n in param_names])
        if use_log:
            _params_0 = 10 ** (params_0 - 1)
        else:
            _params_0 = params_0
        pstr = "".join([f"{float(p):>10.3}" for p in _params_0])
        print(f"{'Call':<5}{'LL':>10} [{namestr}]")
        print(f"{'init':<5}{'-':>10} [{pstr}]")

    # Call the selected scipy optimization function
    if method == "fmin":
        result = scipy.optimize.fmin(
            _objective_func,
            params_0,
            args=args,
            maxiter=max_iter,
            disp=False,
            full_output=True,
        )
        (params_opt, f_opt), flag = result[:2], result[4]

    elif method == "powell":
        result = scipy.optimize.fmin_powell(
            _objective_func,
            parasms_0,
            args=args,
            maxiter=max_iter,
            disp=False,
        )
        (params_opt, f_opt), flag = result[:2], result[5]

    elif method == "lbfgsb":
        if use_log:
            bounds = list(zip(np.log(lower_bounds) + 1,
                              np.log(upper_bounds) + 1))
            epsilon = 1e-3
        else:
            bounds = list(zip(lower_bounds, upper_bounds))
            epsilon = 1e-3
        result = scipy.optimize.fmin_l_bfgs_b(
            _objective_func,
            params_0,
            args=args,
            maxiter=max_iter,
            bounds=bounds,
            epsilon=epsilon,
            approx_grad=True,
            disp=False
        )
        params_opt, f_opt = result[:2]
        flag = result[2]["warnflag"]

    ll_opt = -f_opt
    params_opt = 10 ** (params_opt - 1) if use_log else params_opt

    if report > 0:
        print(f"Finished with flag {flag}")
        print(f"Log-likelihood:\t{ll_opt:.3}")
        print("Fitted parameters:")
        print("    Param\tMLE")
        for name, value in zip(param_names, params_opt):
            print(f"    {name}\t{value:.3}")

    if output is not None:
        builder = _update_builder(builder, options, params_opt)
        graph = demes.Graph.fromdict(builder)
        # Save information about optimization in the output file
        opt_info = {
            "ll": ll_opt,
            "flag": flag,
            "u": u,
        }
        graph.metadata["opt_info"] = opt_info
        if overwrite is False and os.path.isfile(output):
            print(f"{output} already exists; printing model")
            print(str(graph))
        else:
            demes.dump(graph, output)

    return param_names, params_opt, ll_opt


# -----------------------------------------------------------------------------
# Functions subordinate to ``optimize``
# -----------------------------------------------------------------------------


def _objective_func(
    params,
    builder,
    options,
    means,
    varcovs,
    sampled_demes,
    sample_times=None,
    u=None,
    fit_u=False,
    r_bins=None,
    phased=False,
    use_log=False,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    report=0,
):
    """Objective function for parameter optimization."""
    global _counter
    _counter += 1

    if use_log:
        params = 10 ** (params - 1)

    if lower_bounds is not None and np.any(params < lower_bounds):
        return OUT_OF_BOUNDS
    if upper_bounds is not None and np.any(params > upper_bounds):
        return OUT_OF_BOUNDS
    if constraints is not None and np.any(constraints(params) <= 0):
        return OUT_OF_BOUNDS

    if fit_u:
        u = params[-1]

    builder = _update_builder(builder, options, params)
    graph = demes.Graph.fromdict(builder)
    model = H2stats.from_demes(
        graph,
        sampled_demes=sampled_demes,
        sample_times=sample_times,
        r_bins=r_bins,
        u=u,
        phased=phased,
        method="simpsons",
    )
    ll = _compute_composite_ll(model, means, varcovs)

    if report and _counter % report == 0:
        pstr = "".join([f'{float(p):>10.3}' for p in params])
        print(f"{_counter:<5}{np.round(ll, 2):>10} [{pstr}]")

    return -ll


def _perturb_params(
    params,
    fold,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    max_tries=100
):
    """Randomly/uniformly perturb params on ``[p*(1-fold), p*(1+fold)]``."""
    valid = False
    n_tries = 0
    while not valid:
        if n_tries > max_tries:
            raise ValueError("failed to perturb params with bounds/constraints")
        n_tries += 1
        draw = np.random.uniform(params * (1 - fold), params * (1 + fold))
        if np.any(draw <= lower_bounds) or np.any(draw >= upper_bounds):
            for ii in range(len(draw)):
                n_redraws = 0
                while (draw[ii] <= lower_bounds[ii]
                       or draw[ii] >= upper_bounds[ii]):
                    if n_redraws > max_tries:
                        raise ValueError(
                            "failed to perturb parameters within bounds")
                    draw[ii] = np.random.unform(params[ii] * (1 - fold),
                                                params[ii] * (1 + fold))
                    n_redraws += 1
        if constraints is not None:
            if np.all(constraints(draw) > 0):
                valid = True
        else:
            valid = True
    return draw


# -----------------------------------------------------------------------------
# Likelihood functions
# -----------------------------------------------------------------------------


_inv_varcov_cache = dict()


def _compute_composite_ll(model, means, varcovs):
    """Compute the sum of bin log-likelihoods."""
    return _compute_bin_ll(model.h2(), means[:-1], varcovs[:-1]).sum()


def _compute_bin_ll(xs, mus, varcovs):
    """
    Compute LL in each bin and return an array of bin LLs.
    """
    n_bins = len(xs)
    if len(mus) != n_bins or len(varcovs) != n_bins:
        raise ValueError("model, means, varcovs must have same length")
    result = np.zeros(n_bins, dtype=np.float64)
    for ii, (x, mu, varcov) in enumerate(zip(xs, mus, varcovs)):
        key = str(varcov)
        if key in _inv_varcov_cache:
            inv_varcov = _inv_varcov_cache[key]
        else:
            inv_varcov = np.linalg.inv(varcovs[ii])
            _inv_varcov_cache[key] = inv_varcov
        result[ii] = _compute_ll(x, mu, inv_varcov)
    return result


def _compute_ll(x, mu, inv_varcov):
    """
    Compute the log of the gaussian law evaluated at ``x`` with a pre-inverted
    var/cov matrix. Drops the constant coefficient.
    """
    return -0.5 * np.matmul(np.matmul((x - mu).T, inv_varcov), x - mu)


def _compute_exact_bin_ll(xs, mus, varcovs):
    """Compute the log-likelihood in bins, without dropping the coefficient."""
    return np.array([scipy.stats.multivariate_normal(mean=m, cov=c).logpdf(x)
                     for x, m, c in zip(xs, mus, varcovs)])

