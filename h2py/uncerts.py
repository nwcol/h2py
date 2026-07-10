"""
Compute uncertainty in expected parameters.
"""

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
import scipy as sp

from . import inference
from . import H2stats


def compute_uncerts(
    method="GIM",
):
    """
    Compute confidence intervals for MLE parameters.
    """
    return


def compute_lrt_adjustment(
):
    """
    Compute an adjustment factor for the LRT test statistic.
    """
    return


def _get_godambe_matrix():
    return


def _get_fisher_matrix():
    return


def _get_variability_matrix():

    return


def _get_hessian_matrix():
    return


def _hessian_elem():
    return


def _get_score(
    params,
    args,
    means,
    varcovs,
    delta=0.01,
    steps=None,
    bounds=None,
):
    """
    Compute the score, the gradient of the LL function, using finite
    differences.
    """
    # Set up bounds if non are given
    if bounds is None:
        bounds = (np.zeros(len(params)), np.full(len(params), np.inf))

    # Check bounds
    if np.any(params < bounds[0]) or np.any(params > bounds[1]):
        raise ValueError("initial parameters violate bounds")

    if steps is None:
        steps = delta * params
        if np.any(steps == 0):
            steps[steps == 0] = delta

    for ii in range(len(params)):
        if np.any((params - steps < bounds[0]) & (params + steps > bounds[1])):
            raise ValueError("bounds prevent finite differences evaluation")

    lower, upper = bounds
    full_args = (means, varcovs, args)
    score = np.zeros(len(params, dtype=np.float64)

    for ii in range(len(params)):
        e_i = _get_e_i(len(params), ii)

        # Determine which points to evaluate
        if params[ii] == 0:
            form = "forward"
        elif params[ii] - steps[ii] < lower[ii]:
            form = "forward"
        elif params[ii] + steps[ii] > upper[ii]:
            form = "backward"
        else:
            form = "central"

        if form == "forward":
            f_0 = _evaluate_ll(params, *full_args)
            f_f = _evaluate_ll(params + steps * e_i, *full_args)
            score[ii] = (f_f - f_0) / steps[ii]

        elif form == "backward":
            f_b = _evaluate_ll(params - steps * e_i, *full_args)
            f_0 = _evaluate_ll(params, *full_args)
            score[ii] = (f_0 - f_b) / steps[ii]

        else:
            f_b = _evaluate_ll(params - steps * e_i, *full_args)
            f_f = _evaluate_ll(params + steps * e_i, *full_args)
            score[ii] = (f_f - f_b) / (2 * steps[ii])

    return score


def _get_model_args(
    graph_file,
    options_file,
    pops=None,
    r_bins=None,
    u=None,
    fit_u=False,
    phased=False,
):
    """
    Get a tuple of arguments for ``_evaluate_ll``.
    """
    # Check for required keyword arguments
    if pops is None:
        raise ValueError("pops is required")
    if r_bins is None:
        raise ValueError("r_bins is required")
    if u is None:
        raise ValueError("u is required")

    # Build the demography
    builder = _get_demes_dict(graph_file)
    options = _get_params_dict(options_file)
    param_names, params = _set_up_params_and_bounds(options, builder)[:2]

    args = (
        builder,
        options,
        sampled_demes,
        sample_times,
        u,
        fit_u,
        r_bins,
        phased,
    )
    return param_names, params, args


_model_cache = dict()


def _evaluate_ll(params, means, varcovs, args):
    """
    Evaluate the likelihood of the model at a point in parameter space.
    """
    key = tuple(params)
    if key in _model_cache:
        model = _model_cache[key]
    else:
        (
            builder,
            options,
            sampled_demes,
            sample_times,
            u,
            fit_u,
            r_bins,
            phased,
        ) = args

        if fit_u:
            raise ValueError("not implemented")  # TODO

        builder = _update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        model = H2stats.from_demes(
            graph,
            sampled_demes=sampled_demes,
            sample_times=sample_times,
            r_bins=r_bins,
            u=u,
            phased=phased,
        )
        _model_cache[key] = model

    # TODO cache likelihoods?

    return inference._compute_composite_ll(model, means, varcovs)


def _get_e_i(n, idx):
    """Get a vector of length ``n`` with 1 at ``idx`` and 0 elsewhere."""
    return np.array([1 if i == idx else 0 for i in range(n)])












