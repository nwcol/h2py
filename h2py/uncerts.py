"""
Compute uncertainty in inferred parameters.
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
from .stats import H2stats
from .utils import timestamp


def compute_uncerts(
    graph_file,
    options_file,
    means,
    varcovs,
    boot_means=[],
    pops=None,
    r_bins=None,
    u=None,
    fit_u=False,
    phased=False,
    method="GIM",
    return_matrix=False,
    report=True,
    delta=0.01,
    bounds=None,
    steps=None,
):
    """
    Compute confidence intervals for MLE parameters.
    """

    param_names, params, model_args = _get_model_args(
        graph_file,
        options_file,
        pops=pops,
        r_bins=r_bins,
        u=u,
        fit_u=fit_u,
        phased=phased,
    )

    # TODO set up bounds

    args = (means, varcovs, model_args)

    HH = _get_hessian_matrix(
        params
        _evaluate_ll,
        args,
        delta=delta,
        steps=steps,
        bounds=bounds,
        report=report,
    )

    if method == "FIM" or method == "fisher":
        FIM = np.linalg.inv(HH)
        uncerts = np.sqrt(np.diag(matrix))
        matrix = FIM

    elif method == "GIM" or method == "godambe":
        JJ = _get_variability_matrix(

        )
        # Godambe matrix
        GIM = HH @ np.linalg.inv(JJ) @ HH
        matrix = GIM

    else:
        raise ValueError("unrecognized method")

    if return_matrix:
        return param_names, params, uncerts, matrix
    else:
        return param_names, params, uncerts


def compute_lrt_adjustment(
):
    """
    Compute an adjustment factor for the LRT test statistic.
    """
    return


def _get_variability_matrix():

    """
    Compute the observed variability matrix, an approximation to the Hessian
    matrix for composite likelihood which is calculated with the bootstrap.
    """
    matrix = np.zeros((len(params), len(params)), dtype=np.float64)

    for means in boot_means:
        score = _get_score()
        matrix += score @ score.T

    return matrix / len(boot_means)


def _get_hessian_matrix(
    params,
    func,
    args=[],
    delta=0.01,
    steps=None,
    bounds=None,
    report=True,
):
    """
    Evaluate the Hessian matrix of the LL function at a point in parameter
    space.

    Parameters
    ----------

    Returns
    -------
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

    hessian = np.zeros((n_params, n_params), dtype=np.float64)
    # Calculate the elements of the matrix by separate function calls
    for ii in range(n_params):
        for jj in range(n_params):
            elem = _get_hessian_element(params, ii, jj, func, args, steps, bounds)
            if ii == jj:
                hessian[ii, ii] = elem
            else:
                hessian[ii, jj] = hessian[jj, ii] = elem
            if report:
                print(timestamp(), f"Computed Hessian element ({ii}, {jj})")

    return hessian


def _get_hessian_elem(
    params,
    ii,
    jj,
    func,
    args,
    steps,
    bounds,
    return_form=False,
):
    """
    Evaluate element (ii, jj) of the Hessian matrix.
    """
    lower, upper = bounds
    f_0 = func(params, *args)

    if ii == jj:
        e_i = _get_e_i(len(params), ii)

        # Determine which points to evaluate
        if params[ii] == 0 or params[ii] - steps[ii] < lower[ii]:
            form = "forward"
        elif params[ii] + steps[ii] > upper[ii]:
            form = "backward"
        else:
            form = "central"

        if form == "forward":
            f_f = func(params + steps * e_i, *args)
            f_2f = func(params + steps * 2 * e_i, *args)
            elem = (f_0 - 2 * f_f + f_2f) / steps[ii] ** 2

        elif form == "backward":
            f_b = func(params + steps * -e_i, *args)
            f_2b = func(params + steps * -2 * e_i, *args)
            elem = (f_0 - 2 * f_b + f_2b) / steps[ii] ** 2

        else:
            f_b = func(params + steps * -e_i, *args)
            f_f = func(params + steps * e_i, *args)
            elem = (f_f - 2 * f_0 + f_b) / steps[ii] ** 2

    else:
        e_i = _get_e_i(len(params), ii)
        e_j = _get_e_i(len(params), jj)

        # Determine which points to evaluate
        if params[ii] == 0 or params[ii] - steps[ii] < lower[ii]:
            form_i = "forward"
        elif params[ii] + steps[ii] > upper[ii]:
            form_i = "backward"
        else:
            form_i = "central"

        if params[jj] == 0 or params[jj] - steps[jj] < lower[jj]:
            form_j = "forward"
        elif params[jj] + steps[jj] > upper[jj]:
            form_j = "backward"
        else:
            form_j = "central"

        form = ",".join([form_i, form_j])

        if form == "backward,backward":
            f_bb = func(params + steps * (-e_i - e_j), *args)
            f_0b = func(params + steps * -e_j, *args)
            f_b0 = func(params + steps * -e_i, *args)
            elem = (f_bb - f_0b - f_b0 + f_0) / (steps[ii] * steps[jj])

        elif form == "backward,central":
            f_0f = func(params + steps * e_j, *args)
            f_bf = func(params + steps * (-e_i + e_j), *args)
            f_0b = func(params + steps * -e_j, *args)
            f_bb = func(params + steps * (-e_i + -e_j), *args)
            elem = (f_0f - f_bf - f_0b + f_bb) / (2 * steps[ii] * steps[jj])

        elif form == "backward,forward":
            f_0f = func(params + steps * e_j, *args)
            f_bf = func(params + steps * (-e_i + e_j), *args)
            f_b0 = func(params + steps * -e_i, *args)
            elem = (f_0f - f_0 - f_bf + f_b0) / (steps[ii] * steps[jj])

        elif form == "central,backward":
            f_f0 = func(params + steps * e_i, *args)
            f_fb = func(params + steps * (e_i - e_j), *args)
            f_b0 = func(params + steps * -e_i, *args)
            f_bb = func(params + steps * (-e_i - e_j), *args)
            elem = (f_f0 - f_fb - f_b0 + f_bb) / (2 * steps[ii] * steps[jj])

        elif form == "central,central":
            f_ff = func(params + steps * (e_i + e_j), *args)
            f_fb = func(params + steps * (e_i - e_j), *args)
            f_bf = func(params + steps * (-e_i + e_j), *args)
            f_bb = func(params + steps * (-e_i + -e_j), *args)
            elem = (f_ff - f_fb - f_bf + f_bb) / (4 * steps[ii] * steps[jj])

        elif form == "central,forward":
            f_ff = func(params + steps * (e_i + e_j), *args)
            f_f0 = func(params + steps * e_i, *args)
            f_bf = func(params + steps * (-e_i + e_j), *args)
            f_b0 = func(params + steps * -e_i, *args)
            elem = (f_ff - f_f0 - f_bf + f_b0) / (2 * steps[ii] * steps[jj])

        elif form == "forward,backward":
            f_f0 = func(params + steps * e_i, *args)
            f_fb = func(params + steps * (e_i - e_j), *args)
            f_0b = func(params + steps * -e_j, *args)
            elem = (f_f0 - f_fb - f_0 + f_0b) / (steps[ii] * steps[jj])

        elif form == "forward,central":
            f_ff = func(params + steps * (e_i + e_j), *args)
            f_fb = func(params + steps * (e_i - e_j), *args)
            f_0f = func(params + steps * e_j, *args)
            f_0b = func(params + steps * -e_j, *args)
            elem = (f_ff - f_fb - f_0f + f_0b) / (2 * steps[ii] * steps[jj])

        elif form == "forward,forward":
            f_ff = func(params + steps * (e_i + e_j), *args)
            f_f0 = func(params + steps * e_i, *args)
            f_0f = func(params + steps * e_j, *args)
            elem = (f_ff - f_f0 - f_0f + f_0) / (steps[ii] * steps[jj])

        # This will never happen
        else:
            raise ValueError("invalid form")

    if return_form:
        return elem, form
    else:
        return elem


def _get_score(
    params,
    func,
    args=[],
    delta=0.01,
    steps=None,
    bounds=None,
):
    """
    Compute the score, the gradient of the LL function, using finite
    differences.

    Parameters
    ----------
    params : np.ndarray, shape (n_params,)
        MLE parameter values; point about which the gradient is evaluated.
    func : Callable
        The function for which to evaluate the gradient.
    args : tuple
        Additional arguments to ``func``. For evaluating the score this should
        be ``(means, varcovs, model_args)``, where ``model_args`` is from
        ``_get_model_args``.
    delta : float, optional
        Fractional step size for finite differences.
    steps : np.ndarray, optional
        Optional array of manually-specified step sizes.
    bounds : tuple, optional
        Upper and lower bounds for function evaluation.

    Returns
    -------
    score : np.ndarray, shape (n_params,) # TODO may change
        Gradient.
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
    score = np.zeros((1, len(params)), dtype=np.float64)

    for ii in range(len(params)):
        e_i = _get_e_i(len(params), ii)

        # Determine which points to evaluate
        if params[ii] == 0 or  params[ii] - steps[ii] < lower[ii]:
            form = "forward"
        elif params[ii] + steps[ii] > upper[ii]:
            form = "backward"
        else:
            form = "central"

        if form == "forward":
            f_0 = func(params, *args)
            f_f = func(params + steps * e_i, *args)
            score[:, ii] = (f_f - f_0) / steps[ii]

        elif form == "backward":
            f_b = func(params + steps * -e_i, *args)
            f_0 = func(params, *args)
            score[:, ii] = (f_0 - f_b) / steps[ii]

        else:
            f_b = func(params + steps * -e_i, *args)
            f_f = func(params + steps * e_i, *args)
            score[:, ii] = (f_f - f_b) / (2 * steps[ii])

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

