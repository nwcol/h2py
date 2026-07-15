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


# =============================================================================
# Principal functions for uncertainty estimation and LRT adjustment
# =============================================================================


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
    readout=True,
    delta=0.01,
    bounds=None,
    steps=None,
):
    """
    Compute confidence intervals for MLE parameters.

    Parameters
    ----------
    graph_file : str, path
        Path to YAML file containing optimized (MLE) model parameters in Demes
        format.
    options_file : str, path
        Moments.Demes-style parameter specification file.
    means : list
        Arrays of mean H2 statistics- one for each bin.
    varcovs : list
        Covariance matrices generated with the bootstrap.
    boot_means : list, optional
        Bootstrap sample means; required to estimate uncertainty with the
        Godambe information matrix.
    pops : list
        List of populations corresponding to the order of statistics in input
        data. Each element must match the name of a deme in ``graph_file``.
    r_bins : # TODO
        Array of recombination bin edges.
    u : float
        Mutation rate parameter.
    fit_u : bool, optional
        If True, treat the mutation rate as a free parameter and estimate its
        uncertainty.
    phased : bool, optional
        If True, compute phased model expectations. Should only be True when
        H2 was computed from phased data with the haplotype setting.
    method : str, optional
        Uncertainty estimation method to use. Must be "GIM", "godambe", "FIM",
        or "fisher". The "FIM"/"fisher" method is misspecified for composite
        likelihood and should be taken critically.
    return_matrix : bool, optional
        If True, return the estimated Fisher or Godambe information matrix
        (as specified by ``method``) along with minimal outputs.
    readout : bool, optional
        If True, print a table of estimated parameter uncertainties.
    report : bool, optional
        If True, print progress messages.
    delta : float, optional
        Fractional step size for finite difference calculations.
    bounds : tuple, optional
        Optional upper/lower bounds on parameters. If bounds are not specified,
        they are constructed using ``options_file``.
    steps : np.ndarray, shape (n_params,), optional
        Step sizes for finite difference calculations.

    Returns
    -------
    param_names : list
        Parameter names as defined in ``options_file``.
    params : np.ndarray, shape (n_params,)
        MLE parameter values loaded from ``graph_file``.
    uncerts : np.ndarray, shape (n_params,)
        Estimated uncertainties, in standard deviations.
    matrix : np.ndarray, shape (n_params, n_params), optional
        Fisher or Godambe information matrix; returned if ``return_matrix`` is 
        True.
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

    if bounds is None:
        bounds = _get_param_bounds(options_file, param_names, params)

    if method == "FIM" or method == "fisher":
        args = (means, varcovs, model_args)
        HH = _get_hessian_matrix(
            params,
            _evaluate_ll,
            args=args,
            delta=delta,
            steps=steps,
            bounds=bounds,
            report=report,
        )
        FIM = np.linalg.inv(-HH)
        uncerts = np.sqrt(np.diag(FIM))
        matrix = FIM

    elif method == "GIM" or method == "godambe":
        if boot_means is None or len(boot_means) == 0:
            raise ValueError("bootstrap samples are needed to use 'GIM'")

        _, __, GIM = _get_godambe_matrix(
            params,
            _evaluate_ll,
            means,
            varcovs,
            boot_means,
            model_args,
            delta=delta,
            steps=steps,
            bounds=bounds,
            report=report,
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(GIM)))
        matrix = GIM

    else:
        raise ValueError("unrecognized method")

    if readout:
        print(f"Finished with method {method}")
        print(f"Estimated uncertainties:")
        print("    Param\tSE\t\tCI (95%)")
        for name, val, std in zip(param_names, params, uncerts):
            err = 1.96 * std
            print(f"    {name}\t{std:.3}\t({val-err:.3}, {val+err:.3})")

    if return_matrix:
        return param_names, params, uncerts, matrix
    else:
        return param_names, params, uncerts


def compute_lrt_adjustment(
    nested_graph_file,
    full_graph_file,
    options_file,
    means,
    varcovs,
    boot_means,
    nested_params=[], # TODO how to handle? need refresher here
    nested_values=[],
    pops=None,
    r_bins=None,
    u=None,
    fit_u=None, # TODO does this matter here?
    return_matrix=False,
    report=True,
    bounds=None,
    steps=None,
):
    """
    Compute an adjustment factor for the LRT test statistic after Coffman et
    al 2015.

    Parameters
    ----------
    nested_graph_file : str, path
        Demes YAML file with *nested* model at MLE parameters.
    full_graph_file : str, path
    options_file : str, path
        Parameter specification file for the *full* model. Other than the
        presence of nested parameters, this should be identical to the options
        file used to fit the nested model.
    ...
    nested_params : list
        List of nested parameter names.
    nested_values : array-like
        (boundary?) values of nested parameters.

    Returns
    -------
    """


    """
    the idea is to manually drop nested parameters from loaded full-model
    options, use them to load the nested model; then drop the nested-model
    parameters into the full model graph and edit the nested parameters to
    match ``nested_values``.
    """
    # TODO setup
    params = "TODO"
    param_names = "TODO" # Should be a list

    nested_idx = np.array([param_names.index(p) for p in nested_params])

    def nesting_func(p_nest, args):
        """Adjust nested parameters only."""
        p_full = np.array(params, copy=True)
        p_full[nexted_idx] = p_nest
        return _evaluate_ll(p_full, *args)

    HH, JJ, _ = _get_godambe_matrix(
        nested_params,
        means,
        varcovs,
        boot_means,
        model_args,
        func=nesting_func,
        delta=delta,
        steps=steps,
        bounds=bounds,
        report=report,
    )
    fac = len(nested_idx) / np.trace(JJ @ np.linalg.inv(HH))

    if return_matrix:
        return fac, HH, JJ
    else:
        return fac


# -----------------------------------------------------------------------------
# Derivative calculators and utilities
# -----------------------------------------------------------------------------


def _get_godambe_matrix(
    params,
    func,
    means,
    varcovs,
    boot_means,
    model_args,
    delta=0.01,
    steps=None,
    bounds=None,
    report=True,
):
    """
    Compute the Godambe information matrix (GIM).
    """
    args = (means, varcovs, model_args)
    HH = _get_hessian_matrix(
        params,
        func=func,
        args=args,
        delta=delta,
        steps=steps,
        bounds=bounds,
        report=report,
    )
    JJ = _get_variability_matrix(
        params,
        func,
        varcovs,
        boot_means,
        model_args=model_args,
        delta=delta,
        steps=steps,
        bounds=bounds,
        report=report,
    )
    GIM = -HH @ np.linalg.inv(JJ) @ -HH
    return HH, JJ, GIM


def _get_variability_matrix(
    params,
    func,
    varcovs,
    boot_means,
    model_args=[],
    delta=0.01,
    steps=None,
    bounds=None,
    report=True,
):
    """
    Compute the observed variability matrix: an approximation to the Hessian
    matrix for composite likelihood, calculated with the bootstrap.
    """
    matrix = np.zeros((len(params), len(params)), dtype=np.float64)

    for means in boot_means:
        args = (means, varcovs, model_args)
        score = _get_score(
            params,
            func,
            args=args,
            delta=delta,
            steps=steps,
            bounds=bounds,
        )
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
    n_params = len(params)

    # Set up bounds if non are given
    if bounds is None:
        bounds = (np.zeros(n_params), np.full(n_params, np.inf))

    # Check bounds
    if np.any(params < bounds[0]) or np.any(params > bounds[1]):
        raise ValueError("initial parameters violate bounds")

    if steps is None:
        steps = delta * params
        if np.any(steps == 0):
            steps[steps == 0] = delta

    for ii in range(n_params):
        if np.any((params - steps < bounds[0]) & (params + steps > bounds[1])):
            raise ValueError("bounds prevent finite differences evaluation")

    hessian = np.zeros((n_params, n_params), dtype=np.float64)
    # Calculate the elements of the matrix by separate function calls
    for ii in range(n_params):
        for jj in range(n_params):
            elem = _get_hessian_elem(params, ii, jj, func, args, steps, bounds)
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
    score : np.ndarray, shape (n_params, 1)
        Gradient, as a column vector.
    """
    n_params = len(params)

    # Set up bounds if non are given
    if bounds is None:
        bounds = (np.zeros(n_params), np.full(n_params, np.inf))

    # Check bounds
    if np.any(params < bounds[0]) or np.any(params > bounds[1]):
        raise ValueError("initial parameters violate bounds")

    if steps is None:
        steps = delta * params
        if np.any(steps == 0):
            steps[steps == 0] = delta

    for ii in range(n_params):
        if np.any((params - steps < bounds[0]) & (params + steps > bounds[1])):
            raise ValueError("bounds prevent finite differences evaluation")

    lower, upper = bounds
    score = np.zeros((n_params, 1), dtype=np.float64)

    for ii in range(n_params):
        e_i = _get_e_i(n_params, ii)

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
            score[ii, 0] = (f_f - f_0) / steps[ii]

        elif form == "backward":
            f_b = func(params + steps * -e_i, *args)
            f_0 = func(params, *args)
            score[ii, 0] = (f_0 - f_b) / steps[ii]

        else:
            f_b = func(params + steps * -e_i, *args)
            f_f = func(params + steps * e_i, *args)
            score[ii, 0] = (f_f - f_b) / (2 * steps[ii])

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
    Get a tuple of model construction arguments for ``_evaluate_ll``.
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

    deme_names = [d["name"] for d in builder["demes"]]
    sample_times = []
    for pop in pops:
        assert pop in deme_names
        idx = deme_names.index(pop)
        end_time = builder["demes"][idx]["epochs"][-1]["end_time"]
        sample_times.append(end_time)
    sampled_demes = pops

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


def _get_param_bounds(options_file, param_names, params):
    """
    Get upper/lower bounds for parameters.
    """
    # TODO are constraints handled properly?
    options = _get_params_dict(options_file)

    lower = np.zeros(len(params), dtype=np.float64)
    upper = np.full(len(params), np.inf, dtype=np.float64)

    # Get a mapping between parameter names/specifications
    param_dict = {x["name"]: x for x in options["parameters"]}

    for ii, name in enumerate(param_names):
        spec = param_dict[name]
        if "lower_bound" in spec:
            lower[ii] = spec["lower_bound"]
        if "upper_bound" in spec:
            upper[ii] = spec["upper_bound"]

    if "constraints" in options:
        for spec in options["constraints"]:
            idx_0 = param_names.index(spec["params"][0])
            idx_1 = param_names.index(spec["params"][1])
            if spec["constraint"] == "greater_than":
                lower[idx_0] = max(lower[idx_0], params[idx_1])
                upper[idx_1] = min(upper[idx_1], params[idx_0])
            elif spec["constraint"] == "less_than":
                upper[idx_0] = min(upper[idx_0], params[idx_1])
                lower[idx_1] = max(lower[idx_1], params[idx_0])
            else:
                raise ValueError("invalid constraint")

    return lower, upper


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

