"""
Plot observed/expected H2.
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from h2py import utils


def plot_h2_curves_comp(
    h2_stats,
    means,
    varcovs,
    stats=None,
    stats_to_plot=None,
    plot_errs=True,
    labels=None,
    pops=None,
    r_bins=None,
    fig_size=(6, 6),
    rows=None,
    cols=None,
    show=True,
    dpi=244,
    out_file=None,
):
    """
    Plot a two-way comparison between an observed an an expected data set.
    """

    if stats is None:
        stats = h2_stats.h2_names

    if stats_to_plot is None:
        stats_to_plot = [[x] for x in stats]

    if labels is None:
        labels = stats_to_plot

    if rows is None and cols is None:
        n_panels = len(stats)
        cols = int(np.ceil(np.sqrt(n_panels)))
        rows = int(np.ceil(n_panels / cols))
    elif rows is None:
        rows = int(np.ceil(len(stats_to_plot) / cols))
    elif cols is None:
        cols = int(np.ceil(len(stats_to_plot) / rows))

    # Bins are expected to be a list of 2-tuples specifying bin edges.
    xs = np.mean(r_bins, axis=1)

    fig, axs = plt.subplots(rows, cols, figsize=fig_size, layout="tight")
    f_axs = axs.flat

    for ii, (ax_stats, ax_labels) in enumerate(zip(stats_to_plot, labels)):
        ax = f_axs[ii]

        if plot_errs:
            for stat in ax_stats:
                idx = stats.index(stat)
                data = np.array([m[idx] for m in means[:-1]])
                data_err = 1.96*np.array([v[idx, idx] for v in varcovs[:-1]])**0.5
                ax.fill_between(xs, data - data_err, data + data_err, alpha=0.3)

        ax.set_prop_cycle(None)
        for stat in ax_stats:
            idx = stats.index(stat)
            data = [m[idx] for m in means[:-1]]
            ax.plot(xs, data, "--")

        ax.set_prop_cycle(None)
        for jj, stat in enumerate(ax_stats):
            label = ax_labels[jj]
            idx = stats.index(stat)
            data = [x[idx] for x in h2_stats.h2()]
            ax.plot(xs, data, label=label)

        # Format the axis
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=7)
        if ii >= rows * (cols - 1) - 1:
            ax.set_xlabel("$r$")
        if ii % cols == 0:
            ax.set_ylabel("$H_2$")

    # Clean up extra axes
    for ax in f_axs[len(stats_to_plot):]:
        ax.remove()

    if show:
        plt.show()
    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)
    return fig


def plot_h2_curves(
    h2_stats=[],
    means=[],
    varcovs=[],
    stats=None,
    stats_to_plot=None,
    stat_labels=None,
    model_labels=None,
    data_labels=None,
    colors=None,
    alpha=0.3,
    plot_errs=True,
    # rs=None, TODO
    r_bins=None,
    pops=None,
    title=None,
    rows=None,
    cols=None,
    fig_size=(6, 6),
    show=True,
    out_file=None,
    dpi=244,
):
    """
    Plot arbitarily many empirical or expected data sets.

    Places each statistic on its own panel and assigns a unique color to each
    dataset.

    Work in progress: not all intended features are implemented.
    """
    assert stats is not None and len(stats) > 0

    if stats_to_plot is None:
        stats_to_plot = stats
    if stat_labels is None:
        stat_labels = stats_to_plot

    if model_labels is None:
        model_labels = [f"dataset {i}" for i in range(len(h2_stats))]
    if data_labels is None:
        data_labels = [f"model {i}" for  i in range(len(means))]

    n_panels = len(stats_to_plot)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_panels)))
        rows = int(np.ceil(n_panels / cols))
    elif rows is None:
        rows = int(np.ceil(n_panels / cols))
    elif cols is None:
        cols = int(np.ceil(n_panels / rows))

    if colors is None:
        colors = mpl.colors.TABLEAU_COLORS

    xs = np.mean(r_bins, axis=1)
    #xs = (r_bins[1:] + r_bins[:-1]) / 2

    fig, axs = plt.subplots(rows, cols, figsize=fig_size, layout="constrained")

    # Figures with 1x1 panels are a special case
    if rows > 1:
        f_axs = axs.flat
    else:
        if cols == 1:
            f_axs = [axs]
        else:
            f_axs = axs

    # Loop over axes/stats
    for ii, stat in enumerate(stats_to_plot):
        ax = f_axs[ii]
        idx = stats.index(stat)

        # Loop over datasets
        if plot_errs:
            for jj, (ms, vc) in enumerate(zip(means, varcovs)):
                data = np.array([m[idx] for m in ms[:-1]])
                data_err = 1.96 * np.array([v[idx, idx] for v in vc[:-1]]) ** 0.5
                ax.fill_between(xs, data - data_err, data + data_err, alpha=alpha,
                                color=colors[jj], edgecolor="none")
            #ax.set_prop_cycle(None)

        for jj, ms in enumerate(means):
            data = [m[idx] for m in ms[:-1]]
            ax.plot(xs, data, "--", color=colors[jj])

        for jj, model in enumerate(h2_stats):
            data = [x[idx] for x in model.h2()]
            ax.plot(xs, data, color=colors[jj + len(means)])

        # Format the panel
        ax_label = stat_labels[ii]
        ax.set_title(ax_label, fontsize="x-small", y=0.8)
        ax.set_xscale("log")
        #if ii >= cols * (rows - 1) - 1:
        #    ax.set_xlabel("$r$")
        if ii % cols == 0:
            ax.set_ylabel("$H_2$")

    if len(data_labels) > 0 or len(model_labels) > 0:
        _colors = []
        for c in colors:
            if c not in _colors:
                _colors.append(c)
        colors = _colors
        handles0 = [mpl.lines.Line2D([], [], color=c, label=l, linestyle="--")
                    for l, c in zip(data_labels, colors)]
        handles1 = [mpl.lines.Line2D([], [], color=c, label=l)
                    for l, c in zip(model_labels, colors[len(handles0):])]
        fig.legend(
            handles=handles0 + handles1,
            framealpha=0,
            loc="lower center",
            ncols=4,
            bbox_to_anchor=(0.5, -0.05 if rows >= 4 else -0.07)
        )

    for ax in axs[-1]:
        ax.set_xlabel("$r$")

    # Clean up extra axes
    for ax in f_axs[len(stats_to_plot):]:
        ax.remove()

    if title is not None:
        fig.suptitle(title)

    if out_file is not None:
        plt.savefig(out_file, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    return


def plot_param_distributions(
    data,
    groups,
    x_labels=None,
    data_labels=None,
    param_labels=None,
    figsize=(6, 6),
    grid_alpha=0.4,
    spread=0.3,
    capsize=0,
    out=None,
):
    """
    Plot parameter MLE and CI from one or more datasets.

    Parameters
    ----------
    data : list of pandas.DataFrame
        Each dataframe should have columns 'name', 'MLE', 'CI_lower', 'CI_upper'
    groups : list of dict
        Assign parameters to figure panels; parameters left out are not plotted.
        Should have keys 'params' and 'log', with the latter a boolean
        specifying whether to use a log scale for the panel.
    x_labels : list
        Optional labels for panel x-axes
    data_labels : list
        Optional labels for datasets
    param_labels : dict
        Optional mapping of parameter names to labels. Not all parameters need
        to be included.
    figsize : tuple
    grid_alpha : float
    spread : float
        Spread of datasets on the y-axis. If 0, error bars overlap on the
        y-axis entirely.
    capsize : float
        Cap size for error bars. Useful when ``spread`` is 0.
    """
    n_datasets = len(data)
    n_groups = len(groups)
    group_lens = [len(x["params"]) for x in groups]

    fig, axs = plt.subplots(
        n_groups, 1, figsize=figsize, layout="tight", height_ratios=group_lens)

    for i, (group, ax) in enumerate(zip(groups, axs)):
        params = group["params"]
        for j, param in enumerate(params):
            ax.set_prop_cycle(None)
            for k, df in enumerate(data):
                if param in list(df["name"]):
                    row = df[df["name"] == param]
                    mle = next(iter(row["MLE"]))
                    lower = mle - next(iter(row["CI_lower"]))
                    upper = next(iter(row["CI_upper"])) - mle
                    if lower < 0:
                        lower = 0
                    if upper < 0:
                        upper = 0
                else:
                    mle = lower = upper = np.nan

                if n_datasets > 1:
                    offset = spread * ((n_datasets - 1) / 2 - k) / ((n_datasets - 1) / 2)
                else:
                    offset = 0
                label = None
                if data_labels is not None:
                    if i == 0 and j == 0:
                        label = data_labels[k]
                ax.errorbar([mle], [-j + offset], xerr=[[lower], [upper]],
                            fmt="o", markersize=3, capsize=capsize, label=label)

        axs[0].legend(framealpha=0)

        # Label parameter values on the y axis
        group_param_labels = []
        for param in params:
            if param in param_labels:
                label = param_labels[param]
            else:
                label = param
            group_param_labels.append(label)
        ax.set_yticks(range(0, -len(params), -1), group_param_labels)

        if group["log"]:
            ax.set_xscale("log")

        if x_labels is not None:
            ax.set_xlabel(x_labels[i])

        ax.set_ylim(-len(params), 1)

        if grid_alpha > 0:
            ax.grid(alpha=grid_alpha)

    #if data_labels is not None:
    #    colors = mpl.colors.TABLEAU_COLORS
    #    labels = [mpl.lines.Line2D([], [], color=c, label=l)
    #              for c, l in zip(colors, data_labels)]
    #    axs[0].legend(handles=labels, framealpha=0)
    #    fig.legend(handles=labels, framealpha=0, loc="lower center", ncols=4,
    #        bbox_to_anchor=(0.5, 0.05))

    if out is not None:
        plt.savefig(out)

    return



