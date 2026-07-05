"""

plan: two primary functions.
- plot several models/data: one statistic per axis
- plot a single model OR a single model and a single data: several stats per ax
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
    Two-way comparison.
    """

    if stats is None:
        stats = h2_stats.h2_names

    if stats_to_plot is None:
        stats_to_plot = [[x] for x in stats]

    if labels is None:
        labels = stats_to_plot

    if rows is None and cols is None:
        rows = 1
        cols = len(stats_to_plot)
    elif rows is None:
        rows = int(np.ceil(len(stats_to_plot) / cols))
    elif cols is None:
        cols = int(np.ceil(len(stats_to_plot) / rows))

    xs = np.mean(r_bins, axis=1)

    fig, axs = plt.subplots(rows, cols, figsize=fig_size, layout="tight")
    f_axs = axs.flat

    for ii, (ax_stats, ax_labels) in enumerate(zip(stats_to_plot, labels)):
        ax = f_axs[ii]

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




def _plot_h2_curves(
    h2_stats=[],
    means=[],
    varcovs=[],
    stats=None,
    stats_to_plot=[],
    labels=None,
    model_labels=None,
    data_labels=None,
    rs=None,
    r_bins=None,
    pops=None,
    rows=None,
    cols=None,
    fig_size=(6, 6),
    ax=None,
    out_file=None,
    dpi=244,
):
    """
    """


    if not isinstance(h2_stats, list):
        h2_stats = [h2_stats]
    if len(means) > 0:
        if not isinstance(means[0], list):
            means = [means]
    if len(varcovs) > 0:
        if not isinstance(varcovs[0], list):
            varcovs = [varcovs]
    assert len(means) == len(varcovs)

    if rs is not None and r_bins is not None:
        raise ValueError("`rs` and `r_bins` cannot both be given")
    if rs is None and r_bins is None:
        raise ValueError("either `rs` or `r_bins` must be given")

    if rs is not None:
        assert len(means) == 0 and len(varcovs) == 0
        xs = rs
    else:
        # Assumes bins are in tuple format
        xs = np.mean(bins, axis=1)

    if labels is None:
        if pops is not None:
            labels = 0.0
        else:
            labels = stats_to_plot
    else:
        assert len(labels) == len(stats_to_plot)



    fig, axs = plt.subplots(rows, cols, figsize=figsize, layout="tight")
    f_axs = axs.flat

    for ii, (ax_stats, ax_labels) in enumerate(stats_to_plot, labels):
        ax = f_axs[ii]

        for jj, stat in enumerate(ax_stats):
            idx = stats.index(stat)


    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)
    return



