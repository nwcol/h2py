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
        rows = 1
        cols = len(stats_to_plot)
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
        rows = 1
        cols = n_panels
    elif rows is None:
        rows = int(np.ceil(n_panels / cols))
    elif cols is None:
        cols = int(np.ceil(n_panels / rows))

    xs = (r_bins[1:] + r_bins[:-1]) / 2

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
            for ms, vc in zip(means, varcovs):
                data = np.array([m[idx] for m in ms[:-1]])
                data_err = 1.96 * np.array([v[idx, idx] for v in vc[:-1]]) ** 0.5
                ax.fill_between(xs, data - data_err, data + data_err, alpha=0.3)
            ax.set_prop_cycle(None)

        for jj, ms in enumerate(means):
            label = data_labels[jj]
            data = [m[idx] for m in ms[:-1]]
            ax.plot(xs, data, "--", label=label)

        for jj, model in enumerate(h2_stats):
            label = model_labels[jj]
            data = [x[idx] for x in model.h2()]
            ax.plot(xs, data, label=label)

        # Format the panel
        ax_label = stat_labels[ii]
        ax.set_title(ax_label)
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=7)
        if ii >= rows * (cols - 1) - 1:
            ax.set_xlabel("$r$")
        if ii % cols == 0:
            ax.set_ylabel("$H_2$")

    # Clean up extra axes
    for ax in f_axs[len(stats_to_plot):]:
        ax.remove()

    if title is not None:
        fig.suptitle(title)

    if out_file is not None:
        plt.savefig(out_file, dpi=dpi)
    if show:
        plt.show()
    return fig

