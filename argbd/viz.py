import numpy as np
import random
import scipy
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

col_green = "#228833"
col_red = "#EE6677"
col_purp = "#AA3377"
col_blue = "#66CCEE"
col_yellow = "#CCBB44"
col_indigo = "#4477AA"
col_grey = "#BBBBBB"
colorpal = [col_blue, col_green, col_red, col_purp, col_yellow, col_indigo, col_grey]


def qqplot(
    results_list,
    inds_to_plot=None,
    random_samples=None,
    cladesize_plot=False,
    depth_plot=False,
    hist_plot=False,
    colors="",
    legend_labels="",
    save_to_file="",
    outliers_threshold=None,
    figtitle="",
    no_legend=False,
    legend_loc="best",
    legend_font="medium",
    size=None,
):
    """
    Q-Q plot
    :param size:
    :param no_legend:
    :param results_list:
    :param inds_to_plot:
    :param random_samples:
    :param cladesize_plot:
    :param depth_plot:
    :param hist_plot:
    :param colors:
    :param legend_labels:
    :param save_to_file:
    :param outliers_threshold:
    :param figtitle:
    :return:
    """
    if not hist_plot:
        if size is None:
            size = (5, 4.5)
        plt.figure(figsize=size)
    if not colors:
        colors = colorpal
    names = []
    for i, results in enumerate(results_list):
        if inds_to_plot is not None:
            qvals = results.q[inds_to_plot]
            qvals = np.sort(qvals)
            plt.plot(
                [i / (1 + len(inds_to_plot)) for i in range(1, len(inds_to_plot) + 1)],
                qvals,
                color=colors[i],
                lw=4,
            )
            if results.name == "tsinfer+tsdate":
                results.name = "tsdate"
            if results.name == "tsinfer+tsdate (top_ch)":
                results.name = "tsdate (top_ch)"
            names.append(results.name)
        elif random_samples is not None:
            n, m = random_samples
            for j in range(n):
                inds = random.sample(range(0, results.num), m)
                qvals = results.q[inds]
                qvals = np.sort(qvals)
                plt.plot(
                    [i / (1 + len(inds)) for i in range(1, len(inds) + 1)],
                    qvals,
                    color=colors[j],
                    lw=4,
                )
                names = [str(1 + i) for i in range(random_samples[0])]
        elif depth_plot:
            if 1 + max(results.depth) > len(colors):
                colors = plt.cm.get_cmap("rainbow", max(results.depth))
                colors = [colors(i) for i in range(1 + max(results.depth))]
            for j in range(1 + max(results.depth)):
                names.append(str(j))
                inds = np.where(results.depth == j)[0]
                qvals = results.q[inds]
                qvals = np.sort(qvals)
                plt.scatter(
                    [i / (1 + len(inds)) for i in range(1, len(inds) + 1)],
                    qvals,
                    color=colors[j],
                    alpha=0.2,
                    s=2,
                )
                if len(qvals) > 0:
                    print(
                        "K-S test for depth"
                        + str(j)
                        + ":"
                        + str(scipy.stats.kstest(qvals, scipy.stats.uniform.cdf))
                    )
        elif cladesize_plot:
            if max(results.cladesize) > len(colors):
                colors = plt.cm.get_cmap("rainbow", max(results.cladesize))
                colors = [colors(i) for i in range(1 + max(results.cladesize))]
            for j in range(max(results.cladesize)):
                names.append(str(j + 1))
                inds = np.where(results.cladesize == j + 1)[0]
                qvals = results.q[inds]
                qvals = np.sort(qvals)
                plt.scatter(
                    [i / (1 + len(inds)) for i in range(1, len(inds) + 1)],
                    qvals,
                    color=colors[j],
                    alpha=0.2,
                    s=2,
                )
                if len(inds) > 0:
                    print(
                        "K-S test for clade size"
                        + str(1 + j)
                        + " with n = "
                        + str(len(qvals))
                        + " :"
                        + str(scipy.stats.kstest(qvals, scipy.stats.uniform.cdf))
                    )
                else:
                    print("No branches with clade size" + str(1 + j))
        elif hist_plot:
            if size is None:
                size = (3.5, 3.5)
            plt.figure(figsize=size)
            plt.hist(results.q, color=colors[i], alpha=1, density=True)
            if results.name == "tsinfer+tsdate":
                results.name = "tsdate"
            if results.name == "tsinfer+tsdate (top_ch)":
                results.name = "tsdate (top_ch)"
            names.append(results.name)
        else:
            qvals = results.qsorted
            plt.plot(
                [i / (1 + len(qvals)) for i in range(1, len(qvals) + 1)],
                qvals,
                color=colors[i],
                lw=3,
            )
            if results.name == "tsinfer+tsdate":
                results.name = "tsdate"
            if results.name == "tsinfer+tsdate (top_ch)":
                results.name = "tsdate (top_ch)"
            names.append(results.name)
    if outliers_threshold:
        for i, results in enumerate(results_list):
            inds = np.where(results.qsorted > outliers_threshold)[0]
            plt.plot(
                inds / (1 + results.num), results.qsorted[inds], color=col_red, lw=4
            )
    if not hist_plot:
        plt.plot([0, 1], [0, 1], color="black", ls="--")
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Empirical quantiles")
    else:
        plt.xlim((0, 1))
        plt.xlabel("q")
        plt.tight_layout()
    plt.title(figtitle)
    if legend_labels:
        names = legend_labels
    if cladesize_plot:
        maxcladesize = max([max(results.cladesize) for results in results_list])
        if maxcladesize > 5:
            i1 = math.floor(maxcladesize / 4)
            lines = [Line2D([0], [0], color=colors[i1 * i], alpha=1) for i in range(5)]
            names = [str(i1 * i) for i in range(5)]
        else:
            lines = [
                Line2D([0], [0], color=colors[i], alpha=1) for i in range(maxcladesize)
            ]
    elif depth_plot:
        maxdepth = max([max(results.depth) for results in results_list])
        if maxdepth > 5:
            i1 = math.floor(maxdepth / 4)
            lines = [Line2D([0], [0], color=colors[i1 * i], alpha=1) for i in range(5)]
            names = [str(i1 * i) for i in range(5)]
        else:
            lines = [
                Line2D([0], [0], color=colors[i], alpha=1) for i in range(maxdepth)
            ]
    else:
        lines = [Line2D([0], [0], color=c, alpha=1) for c in colors[0 : len(names)]]
    if not no_legend:
        if hist_plot:
            if legend_loc is None:
                plt.legend(lines, names, loc="lower right", fontsize=legend_font)
            else:
                plt.legend(lines, names, loc=legend_loc, fontsize=legend_font)
        else:
            plt.legend(lines, names, loc=legend_loc, fontsize=legend_font)
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches="tight")
    plt.show()


def outliers_plot(
    results,
    outliers_threshold=0.05,
    save_to_file="",
    size=(10, 4),
    xticks=None,
    yticks=None,
):
    """
    Outliers plot
    :param yticks:
    :param xticks:
    :param results:
    :param outliers_threshold:
    :param save_to_file:
    :param size:
    :return:
    """
    num = results.num
    num_ = results.num_
    fig, ax = plt.subplots(1, 1, figsize=size)
    inds = np.where(results.sfsorted < outliers_threshold / num)[0]
    ax.scatter(
        [i / (1 + num_) for i in range(1, num_ + 1)],
        results.sfsorted,
        color=col_blue,
        s=4,
    )
    ax.scatter(
        (inds + 1) / (1 + num_),
        (results.sfsorted[inds]),
        color=col_red,
        s=6,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticks is not None:
        ax.set_xticks(xticks)

    ax.plot([0, 1], [0, 1], color="black", ls="--")
    ax.set_xlabel("Expected $p$-values")
    ax.set_ylabel("Observed $p$-values")

    plt.tight_layout()
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches="tight")

    plt.show()


def mean_change_plot(
    results_ranked, sim_type="height", save_to_file=None, size=(4.5, 4), no_legend=False
):
    plt.figure(figsize=size)
    x = [i for i in range(len(results_ranked[0]))]
    plt.bar(x, results_ranked[3], width=1, color=col_green, align="center")
    plt.bar(
        x,
        results_ranked[2],
        width=1,
        bottom=results_ranked[3],
        color=col_blue,
        align="center",
    )
    plt.bar(
        x,
        results_ranked[1],
        width=1,
        bottom=results_ranked[3] + results_ranked[2],
        color=col_red,
        align="center",
    )
    plt.xlim(
        (
            np.where(results_ranked[0] > 1)[0][0] - 0.5,
            np.where(results_ranked[0] > 1)[0][-1] + 0.5,
        )
    )
    plt.ylim((0, 1))
    plt.xlabel("Tree " + sim_type)
    plt.ylabel("Probability")
    red_patch = matplotlib.patches.Patch(color=col_red, label="Decrease")
    blue_patch = matplotlib.patches.Patch(color=col_blue, label="No change")
    green_patch = matplotlib.patches.Patch(color=col_green, label="Increase")
    if not no_legend:
        plt.legend(
            bbox_to_anchor=(0.99, 0.99),
            loc="upper right",
            handles=[red_patch, blue_patch, green_patch],
        )
    plt.tight_layout()
    if save_to_file:
        plt.savefig(save_to_file, dpi=300, bbox_inches="tight")
    plt.show()

