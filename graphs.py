import pandas as pd
import pyarrow.feather as feather
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import matplotlib
import forestplot as fp

from calculations import energy, dictmod


def plt_rcParams():
    fsize = 10
    tsize = 4
    tdir = "out"
    major = 5.0
    minor = 3.0
    lwidth = 0.8
    lhandle = 2.0
    plt.style.use("default")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = fsize
    plt.rcParams["legend.fontsize"] = tsize
    plt.rcParams["xtick.direction"] = tdir
    plt.rcParams["ytick.direction"] = tdir
    plt.rcParams["xtick.major.size"] = major
    plt.rcParams["xtick.minor.size"] = minor
    plt.rcParams["ytick.major.size"] = 3.0
    plt.rcParams["ytick.minor.size"] = 1.0
    plt.rcParams["axes.linewidth"] = lwidth
    plt.rcParams["legend.handlelength"] = lhandle
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.axisbelow"] = True
    return None


def fig_final():
    fig, axes = plt.subplots(7, 5, figsize=(7.5, 10.208))
    k = 0
    s = 1
    dfpos = pd.read_excel("results/results_incidence_pos.xlsx", index_col=[2])
    dfneg = pd.read_excel("results/results_incidence_neg.xlsx", index_col=[2])
    data = feather.read_feather("processed_data/GBD data incidence.feather").swaplevel()

    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    for level in dfpos.index[:35]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(data.loc[level].stack()), index=energy.stack().index)[0]
        x = x0 * y0 / y0
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x = x.dropna()
        y = y0 * x0 / x0
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        y = y.dropna()

        ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
        xy = np.vstack([x, y])

        z = gaussian_kde(xy)(xy)
        x_lin = np.linspace(x.min(), x.max(), 100)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=s, label="Both sexes")

        df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
        df = df.loc[df["x"] > x.quantile(0.05)]
        # df = df.loc[df["x"] < x.quantile(0.9)]

        x = df["x"]
        y = df["y"]

        x_lin = np.linspace(x.quantile(0.05), x.quantile(0.995), 100)

        # regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * dfpos.loc[level, "slope"] + dfpos.loc[level, "intercept"], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()
    # plt.savefig("Positive1.pdf")
    plt.savefig("results/Positive1.png", dpi=200)
    # plt.savefig("Positive1.svg")

    fig, axes = plt.subplots(7, 5, figsize=(7.5, 10.208))
    k = 0
    s = 1
    for level in dfneg.index[:35]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(data.loc[level].stack()), index=energy.stack().index)[0]
        x = x0 * y0 / y0
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x = x.dropna()
        y = y0 * x0 / x0
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        y = y.dropna()

        ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
        xy = np.vstack([x, y])

        z = gaussian_kde(xy)(xy)
        x_lin = np.linspace(x.min(), x.max(), 100)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=s, label="Both sexes")

        df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
        df = df.loc[df["x"] > x.quantile(0.05)]
        # df = df.loc[df["x"] < x.quantile(0.9)]

        x = df["x"]
        y = df["y"]

        x_lin = np.linspace(x.quantile(0.05), x.quantile(0.995), 100)

        # regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * dfneg.loc[level, "slope"] + dfneg.loc[level, "intercept"], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()
    # plt.savefig("Negative1_17-01-2023.pdf")
    plt.savefig("results/Negative1.png", dpi=200)
    # plt.savefig("Negative1_17-01-2023.svg")

    fig, axes = plt.subplots(2, 5, figsize=(7.5, 10.208 * 2.1 / 7))
    k = 0
    s = 1
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    for level in dfneg.index[35:]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(data.loc[level].stack()), index=energy.stack().index)[0]
        x = x0 * y0 / y0
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x = x.dropna()
        y = y0 * x0 / x0
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        y = y.dropna()

        ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
        xy = np.vstack([x, y])

        z = gaussian_kde(xy)(xy)
        x_lin = np.linspace(x.min(), x.max(), 100)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=s, label="Both sexes")

        df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
        df = df.loc[df["x"] > x.quantile(0.05)]
        # df = df.loc[df["x"] < x.quantile(0.9)]

        x = df["x"]
        y = df["y"]

        x_lin = np.linspace(x.quantile(0.05), x.quantile(0.995), 100)

        # regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * dfneg.loc[level, "slope"] + dfneg.loc[level, "intercept"], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()
    # plt.savefig("Negative2_17-01-2023.pdf")
    plt.savefig("results/Negative2.png", dpi=200)
    # plt.savefig("Negative2_17-01-2023.svg")

    fig, axes = plt.subplots(5, 5, figsize=(7.5, 10.208 * 5 / 7))
    k = 0
    s = 1
    for level in dfpos.index[35:]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(data.loc[level].stack()), index=energy.stack().index)[0]
        x = x0 * y0 / y0
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x = x.dropna()
        y = y0 * x0 / x0
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        y = y.dropna()

        ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
        xy = np.vstack([x, y])

        z = gaussian_kde(xy)(xy)
        x_lin = np.linspace(x.min(), x.max(), 100)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax.scatter(x, y, c=z, s=s, label="Both sexes")

        df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
        df = df.loc[df["x"] > x.quantile(0.05)]
        # df = df.loc[df["x"] < x.quantile(0.9)]

        x = df["x"]
        y = df["y"]

        x_lin = np.linspace(x.quantile(0.05), x.quantile(0.995), 100)

        # regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * dfpos.loc[level, "slope"] + dfpos.loc[level, "intercept"], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()
    # plt.savefig("Positive2_17-01-2023.pdf")
    plt.savefig("results/Positive2.png", dpi=200)
    # plt.savefig("Positive2_17-01-2023.svg")


def fig_appendix():
    for indicator in ["incidence", "YLDs", "YLLs"]:
        data = feather.read_feather("processed_data/GBD data " + indicator + ".feather").swaplevel()
        matplotlib.rc("xtick", labelsize=6)
        matplotlib.rc("ytick", labelsize=6)

        df_inc = (
            pd.read_excel("results/data_for_ma_" + indicator + ".xlsx", sheet_name="all_years", index_col=[0, 1, 7])
            .rename(index=dictmod)
            .reorder_levels([1, 0, 2])
        )

        first_level = 0
        last_level = 35
        for i in range(0, 1, 1):
            fig, axes = plt.subplots(7, 5, figsize=(7.5, 10.208))
            k = 0
            s = 1
            fig.suptitle(indicator)
            for level in df_inc.reset_index()["cause_name"].values[first_level:last_level]:
                code = df_inc.loc[level].index[0][0]
                ax = axes[k // 5, k % 5]
                ax.set_xlim([-1, 8])

                x0 = np.log(energy.stack(dropna=False))
                y0 = pd.DataFrame(np.log(data.loc[level].stack()), index=energy.stack().index)[0]
                x = x0 * y0 / y0
                x.replace([np.inf, -np.inf], np.nan, inplace=True)
                x = x.dropna()
                y = y0 * x0 / x0
                y.replace([np.inf, -np.inf], np.nan, inplace=True)
                y = y.dropna()

                ax.set_ylim([y.min() - 0.1, y.max() + 0.1])
                xy = np.vstack([x, y])

                z = gaussian_kde(xy)(xy)
                x_lin = np.linspace(x.min(), x.max(), 100)
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]
                ax.scatter(x, y, c=z, s=s, label="Both sexes")

                df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
                df = df.loc[df["x"] > x.quantile(0.05)]
                # df = df.loc[df["x"] < x.quantile(0.9)]

                x = df["x"]
                y = df["y"]

                x_lin = np.linspace(x.quantile(0.05), x.quantile(0.995), 100)

                # regression = scipy.stats.linregress(x, y)

                ax.plot(
                    x_lin,
                    x_lin * df_inc.loc[level, "slope"].values + df_inc.loc[level, "intercept"].values,
                    color="black",
                )
                if len(level) > 20:
                    ax.set_title(code + " - " + level[: 17 - len(code)] + "[...]", fontsize=8)
                else:
                    ax.set_title(code + " - " + level, fontsize=8)

                # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

                k += 1
            plt.tight_layout()
            plt.savefig("results/Appendix/" + indicator + str(first_level / 35) + ".png", dpi=200)
            first_level += 35
            last_level += 35


def forest():
    data_forest = pd.DataFrame()
    for ind in ["YLDs", "incidence", "prevalence", "YLLs"]:
        df = pd.read_excel("ma_results_" + ind + ".xlsx", index_col=0).rename(index=dictmod)
        df.loc["year"] = [i[1:5] for i in df.columns]
        df.loc["indicator"] = [i[6:] for i in df.columns]
        df = df.T.reset_index().set_index(["year", "indicator"]).drop("index", axis=1).T
        data_forest[ind] = df.unstack()
    df2 = data_forest.unstack().T.rename(index=dictmod)
    df2 = df2["ll_y"].loc[["incidence", "YLDs", "YLLs"]]
    df2 = df2.unstack(level=0).sort_values(by=("estimate", "incidence"), ascending=False).stack().swaplevel()
    df2["label"] = "      " + df2.index.get_level_values(level=0)
    df2["group"] = df2.index.get_level_values(level=1)
    df2 = df2.loc[["incidence", "YLDs", "YLLs"]]
    matplotlib.rcdefaults()
    ax = fp.forestplot(
        df2,  # the dataframe with results data
        estimate="estimate",  # col containing estimated effect size
        ll="ci.lb",
        hl="ci.ub",  # columns containing conf. int. lower and higher limits
        varlabel="label",  # column containing variable label
        # pval="pval",
        groupvar="group",
        xlabel="Pearson correlation",  # x-label title
        capitalize="capitalize",
        # rightannote=["formatted_pval"],  # columns to report on right of plot
        # right_annoteheaders=["P-value"],  # ^corresponding headers
        # sort=True,
        figsize=(6, 22),
        color_alt_rows=True,
        ylabel="Est.(95% Conf. Int.)",
        **{"markercolor": ["darkorange"] * 2 + ["darkgray", "darkorange", "darkblue", "black"] * 21, "ylabel1_size": 11}
    )
    ax.tick_params(axis="y", pad=500)
    lines = ax.get_lines()
    for line in lines[1:3]:
        line.set_xdata([-1.73, -0.43])
    plt.savefig("ma2.pdf", bbox_inches="tight")
    plt.savefig("ma2.svg", bbox_inches="tight")
    plt.show()


def forest_sect():

    for mod in mod1:
        data_forest = pd.DataFrame()
        for ind in ["incidence", "YLDs", "YLLs"]:
            df = pd.read_excel("ma_results_" + ind + ".xlsx", index_col=0)
            df.loc["year"] = [i[1:5] for i in df.columns]
            df.loc["indicator"] = [i[6:] for i in df.columns]
            df = df.T.reset_index().set_index(["year", "indicator"]).drop("index", axis=1).T.drop("ll_y", axis=1)
            data_forest[ind] = df.unstack()
        df3 = data_forest.unstack().T
        df3 = df3.swaplevel().loc[dictmod[mod]].stack(level=0)

        df3["label"] = "      " + df3.index.get_level_values(level=1)
        df3["group"] = df3.index.get_level_values(level=0)

        matplotlib.rcdefaults()
        ax = fp.forestplot(
            df3,  # the dataframe with results data
            estimate="estimate",  # col containing estimated effect size
            ll="ci.lb",
            hl="ci.ub",  # columns containing conf. int. lower and higher limits
            varlabel="label",  # column containing variable label
            # pval="pval",
            groupvar="group",
            xlabel="Pearson correlation",  # x-label title
            capitalize="capitalize",
            # rightannote=["formatted_pval"],  # columns to report on right of plot
            # right_annoteheaders=["P-value"],  # ^corresponding headers
            # sort=True,
            figsize=(6, 22),
            color_alt_rows=True,
            ylabel=dictmod[mod],
            **{
                "markercolor": "black",
            }
        )
        ax.tick_params(axis="y", pad=250)
        lines = ax.get_lines()
        for line in lines[1:3]:
            line.set_xdata([-1, -0.5])
        plt.savefig("results/" + dictmod[mod][:4] + ".pdf", bbox_inches="tight")
        plt.savefig("results/" + dictmod[mod][:4] + ".png", bbox_inches="tight")
        plt.show()
