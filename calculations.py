import pandas as pd
import pyarrow.feather as feather
import numpy as np
from scipy import stats

from data_processing import rename_region

# energy data
footprint = rename_region(feather.read_feather("processed_data/energy footprint.feather"), level="Country Name")

# population data
pop = rename_region(
    pd.read_csv("raw_data/World Bank/pop.csv", header=[2], index_col=[0])[[str(i) for i in range(1990, 2020, 1)]],
    level="Country Name",
).drop("Z - Aggregated categories")
pop.columns = [int(i) for i in pop.columns]
pop.loc["Taiwan"] = pd.read_excel("raw_data/pop Taiwan.xls", header=0, index_col=0)["TW"]

# energy per capita
GJcap = (
    footprint.drop(
        [
            "Antigua",
            "Z - Aggregated categories",
            "Gaza Strip",
            "Netherlands Antilles",
            "United Arab Emirates",
        ]
    )
    / pop.loc[
        footprint.drop(
            [
                "Antigua",
                "Z - Aggregated categories",
                "Gaza Strip",
                "Netherlands Antilles",
                "United Arab Emirates",
            ]
        ).index
    ]
    * 1000
).drop([2020, 2021], axis=1)

# energy per capita without regions not in GBD
energy = GJcap.drop(
    [
        "Aruba",
        "British Virgin Islands",
        "Cayman Islands",
        "French Polynesia",
        "Hong Kong",
        "Liechtenstein",
        "Macau",
        "New Caledonia",
    ]
)


def data_for_meta_analysis():
    """Calculates regressions for the whole 1990-2019 period.

    Saves it in "results/data_for_ma_indicator.xlsx" and "results/result_all.xlsx"

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    codebook = (
        (pd.read_excel("processed_data/codebook_short.xlsx", index_col=1).sort_index())
        # .loc[feather.read_feather("GBD data 2/GBD data incidence.feather").unstack(level=0).index]
        .reset_index()
        .set_index("Cause Outline")
        .sort_values(by="Cause Outline")
    )
    energy = GJcap.drop(
        [
            "Aruba",
            "British Virgin Islands",
            "Cayman Islands",
            "French Polynesia",
            "Hong Kong",
            "Liechtenstein",
            "Macau",
            "New Caledonia",
        ]
    )
    result_all = pd.DataFrame()
    for ind in ["incidence", "prevalence", "YLDs", "YLLs"]:
        slope = pd.DataFrame()
        intercept = pd.DataFrame()
        rvalue = pd.DataFrame()
        pvalue = pd.DataFrame()
        stderr = pd.DataFrame()
        ignored = []
        for level in codebook["cause_name"].values:
            try:

                indicator = (
                    feather.read_feather("processed_data/GBD data " + ind + ".feather").swaplevel().loc[level].stack()
                )

                try:
                    x0 = np.log(energy.stack(dropna=False))
                    y0 = pd.DataFrame(np.log(indicator), index=energy.stack(dropna=False).index)[0]
                    x = x0 * y0 / y0
                    x.replace([np.inf, -np.inf], np.nan, inplace=True)
                    x = x.dropna()
                    y = y0 * x0 / x0
                    y.replace([np.inf, -np.inf], np.nan, inplace=True)
                    y = y.dropna()

                    df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
                    df = df.loc[df["x"] > x.quantile(0.05)]
                    # df = df.loc[df["x"] < x.quantile(0.9)]

                    x = df["x"]
                    y = df["y"]

                    regression = stats.linregress(x, y)

                    rvalue = rvalue.append(pd.DataFrame([regression[2]], index=[level]))
                    pvalue = pvalue.append(pd.DataFrame([regression[3]], index=[level]))
                    stderr = stderr.append(pd.DataFrame([regression[4]], index=[level]))
                    slope = slope.append(pd.DataFrame([regression[0]], index=[level]))
                    intercept = intercept.append(pd.DataFrame([regression[1]], index=[level]))
                except ValueError:
                    ignored.append(level)
            except KeyError:
                ignored.append(level)

        mod_excel = pd.read_excel("mod_dict.xlsx")
        dict_mod1 = dict(zip(list(mod_excel["Disease"].values), list(mod_excel["mod"].values)))

        dict_name = dict(zip(list(codebook["cause_name"].values), list(codebook.index.values)))
        result = pd.concat(
            [rvalue[0], stderr[0], pvalue[0], slope[0], intercept[0]],
            keys=["rvalue", "stderr", "pvalue", "slope", "intercept"],
            axis=1,
        ).T
        result.loc["Code"] = result.rename(columns=dict_name).columns.values
        result.columns.name = "Disease"
        result = result.T.reset_index().set_index(["Code", "Disease"])
        result = pd.DataFrame(result, index=codebook.reset_index().set_index(["Cause Outline", "cause_name"]).index)

        df = result
        df["mod"] = df.rename(index=dict_mod1).rename(index=dictmod).index.get_level_values(level="cause_name")
        df = df[df["mod"] != "z"]
        df.drop(df[df.isna().any(axis=1)].index).to_excel(
            "results/data_for_ma_" + ind + ".xlsx", merge_cells=False, sheet_name="all_years"
        )

        result_all[ind] = result.stack(dropna=False)
    result_all.to_excel("results/result_all.xlsx")


def data_for_meta_analysis_all_years():
    """Calculates regressions for each year during 1990-2019 period.

    Adds yearly results in sheets from "results/data_for_ma_indicator.xlsx"

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    energy = GJcap.drop(
        [
            "Aruba",
            "British Virgin Islands",
            "Cayman Islands",
            "French Polynesia",
            "Hong Kong",
            "Liechtenstein",
            "Macau",
            "New Caledonia",
        ]
    )
    codebook = (
        (pd.read_excel("processed_data/codebook_short.xlsx", index_col=1).sort_index())
        # .loc[feather.read_feather("GBD data 2/GBD data incidence.feather").unstack(level=0).index]
        .reset_index()
        .set_index("Cause Outline")
        .sort_values(by="Cause Outline")
    )
    for year in range(1990, 2020, 1):
        # result_all = pd.DataFrame()

        for ind in ["incidence", "prevalence", "YLDs", "YLLs"]:
            slope = pd.DataFrame()
            intercept = pd.DataFrame()
            rvalue = pd.DataFrame()
            pvalue = pd.DataFrame()
            stderr = pd.DataFrame()
            ignored = []
            for level in codebook["cause_name"].values:
                try:

                    indicator = (
                        feather.read_feather("processed_data/GBD data " + ind + ".feather").swaplevel().loc[level][year]
                    )

                    try:
                        x0 = np.log(energy[year])
                        y0 = pd.DataFrame(np.log(indicator), index=energy[year].index)[year]
                        x = x0 * y0 / y0
                        x.replace([np.inf, -np.inf], np.nan, inplace=True)
                        x = x.dropna()
                        y = y0 * x0 / x0
                        y.replace([np.inf, -np.inf], np.nan, inplace=True)
                        y = y.dropna()

                        df = pd.concat([x, y], keys=["x", "y"]).unstack(level=0).sort_values(by="x")
                        df = df.loc[df["x"] > x.quantile(0.05)]
                        # df = df.loc[df["x"] < x.quantile(0.9)]

                        x = df["x"]
                        y = df["y"]

                        regression = stats.linregress(x, y)
                        if abs(regression[3]) != 0:
                            rvalue = rvalue.append(pd.DataFrame([regression[2]], index=[level]))
                            pvalue = pvalue.append(pd.DataFrame([regression[3]], index=[level]))
                            stderr = stderr.append(pd.DataFrame([regression[4]], index=[level]))
                            slope = slope.append(pd.DataFrame([regression[0]], index=[level]))
                            intercept = intercept.append(pd.DataFrame([regression[1]], index=[level]))
                    except ValueError:
                        ignored.append(level)
                except KeyError:
                    ignored.append(level)

            mod_excel = pd.read_excel("mod_dict.xlsx")
            dict_mod1 = dict(zip(list(mod_excel["Disease"].values), list(mod_excel["mod"].values)))

            dict_name = dict(zip(list(codebook["cause_name"].values), list(codebook.index.values)))
            result = pd.concat(
                [rvalue[0], stderr[0], pvalue[0], slope[0], intercept[0]],
                keys=["rvalue", "stderr", "pvalue", "slope", "intercept"],
                axis=1,
            ).T
            result.loc["Code"] = result.rename(columns=dict_name).columns.values
            result.columns.name = "Disease"
            result = result.T.reset_index().set_index(["Code", "Disease"])
            result = pd.DataFrame(result, index=codebook.reset_index().set_index(["Cause Outline", "cause_name"]).index)

            df = result
            df["mod"] = df.rename(index=dict_mod1).rename(index=dictmod).index.get_level_values(level="cause_name")
            df = df[df["mod"] != "z"]
            with pd.ExcelWriter("results/data_for_ma_" + ind + ".xlsx", mode="a") as writer:
                df.drop(df[df.isna().any(axis=1)].index).to_excel(writer, merge_cells=False, sheet_name=str(year))


mod1 = [
    "HIV/AIDS and sexually transmitted infections",
    "Respiratory infections and tuberculosis",
    "Enteric infections",
    "Neglected tropical diseases and malaria",
    "Other infectious diseases",
    "Maternal and neonatal disorders",
    "Nutritional deficiencies",
    "Neoplasms",
    "Cardiovascular diseases",
    "Chronic respiratory diseases",
    "Digestive diseases",
    "Neurological disorders",
    "Mental disorders",
    "Substance use disorders",
    "Diabetes and kidney diseases",
    "Skin and subcutaneous diseases",
    "Sense organ diseases",
    "Musculoskeletal disorders",
    "Other non-communicable diseases",
    "Transport injuries",
    "Unintentional injuries",
    "Self-harm and interpersonal violence",
]
mod2 = [
    "A.1 - HIV/AIDS and sexually trans. infections",
    "A.2 - Respiratory infections and tuberculosis",
    "A.3 - Enteric infections",
    "A.4 - Neglected tropical diseases and malaria",
    "A.5 - Other infectious diseases",
    "A.6 - Maternal and neonatal disorders",
    "A.7 - Nutritional deficiencies",
    "B.1 - Neoplasms",
    "B.2 - Cardiovascular diseases",
    "B.3 - Chronic respiratory diseases",
    "B.4 - Digestive diseases",
    "B.5 - Neurological disorders",
    "B.6 - Mental disorders",
    "B.7 - Substance use disorders",
    "B.8 - Diabetes and kidney diseases",
    "B.9 - Skin and subcutaneous diseases",
    "B.10 - Sense organ diseases",
    "B.11 - Musculoskeletal disorders",
    "B.12 - Other non-communicable diseases",
    "C.1 - Transport injuries",
    "C.2 - Unintentional injuries",
    "C.3 - Self-harm and interpersonal violence",
]
dictmod = dict(zip(mod1, mod2))


def tables():
    """From "results/data_for_ma_indicator.xlsx", exports incidence data with positive correlation and negative correlation to "results/results_incidence_neg.xlsx" and "results/results_incidence_pos.xlsx"

    These files will be used for the main plots of the paper.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    indicator = "incidence"
    df = pd.read_excel(
        "results/data_for_ma_" + indicator + ".xlsx", sheet_name="all_years", index_col=[0, 1, 7]
    ).rename(index=dictmod)
    df.index.names = ["Code", "Disease", "mod"]
    df2 = df[df["rvalue"] ** 2 > 0.2]
    dfpos = df2[df["slope"] >= 0]
    dfneg = df2[df["slope"] <= 0]

    dfneg = (
        dfneg.reset_index()
        .set_index(["mod", "slope"])
        .sort_index(ascending=True)
        .reset_index()
        .set_index(["mod", "Code", "Disease"])
    )
    dfneg["R2"] = dfneg["rvalue"] ** 2
    dfneg.to_excel("results/results_" + indicator + "_neg.xlsx")
    dfpos = (
        dfpos.reset_index()
        .set_index(["mod", "slope"])
        .sort_index(ascending=False)
        .reset_index()
        .set_index(["mod", "Code", "Disease"])
    )
    dfpos["R2"] = dfpos["rvalue"] ** 2
    dfpos.to_excel("results/results_" + indicator + "_pos.xlsx")
