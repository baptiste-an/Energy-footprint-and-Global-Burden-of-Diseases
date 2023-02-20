import pandas as pd
import pyarrow.feather as feather
import numpy as np
from scipy import stats

from data_processing import rename_region

footprint = rename_region(feather.read_feather("processed_data/energy footprint.feather"), level="Country Name")

pop = rename_region(
    pd.read_csv("raw_data/World Bank/pop.csv", header=[2], index_col=[0])[[str(i) for i in range(1990, 2020, 1)]],
    level="Country Name",
).drop("Z - Aggregated categories")
pop.columns = [int(i) for i in pop.columns]
pop.loc["Taiwan"] = pd.read_excel("raw_data/pop Taiwan.xls", header=0, index_col=0)["TW"]

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


GJcap_world = (
    footprint.drop(
        [
            "Antigua",
            "Z - Aggregated categories",
            "Gaza Strip",
            "Netherlands Antilles",
            "United Arab Emirates",
        ]
    ).sum()
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
    ].sum()
    * 1000
).drop([2020, 2021])


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


index_negative = [
    "Typhoid fever",
    "Rabies",
    "Malaria",
    "Tetanus",
    "Diphtheria",
    "HIV-AIDS - Drug-susceptible Tuberculosis",
    "Invasive Non-typhoidal Salmonella (iNTS)",
    "Multidrug-resistant tuberculosis without extensive drug resistance",
    "Drug-susceptible tuberculosis",
    "Yellow fever",
    "Leprosy",
    "Acute hepatitis B",
    "Meningitis",
    "Syphilis",
    "Whooping cough",
    "Lower respiratory infections",
    "Diarrheal diseases",
    "Acute hepatitis C",
    "Encephalitis",
    "Acute hepatitis E",
    "Trichomoniasis",
    "Acute hepatitis A",
    ##
    "Maternal hypertensive disorders",
    "Maternal hemorrhage",
    "Maternal sepsis and other maternal infections",
    "Maternal abortion and miscarriage",
    "Ectopic pregnancy",
    ##
    "Hemolytic disease and other neonatal jaundice",
    "Neonatal encephalopathy due to birth asphyxia and trauma",
    "Neonatal sepsis and other neonatal infections",
    "Neonatal preterm birth",
    ##
    "Vitamin A deficiency",
    ##
    "Neural tube defects",
    "Congenital heart anomalies",
    "Hemoglobinopathies and hemolytic anemias",
    "Endometriosis",
    ##
    "Rheumatic heart disease",
    "Intracerebral hemorrhage",
    ##
    "Dysthymia",
    ##
    "Dermatitis",
    "Scabies",
    ##
    "Non-venomous animal contact",
]


index_positive = [
    "Testicular cancer",
    "Malignant skin melanoma",
    "Non-melanoma skin cancer (basal-cell carcinoma)",
    "Non-melanoma skin cancer (squamous-cell carcinoma)",
    "Kidney cancer",
    "Benign and in situ intestinal neoplasms",
    "Chronic lymphoid leukemia",
    "Myelodysplastic, myeloproliferative, and other hematopoietic neoplasms",
    "Colon and rectum cancer",
    "Acute lymphoid leukemia",
    "Brain and central nervous system cancer",
    "Non-Hodgkin lymphoma",
    "Uterine cancer",
    "Hodgkin lymphoma",
    "Thyroid cancer",
    "Bladder cancer",
    "Multiple myeloma",
    "Pancreatic cancer",
    "Breast cancer",
    "Tracheal, bronchus, and lung cancer",
    "Prostate cancer",
    "Ovarian cancer",
    "Acute myeloid leukemia",
    "Other malignant neoplasms",
    ##
    "Osteoarthritis hip",
    ##
    "Digestive congenital anomalies",
    "Down syndrome",
    "Urinary diseases and male infertility",
    ##
    "Non-rheumatic calcific aortic valve disease",
    "Endocarditis",
    "Peripheral artery disease",
    ##
    "Inflammatory bowel disease",
    "Gallbladder and biliary diseases",
    "Vascular intestinal disorders",
    ##
    "Parkinson's disease",
    "Multiple sclerosis",
    "Motor neuron disease",
    "Tension-type headache",
    ##
    "Schizophrenia",
    "Cocaine use disorders",
    "Cannabis use disorders",
    "Anorexia nervosa",
    "Bulimia nervosa",
    "Other drug use disorders",
    "Attention-deficit/hyperactivity disorder",
    "Chronic kidney disease due to hypertension",
    ##
    "Decubitus ulcer",
    "Psoriasis",
    "Other skin and subcutaneous diseases",
    ##
    "Falls",
    "Poisoning by carbon monoxide",
    "Other exposure to mechanical forces",
    "Poisoning by other means",
    "Other unintentional injuries",
    "Adverse effects of medical treatment",
]


def data_for_meta_analysis():
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
                        if abs(regression[2]) != 1:
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
