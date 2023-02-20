import pandas as pd
import requests
import pyarrow.feather as feather
import country_converter as coco
import numpy as np
import matplotlib.pyplot as plt
import pymrio
import scipy.io
from sklearn.linear_model import LinearRegression
from matplotlib import colors as mcolors
import math
import country_converter as coco
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.metrics import r2_score
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio
import zipfile
from scipy.stats import gaussian_kde
import scipy.optimize as optimize
import pwlf
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import forestplot as fp


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


plt_rcParams()

dict_regions = dict()  # create a dict that will be used to rename regions
cc = coco.CountryConverter(
    include_obsolete=True
)  # documentation for coco here: https://github.com/konstantinstadler/country_converter
for i in [
    n for n in cc.valid_class if n != "name_short"
]:  # we convert all the regions in cc to name short and add it to the dict
    dict_regions.update(cc.get_correspondence_dict(i, "name_short"))
name_short = cc.ISO3as("name_short")["name_short"].values  # array containing all region names in short_name format


def dict_regions_update():
    """Adds to dict the encountered region names that were not in coco.

    If a region is wider than a country (for example "European Union"), it is added to "Z - Aggregated categories" in order to be deleted later.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dict_regions["Bolivia (Plurinational State of)"] = "Bolivia"
    dict_regions["Czechia"] = "Czech Republic"
    dict_regions["Iran (Islamic Republic of)"] = "Iran"
    dict_regions["China, Taiwan Province of China"] = "Taiwan"
    dict_regions["Congo"] = "Congo Republic"
    dict_regions["Venezuela (Bolivarian Republic of)"] = "Venezuela"
    dict_regions["Dem. People's Republic of Korea"] = "North Korea"
    dict_regions["Bahamas, The"] = "Bahamas"
    dict_regions["Congo, Dem. Rep."] = "DR Congo"
    dict_regions["Congo, Rep."] = "Congo Republic"
    dict_regions["Egypt, Arab Rep."] = "Egypt"
    dict_regions["Faroe Islands"] = "Faeroe Islands"
    dict_regions["Gambia, The"] = "Gambia"
    dict_regions["Hong Kong SAR, China"] = "Hong Kong"
    dict_regions["Iran, Islamic Rep."] = "Iran"
    dict_regions["Korea, Dem. People's Rep."] = "North Korea"
    dict_regions["Korea, Rep."] = "South Korea"
    dict_regions["Lao PDR"] = "Laos"
    dict_regions["Macao SAR, China"] = "Macau"
    dict_regions["North Macedonia"] = "Macedonia"
    dict_regions["Russian Federation"] = "Russia"
    dict_regions["Sint Maarten (Dutch part)"] = "Sint Maarten"
    dict_regions["Slovak Republic"] = "Slovakia"
    dict_regions["St. Martin (French part)"] = "Saint-Martin"
    dict_regions["Syrian Arab Republic"] = "Syria"
    dict_regions["Virgin Islands (U.S.)"] = "United States Virgin Islands"
    dict_regions["West Bank and Gaza"] = "Palestine"
    dict_regions["Yemen, Rep."] = "Yemen"
    dict_regions["Venezuela, RB"] = "Venezuela"
    dict_regions["Brunei"] = "Brunei Darussalam"
    dict_regions["Cape Verde"] = "Cabo Verde"
    dict_regions["Dem. People's Rep. Korea"] = "North Korea"
    dict_regions["Swaziland"] = "Eswatini"
    dict_regions["Taiwan, China"] = "Taiwan"
    dict_regions["Virgin Islands"] = "United States Virgin Islands"
    dict_regions["Yemen, PDR"] = "Yemen"
    dict_regions["Réunion"] = "Reunion"
    dict_regions["Saint Helena"] = "St. Helena"
    dict_regions["China, Hong Kong SAR"] = "Hong Kong"
    dict_regions["China, Macao SAR"] = "Macau"
    dict_regions["Bonaire, Sint Eustatius and Saba"] = "Bonaire, Saint Eustatius and Saba"
    dict_regions["Curaçao"] = "Curacao"
    dict_regions["Saint Barthélemy"] = "St. Barths"
    dict_regions["Saint Martin (French part)"] = "Saint-Martin"
    dict_regions["Micronesia (Fed. States of)"] = "Micronesia, Fed. Sts."
    dict_regions["Micronesia, Federated State=s of"] = "Micronesia, Fed. Sts."
    dict_regions["Bonaire"] = "Bonaire, Saint Eustatius and Saba"
    dict_regions["São Tomé and Principe"] = "Sao Tome and Principe"
    dict_regions["Virgin Islands, British"] = "British Virgin Islands"
    dict_regions["Wallis and Futuna"] = "Wallis and Futuna Islands"
    dict_regions["Micronesia, Federated States of"] = "Micronesia, Fed. Sts."

    dict_regions["VIR"] = "United States Virgin Islands"
    dict_regions["GMB"] = "Gambia"
    dict_regions["NAM"] = "Namibia"
    dict_regions["BHS"] = "Bahamas"
    dict_regions["The Bahamas"] = "Bahamas"
    dict_regions["The Gambia"] = "Gambia"
    dict_regions["Virgin Islands, U.S."] = "United States Virgin Islands"
    dict_regions["Congo, DRC"] = "DR Congo"
    dict_regions["Marshall Is."] = "Marshall Islands"
    dict_regions["Solomon Is."] = "Solomon Islands"
    dict_regions["Timor Leste"] = "Timor-Leste"

    dict_regions["Cote dIvoire"] = "Cote d'Ivoire"
    dict_regions["Macao SAR"] = "Macau"
    dict_regions["TFYR Macedonia"] = "Macedonia"
    dict_regions["UAE"] = "United Arab Emirates"
    dict_regions["UK"] = "United Kingdom"
    dict_regions["Gaza Strip"] = "Gaza Strip"
    dict_regions["Antigua"] = "Antigua"
    dict_regions["Taiwan (Province of China)"] = "Taiwan"
    dict_regions["Micronesia (Federated States of)"] = "Micronesia, Fed. Sts."

    dict_regions["Turkiye"] = "Turkey"

    for j in [
        "Africa Eastern and Southern",
        "Africa Western and Central",
        "Arab World",
        "Caribbean small states",
        "Central Europe and the Baltics",
        "Early-demographic dividend",
        "East Asia & Pacific",
        "East Asia & Pacific (excluding high income)",
        "East Asia & Pacific (IDA & IBRD countries)",
        "Euro area",
        "Europe & Central Asia",
        "Europe & Central Asia (excluding high income)",
        "Europe & Central Asia (IDA & IBRD countries)",
        "European Union",
        "Fragile and conflict affected situations",
        "Heavily indebted poor countries (HIPC)",
        "High income",
        "IBRD only",
        "IDA & IBRD total",
        "IDA blend",
        "IDA only",
        "IDA total",
        "Late-demographic dividend",
        "Latin America & Caribbean",
        "Latin America & Caribbean (excluding high income)",
        "Latin America & the Caribbean (IDA & IBRD countries)",
        "Least developed countries: UN classification",
        "Low & middle income",
        "Low income",
        "Lower middle income",
        "Middle East & North Africa",
        "Middle East & North Africa (excluding high income)",
        "Middle East & North Africa (IDA & IBRD countries)",
        "Middle income",
        "North America",
        "Not classified",
        "OECD members",
        "Other small states",
        "Pacific island small states",
        "Post-demographic dividend",
        "Pre-demographic dividend",
        "Small states",
        "South Asia",
        "South Asia (IDA & IBRD)",
        "Sub-Saharan Africa",
        "Sub-Saharan Africa (excluding high income)",
        "Sub-Saharan Africa (IDA & IBRD countries)",
        "Upper middle income",
        "World",
        "Arab League states",
        "China and India",
        "Czechoslovakia",
        "East Asia & Pacific (all income levels)",
        "East Asia & Pacific (IDA & IBRD)",
        "East Asia and the Pacific (IFC classification)",
        "EASTERN EUROPE",
        "Europe & Central Asia (all income levels)",
        "Europe & Central Asia (IDA & IBRD)",
        "Europe and Central Asia (IFC classification)",
        "European Community",
        "High income: nonOECD",
        "High income: OECD",
        "Latin America & Caribbean (all income levels)",
        "Latin America & Caribbean (IDA & IBRD)",
        "Latin America and the Caribbean (IFC classification)",
        "Low income, excluding China and India",
        "Low-income Africa",
        "Middle East & North Africa (all income levels)",
        "Middle East & North Africa (IDA & IBRD)",
        "Middle East (developing only)",
        "Middle East and North Africa (IFC classification)",
        "Other low-income",
        "Serbia and Montenegro",
        "Severely Indebted",
        "South Asia (IFC classification)",
        "Sub-Saharan Africa (all income levels)",
        "SUB-SAHARAN AFRICA (excl. Nigeria)",
        "Sub-Saharan Africa (IDA & IBRD)",
        "Sub-Saharan Africa (IFC classification)",
        "WORLD",
        "UN development groups",
        "More developed regions",
        "Less developed regions",
        "Least developed countries",
        "Less developed regions, excluding least developed countries",
        "Less developed regions, excluding China",
        "Land-locked Developing Countries (LLDC)",
        "Small Island Developing States (SIDS)",
        "World Bank income groups",
        "High-income countries",
        "Middle-income countries",
        "Upper-middle-income countries",
        "Lower-middle-income countries",
        "Low-income countries",
        "No income group available",
        "Geographic regions",
        "Latin America and the Caribbean",
        "Sustainable Development Goal (SDG) regions",
        "SUB-SAHARAN AFRICA",
        "NORTHERN AFRICA AND WESTERN ASIA",
        "CENTRAL AND SOUTHERN ASIA",
        "EASTERN AND SOUTH-EASTERN ASIA",
        "LATIN AMERICA AND THE CARIBBEAN",
        "AUSTRALIA/NEW ZEALAND",
        "OCEANIA (EXCLUDING AUSTRALIA AND NEW ZEALAND)",
        "EUROPE AND NORTHERN AMERICA",
        "EUROPE",
        "Holy See",
        "NORTHERN AMERICA",
        "East Asia & Pacific (ICP)",
        "Europe & Central Asia (ICP)",
        "Latin America & Caribbean (ICP)",
        "Middle East & North Africa (ICP)",
        "North America (ICP)",
        "South Asia (ICP)",
        "Sub-Saharan Africa (ICP)",
        "Andean Latin America",
        "Australasia",
        "Central Latin America",
        "Central Sub-Saharan Africa",
        "East Asia",
        "Eastern Sub-Saharan Africa",
        "Global",
        "High-income",
        "High-income Asia Pacific",
        "High-income North America",
        "Latin America and Caribbean",
        "North Africa and Middle East",
        "Southeast Asia",
        "Southern Latin America",
        "Southern Sub-Saharan Africa",
        "Tropical Latin America",
        "Western Sub-Saharan Africa",
        "Central Europe",
        "Oceania",
        "Central Asia",
        "Western Europe",
        "Eastern Europe",
        "Former USSR",
        "Rest of World",
    ]:
        dict_regions[j] = "Z - Aggregated categories"
    return None


dict_regions_update()

# all the regions that do not correspond to a country are in 'Z - Aggregated categories'
# rename the appropriate level of dataframe using dict_regions
def rename_region(df, level="LOCATION"):
    """Renames the regions of a DataFrame into name_short format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose regions must be renamed
    level : string
        Name of the level containing the region names

    Returns
    df : pd.DataFrame
        DataFrame with regions in name_short format
    -------
    None
    """
    if level in df.index.names:
        axis = 0
    else:
        axis = 1
        df = df.T

    index_names = df.index.names
    df = df.reset_index()
    df = df.set_index(level)
    df = df.rename(index=dict_regions)  # rename index according to dict
    ind = df.index.values
    for i in range(0, len(ind), 1):
        if type(ind[i]) == list:
            # if len(ind[i])==0:
            ind[i] = ind[i][0]
    df = df.reindex(ind)
    df = df.reset_index().set_index(index_names)
    for i in df.index.get_level_values(level).unique():
        if i not in name_short and i != "Z - Aggregated categories":
            print(
                i
                + " is not in dict_regions\nAdd it using\n  >>> dict_regions['"
                + i
                + "'] = 'region' # name_short format\n"
            )
    if axis == 1:
        df = df.T
    return df


def footprint():
    df = pd.DataFrame()
    for i in range(1990, 2022, 1):
        url = "https://worldmrio.com/ComputationsM/Phase199/Loop082/leontief/tradereport_" + str(i) + ".txt"
        df = df.append(pd.read_csv(url, index_col=[0, 1, 2, 3, 4, 5], header=0, sep="\t"))
    feather.write_feather(df, "tradereport.feather")

    feather.write_feather(
        df.xs("Energy Usage (TJ)", level="Indicator Description")["Value"]
        .unstack(level="Record")["Footprint"]
        .unstack(level="Year")
        .groupby(level="Country Name")
        .sum(),
        "energy footprint.feather",
    )
    feather.write_feather(
        df.xs("Energy Usage (TJ)", level="Indicator Description")["Value"]
        .unstack(level="Record")["FootprintDirectConsumption"]
        .unstack(level="Year")
        .groupby(level="Country Name")
        .sum(),
        "energy footprint hh.feather",
    )


footprint = rename_region(feather.read_feather("energy footprint.feather"), level="Country Name")

footprint_hh = rename_region(feather.read_feather("energy footprint hh.feather"), level="Country Name")

pop = rename_region(
    pd.read_csv("World Bank/pop.csv", header=[2], index_col=[0])[[str(i) for i in range(1990, 2020, 1)]],
    level="Country Name",
).drop("Z - Aggregated categories")
pop.columns = [int(i) for i in pop.columns]
pop.loc["Taiwan"] = pd.read_excel("pop Taiwan.xls", header=0, index_col=0)["TW"]


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


def data_prevalence():
    df = pd.DataFrame()
    for j in range(1, 27, 1):
        zf = zipfile.ZipFile("GBD data/IHME-GBD_2019_DATA-12f4c182-" + str(j) + ".zip")
        df = df.append(
            pd.read_csv(
                zf.open("IHME-GBD_2019_DATA-12f4c182-" + str(j) + ".csv"),
                index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
        )
    feather.write_feather(df, "GBD data/GBD data.feather")
    feather.write_feather(
        rename_region(
            df.xs("All ages", level="age_name")["val"]
            .unstack(level="year")
            .groupby(level=["location_name", "sex_name", "cause_name"])
            .sum(),
            level="location_name",
        ),
        "GBD data/GBD data all ages.feather",
    )
    feather.write_feather(
        rename_region(
            df.xs("Age-standardized", level="age_name")["val"]
            .unstack(level="year")
            .groupby(level=["location_name", "sex_name", "cause_name"])
            .sum(),
            level="location_name",
        ),
        "GBD data/GBD data age-standardized.feather",
    )


def data_incidence():
    df = pd.DataFrame()
    for j in range(1, 24, 1):
        zf = zipfile.ZipFile("GBD data incidence/IHME-GBD_2019_DATA-b676a33c-" + str(j) + ".zip")
        df = df.append(
            pd.read_csv(
                zf.open("IHME-GBD_2019_DATA-b676a33c-" + str(j) + ".csv"),
                index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
        )
    feather.write_feather(df, "GBD data incidence/GBD data incidence.feather")
    feather.write_feather(
        rename_region(
            df.xs("All ages", level="age_name")["val"]
            .unstack(level="year")
            .groupby(level=["location_name", "sex_name", "cause_name"])
            .sum(),
            level="location_name",
        ),
        "GBD data incidence/GBD data incidence all ages.feather",
    )
    feather.write_feather(
        rename_region(
            df.xs("Age-standardized", level="age_name")["val"]
            .unstack(level="year")
            .groupby(level=["location_name", "sex_name", "cause_name"])
            .sum(),
            level="location_name",
        ),
        "GBD data incidence/GBD data incidence age-standardized.feather",
    )


def data_incidence_world():
    zf = zipfile.ZipFile("GBD data incidence/IHME-GBD_2019_DATA-75201447-1.zip")
    df = pd.read_csv(zf.open("IHME-GBD_2019_DATA-75201447-1.csv"), index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    df = df.reset_index().set_index(["cause_id", "cause_name", "year"])["val"].unstack()
    feather.write_feather(df, "world_incidence.feather")


###### with several fits


def number_of_points():
    agestd = feather.read_feather("GBD data incidence/GBD data incidence age-standardized.feather").rename(
        index={
            "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
            "HIV/AIDS": "HIV-AIDS",
            "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
            "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
            "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
            "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
        }
    )
    (
        pd.DataFrame(
            agestd.xs("Both", level="sex_name").stack().unstack(level="cause_name"), index=energy.stack().index
        )
        > 0.00000000000000001
    ).sum().sort_values(ascending=False).to_excel("number of points.xlsx")
    return None


######## FINAL ##########


def data_final():
    slope = pd.DataFrame()
    R2 = pd.DataFrame()
    pvalue = pd.DataFrame()
    ignored = []

    codebook = (
        pd.read_excel("codebook.xlsx", index_col=5)
        .sort_index()
        .replace(
            {
                "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
                "HIV/AIDS": "HIV-AIDS",
                "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
                "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
                "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
                "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
            }
        )
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
    agestd = feather.read_feather("GBD data incidence/GBD data incidence age-standardized.feather").rename(
        index={
            "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
            "HIV/AIDS": "HIV-AIDS",
            "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
            "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
            "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
            "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
        }
    )

    for level in codebook["Cause Name"].values:
        try:

            prev = agestd.stack().unstack(level="sex_name").reorder_levels([1, 0, 2]).loc[level]

            try:
                x0 = np.log(energy.stack(dropna=False))
                y0 = pd.DataFrame(np.log(prev["Both"]), index=energy.stack().index)["Both"]
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

                regression = scipy.stats.linregress(x, y)

                R2 = R2.append(pd.DataFrame([regression[2] ** 2], index=[level]))
                pvalue = pvalue.append(pd.DataFrame([regression[3]], index=[level]))
                slope = slope.append(pd.DataFrame([regression[0]], index=[level]))
            except ValueError:
                ignored.append(level)
        except KeyError:
            ignored.append(level)
    dict_name = dict(zip(list(codebook["Cause Name"].values), list(codebook.index.values)))
    result = pd.concat([R2[0], pvalue[0], slope[0]], keys=["R2", "pvalue", "slope"], axis=1).T
    result.loc["Code"] = result.rename(columns=dict_name).columns.values
    result.columns.name = "Disease"
    result = result.T.reset_index().set_index(["Code", "Disease"])


# number of regions:
# test = agestd.stack().unstack(level="sex_name").reorder_levels([1, 0, 2]).unstack(level=0)
# pd.DataFrame(test,index=energy.stack().index).unstack().sum(axis=1).sort_values()

###### FIG FINAL #######

agestd = feather.read_feather("GBD data incidence/GBD data incidence age-standardized.feather").rename(
    index={
        "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
        "HIV/AIDS": "HIV-AIDS",
        "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
        "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
        "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
        "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
    }
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


def fig_final():
    fig, axes = plt.subplots(7, 5, figsize=(7.5, 8.75 * 7 / 6))
    k = 0
    s = 1
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    for level in index_positive[:35]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        prev = agestd.stack().unstack(level="sex_name").reorder_levels([1, 0, 2]).loc[level]

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(prev["Both"]), index=energy.stack().index)["Both"]
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

        regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * regression[0] + regression[1], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()
    plt.savefig("Positive1.pdf")
    plt.savefig("Positive1.png", dpi=200)
    plt.savefig("Positive1.svg")

    fig, axes = plt.subplots(7, 5, figsize=(7.5, 8.75 * 7 / 6))
    k = 0
    s = 1
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    for level in index_negative[:35]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        prev = agestd.stack().unstack(level="sex_name").reorder_levels([1, 0, 2]).loc[level]

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(prev["Both"]), index=energy.stack().index)["Both"]
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

        regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * regression[0] + regression[1], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.1)), size=8)

        k += 1
    plt.tight_layout()
    plt.savefig("Negative1.pdf")
    plt.savefig("Negative1.png", dpi=200)
    plt.savefig("Negative1.svg")

    fig, axes = plt.subplots(2, 5, figsize=(7.5, 8.75 * 1 / 3))
    k = 0
    s = 1
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    for level in index_negative[35:]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        prev = agestd.stack().unstack(level="sex_name").reorder_levels([1, 0, 2]).loc[level]

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(prev["Both"]), index=energy.stack().index)["Both"]
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

        regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * regression[0] + regression[1], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.1)), size=8)

        k += 1
    plt.tight_layout()
    plt.savefig("Negative2.pdf")
    plt.savefig("Negative2.png", dpi=200)
    plt.savefig("Negative2.svg")

    fig, axes = plt.subplots(4, 5, figsize=(7.5, 8.75 * 2 / 3))
    k = 0
    s = 1
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    for level in index_positive[35:]:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        prev = agestd.stack().unstack(level="sex_name").reorder_levels([1, 0, 2]).loc[level]

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(prev["Both"]), index=energy.stack().index)["Both"]
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

        regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * regression[0] + regression[1], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()
    plt.savefig("Positive2.pdf")
    plt.savefig("Positive2.png", dpi=200)
    plt.savefig("Positive2.svg")


############################### world data


def world_data():
    slope = pd.DataFrame()
    R2 = pd.DataFrame()
    ignored = []
    codebook = (
        pd.read_excel("codebook.xlsx", index_col=5)
        .sort_index()
        .replace(
            {
                "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
                "HIV/AIDS": "HIV-AIDS",
                "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
                "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
                "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
                "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
            }
        )
    )
    dfworld = feather.read_feather("world_incidence.feather")
    agestd = dfworld.rename(
        index={
            "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
            "HIV/AIDS": "HIV-AIDS",
            "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
            "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
            "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
            "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
        }
    )

    for level in codebook["Cause Name"].values:
        try:

            prev = agestd.swaplevel().loc[level]

            try:
                x = df.columns
                y = prev.values

                regression = scipy.stats.linregress(x, y)

                R2 = R2.append(pd.DataFrame([regression[2] ** 2], index=[level]))
                slope = slope.append(pd.DataFrame([regression[0]], index=[level]))
            except ValueError:
                ignored.append(level)
        except KeyError:
            ignored.append(level)
    dict_name = dict(zip(list(codebook["Cause Name"].values), list(codebook.index.values)))
    result = pd.concat([R2[0], slope[0]], keys=["R2", "slope"], axis=1).T
    result.loc["Code"] = result.rename(columns=dict_name).columns.values
    result.columns.name = "Disease"
    result = result.T.reset_index().set_index(["Code", "Disease"])

    result[result["R2"] > 0.2][result[result["R2"] > 0.2]["slope"] > 0]


########### v2 ################


def data_GBD():
    df = pd.DataFrame()
    for j in range(1, 14, 1):
        zf = zipfile.ZipFile("GBD data 2/IHME-GBD_2019_DATA-939f6de0/IHME-GBD_2019_DATA-939f6de0-" + str(j) + ".zip")
        df = df.append(
            pd.read_csv(
                zf.open("IHME-GBD_2019_DATA-939f6de0-" + str(j) + ".csv"),
                index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
        )
    feather.write_feather(df, "GBD data 2/GBD data.feather")
    df = (
        df["val"]
        .unstack(level="year")
        .xs("Age-standardized", level="age_name")
        .xs("Both", level="sex_name")
        .groupby(level=["measure_name", "location_name", "cause_name"])
        .sum()
    )
    feather.write_feather(df.loc["Incidence"], "GBD data 2/GBD data incidence.feather")
    feather.write_feather(df.loc["Prevalence"], "GBD data 2/GBD data prevalence.feather")
    feather.write_feather(df.loc["YLDs (Years Lived with Disability)"], "GBD data 2/GBD data YLDs.feather")
    feather.write_feather(df.loc["YLLs (Years of Life Lost)"], "GBD data 2/GBD data YLLs.feather")

    (pd.read_excel("codebook.xlsx", index_col=1).sort_index()).loc[
        feather.read_feather("GBD data 2/GBD data.feather")
        .reset_index()
        .set_index("cause_name")
        .groupby(level="cause_name")
        .sum()
        .index
    ].reset_index().set_index("Cause Outline").sort_values(by="Cause Outline").to_excel("codebook_short.xlsx")


def meta_analysis():
    codebook = (
        (pd.read_excel("codebook_short.xlsx", index_col=1).sort_index())
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
                    feather.read_feather("GBD data 2/GBD data " + ind + ".feather").swaplevel().loc[level].stack()
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

                    regression = scipy.stats.linregress(x, y)

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
            "data_for_ma_" + ind + ".xlsx", merge_cells=False, sheet_name="all_years"
        )

        result_all[ind] = result.stack(dropna=False)
    result_all.to_excel("result_all.xlsx")


def meta_analysis_all_years():

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
        (pd.read_excel("codebook_short.xlsx", index_col=1).sort_index())
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
                        feather.read_feather("GBD data 2/GBD data " + ind + ".feather").swaplevel().loc[level][year]
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

                        regression = scipy.stats.linregress(x, y)
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
            with pd.ExcelWriter("data_for_ma_" + ind + ".xlsx", mode="a") as writer:
                df.drop(df[df.isna().any(axis=1)].index).to_excel(writer, merge_cells=False, sheet_name=str(year))


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

    ####


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


codebook = pd.read_excel("codebook_short.xlsx", index_col=1)
#

# .replace(
#     {
#         "HIV/AIDS and sexually transmitted infections": "HIV-AIDS and sexually transmitted infections",
#         "HIV/AIDS": "HIV-AIDS",
#         "HIV/AIDS - Drug-susceptible Tuberculosis": "HIV-AIDS - Drug-susceptible Tuberculosis",
#         "HIV/AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance": "HIV-AIDS - Multidrug-resistant Tuberculosis without extensive drug resistance",
#         "HIV/AIDS - Extensively drug-resistant Tuberculosis": "HIV-AIDS - Extensively drug-resistant Tuberculosis",
#         "HIV/AIDS resulting in other diseases": "HIV-AIDS resulting in other diseases",
#     }
# )


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
    df = pd.read_excel("data_for_ma_" + indicator + ".xlsx", sheet_name="all_years", index_col=[0, 1, 7]).rename(
        index=dictmod
    )
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
    dfneg[["slope", "R2", "pvalue"]].to_excel("results_neg.xlsx")
    dfpos = (
        dfpos.reset_index()
        .set_index(["mod", "slope"])
        .sort_index(ascending=False)
        .reset_index()
        .set_index(["mod", "Code", "Disease"])
    )
    dfpos["R2"] = dfpos["rvalue"] ** 2
    dfpos[["slope", "R2", "pvalue"]].to_excel("results_pos.xlsx")
    df["R2"] = df["rvalue"] ** 2
    df[["slope", "R2", "pvalue"]].to_excel("results_all.xlsx")


df2 = df[df["pvalue"] < 0.05]
# df2[df2['slope']>0]
len(df2[df2["rvalue"] ** 2 > 0.3])
len(df2[df2["rvalue"] ** 2 > 0.5]) - len(df2[df2["rvalue"] ** 2 > 0.4])


def fitR2():
    fig, axes = plt.subplots(7, 5, figsize=(7.5, 8.75 * 7 / 6))
    k = 0
    s = 1
    matplotlib.rc("xtick", labelsize=6)
    matplotlib.rc("ytick", labelsize=6)
    ind = "Incidence"
    indicator = feather.read_feather("GBD data 2/GBD data " + ind + ".feather").swaplevel()
    index = df2[(df2["rvalue"] ** 2 > 0.1) & (df2["rvalue"] ** 2 < 0.15)].unstack().unstack(level=0).index
    for level in index:
        ax = axes[k // 5, k % 5]
        ax.set_xlim([-1, 8])

        prev = indicator.loc[level].stack()

        x0 = np.log(energy.stack(dropna=False))
        y0 = pd.DataFrame(np.log(prev), index=energy.stack().index)[0]
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

        regression = scipy.stats.linregress(x, y)

        ax.plot(x_lin, x_lin * regression[0] + regression[1], color="black")
        if len(level) > 20:
            ax.set_title(level[:20] + "[...]", fontsize=8)
        else:
            ax.set_title(level, fontsize=8)

        # ax.annotate("R2 =" + str(round(regression[2] ** 2, 2)), (0, y.quantile(0.95)), size=8)

        k += 1
    plt.tight_layout()


def to_xlsx():
    df = (
        pd.read_excel("result_all.xlsx", index_col=[0, 1, 2])
        .unstack()
        .stack(level=0)
        .drop(["intercept", "mod", "stderr"], axis=1)
    )
    df["R2"] = df["rvalue"] ** 2
    df.drop(["rvalue"], axis=1).unstack().swaplevel(axis=1).sort_index(axis=1).drop("prevalence", axis=1).to_excel(
        "all_results_17-01-2023.xlsx"
    )

    df = pd.read_excel("result_all.xlsx", index_col=[0, 1, 2]).unstack().stack(level=0).drop(["mod", "stderr"], axis=1)
    df["R2"] = df["rvalue"] ** 2
    df2 = df.xs("incidence", level=2)
    df2 = df2[df2["pvalue"] < 0.05]
    df3 = df2[df2["rvalue"] ** 2 > 0.2]
    df3 = df3.sort_index().drop(["D", "E", "F"])
    dfpos = df3[df3["slope"] > 0]
    dfneg = df3[df3["slope"] < 0]
    dfpos.to_excel("results_pos_date.xlsx")
    dfneg.to_excel("results_neg_date.xlsx")


def fig_final():
    fig, axes = plt.subplots(7, 5, figsize=(7.5, 10.208))
    k = 0
    s = 1
    dfpos = pd.read_excel("results_pos_17-01-2023.xlsx", index_col=[1])
    dfneg = pd.read_excel("results_neg_17-01-2023.xlsx", index_col=[1])
    data = feather.read_feather("GBD data 2/GBD data incidence.feather").swaplevel()

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
    plt.savefig("Positive1_17-01-2023.pdf")
    plt.savefig("Positive1_17-01-2023.png", dpi=200)
    plt.savefig("Positive1_17-01-2023.svg")

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
    plt.savefig("Negative1_17-01-2023.pdf")
    plt.savefig("Negative1_17-01-2023.png", dpi=200)
    plt.savefig("Negative1_17-01-2023.svg")

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
    plt.savefig("Negative2_17-01-2023.pdf")
    plt.savefig("Negative2_17-01-2023.png", dpi=200)
    plt.savefig("Negative2_17-01-2023.svg")

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
    plt.savefig("Positive2_17-01-2023.pdf")
    plt.savefig("Positive2_17-01-2023.png", dpi=200)
    plt.savefig("Positive2_17-01-2023.svg")


def fig_SI():
    fig, axes = plt.subplots(7, 5, figsize=(7.5, 10.208))
    k = 0
    s = 1
    dfpos = pd.read_excel("results_pos_17-01-2023.xlsx", index_col=[1])
    dfneg = pd.read_excel("results_neg_17-01-2023.xlsx", index_col=[1])
    data = feather.read_feather("GBD data 2/GBD data incidence.feather").swaplevel()

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
    plt.savefig("Positive1_17-01-2023.pdf")
    plt.savefig("Positive1_17-01-2023.png", dpi=200)
    plt.savefig("Positive1_17-01-2023.svg")

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
    plt.savefig("Negative1_17-01-2023.pdf")
    plt.savefig("Negative1_17-01-2023.png", dpi=200)
    plt.savefig("Negative1_17-01-2023.svg")

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
    plt.savefig("Negative2_17-01-2023.pdf")
    plt.savefig("Negative2_17-01-2023.png", dpi=200)
    plt.savefig("Negative2_17-01-2023.svg")

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
    plt.savefig("Positive2_17-01-2023.pdf")
    plt.savefig("Positive2_17-01-2023.png", dpi=200)
    plt.savefig("Positive2_17-01-2023.svg")


### fig SI
def fig_appendix_inc():
    for indicator in ["incidence", "YLDs", "YLLs"]:
        data = feather.read_feather("GBD data 2/GBD data " + indicator + ".feather").swaplevel()
        matplotlib.rc("xtick", labelsize=6)
        matplotlib.rc("ytick", labelsize=6)

        df_inc = (
            pd.read_excel("data_for_ma_" + indicator + ".xlsx", sheet_name="all_years", index_col=[0, 1, 7])
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
