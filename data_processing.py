import pandas as pd
import pyarrow.feather as feather
import country_converter as coco
import zipfile


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
    feather.write_feather(df, "raw_data/tradereport.feather")

    feather.write_feather(
        df.xs("Energy Usage (TJ)", level="Indicator Description")["Value"]
        .unstack(level="Record")["Footprint"]
        .unstack(level="Year")
        .groupby(level="Country Name")
        .sum(),
        "processed_data/energy footprint.feather",
    )


def data_GBD():
    df = pd.DataFrame()
    for j in range(1, 14, 1):
        zf = zipfile.ZipFile("raw_data/GBD/IHME-GBD_2019_DATA-939f6de0/IHME-GBD_2019_DATA-939f6de0-" + str(j) + ".zip")
        df = df.append(
            pd.read_csv(
                zf.open("IHME-GBD_2019_DATA-939f6de0-" + str(j) + ".csv"),
                index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
        )
    feather.write_feather(df, "processed_data/GBD data.feather")
    df = (
        df["val"]
        .unstack(level="year")
        .xs("Age-standardized", level="age_name")
        .xs("Both", level="sex_name")
        .groupby(level=["measure_name", "location_name", "cause_name"])
        .sum()
    )
    feather.write_feather(df.loc["Incidence"], "processed_data/GBD data incidence.feather")
    feather.write_feather(df.loc["Prevalence"], "processed_data/GBD data prevalence.feather")
    feather.write_feather(df.loc["YLDs (Years Lived with Disability)"], "processed_data/GBD data YLDs.feather")
    feather.write_feather(df.loc["YLLs (Years of Life Lost)"], "processed_data/GBD data YLLs.feather")

    (pd.read_excel("raw_data/GBD/codebook.xlsx", index_col=1).sort_index()).loc[
        feather.read_feather("processed_data/GBD data.feather")
        .reset_index()
        .set_index("cause_name")
        .groupby(level="cause_name")
        .sum()
        .index
    ].reset_index().set_index("Cause Outline").sort_values(by="Cause Outline").to_excel(
        "processed_data/codebook_short.xlsx"
    )
