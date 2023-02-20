# Energy footprint and global burden of diseases

The goal of this project is to evaluate the relationship between energy footprint per capita (EFC) and the incidence, Years Lived with Disability (YLDs) and Years of Life Lost (YLLs) of all the diseases from the Global Burden of Diseases (GBD).

Energy data comes from the eora global Multi regional input-output databse: https://worldmrio.com/
Diseases data come from the GBD: https://vizhub.healthdata.org/gbd-results/

## Requirements

```bash
$ pip install -r requirements.txt
```

## Usage

Follow the steps in the runme.py file.

The steps are:

* process the data so that Eora and GBD data are in the same format
* calculate regression coefficients
* run functions from exiobase_functions.py to calculate the footprints
* run functions from sankey_function.py to generate files that will be used as inputs for building the sankey diagrams
* run save_sankey() from fig_sankey.py to generation sankey figures and save them in Results/Sankey_figs


## Citation

Andrieu, B., Le Boulzec, H., Delannoy, L., Verzier, F., Winter, G., Vidal, O., Mapping global greenhouse gases emissions: an interactive, open-access, web application. Available at: