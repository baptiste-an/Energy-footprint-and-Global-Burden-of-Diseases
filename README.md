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
* calculate regression coefficients and export them to excel format
* plot all regression graphs
* run the r code for the meta analyses
* plot the meta analysis forest plots


## Citation

Andrieu, B., Marrauld, L., Chevance, G., Egnell, M., Vidal, O., Boyer, L., Fond, G., Energy footprint and Global Burden of Disease: an analysis of 176 countries over the period 1990-2019. Available at: