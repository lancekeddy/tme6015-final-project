# Licence notes

## Code

The code in `src/` and `notebooks/` is under the MIT licence; see [LICENSE](LICENSE). The MIT licence does not apply to the data in `data/`.

## Data

The files in `data/processed_v2/` are derived from two third-party datasets. The derived dataset combines both and is shared for non-commercial academic use with attribution. The `source` column marks the origin of every row (`enernoc` or `bdg2`).

### EnerNOC open dataset

- Anonymized 5-minute energy data for 100 commercial and industrial sites, 2012.
- Terms as stated in the licence file supplied with the data: "All data is Copyright EnerNOC and is being made available for non-commercial and research purposes only under the Creative Commons Attribution-NonCommercial 3.0 license." Full terms: https://creativecommons.org/licenses/by-nc/3.0/
- Source: https://open-enernoc-data.s3.amazonaws.com/anon/README
- Attribution: credit EnerNOC for rows with `source = enernoc`.

### Building Data Genome Project 2 (BDG2)

- Hourly meter data for 1,636 non-residential buildings, 2016 and 2017.
- Terms: see the source repository, https://github.com/buds-lab/building-data-genome-project-2
- Citation: Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data* 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x
- Attribution: cite the paper above for rows with `source = bdg2`.
