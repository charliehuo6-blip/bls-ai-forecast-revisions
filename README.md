# Are Official Forecasts Pricing In AI?

Charlie Huo · Swarthmore College · November 2026

This repository accompanies *Are Official Forecasts Pricing In AI? Evidence from BLS Occupational Projections, 2021–2035*. It compares four consecutive revisions to ten-year projected employment growth across 641 occupations and counts explicit mentions of AI in BLS Table 1.12.

## Reproduce the primary results

Download or clone this repository. From its root, run:

```sh
python -m pip install numpy pandas scipy
python reproduce.py
```

The script prints the four primary usage coefficients and clustered 95% confidence intervals, the employment-weighted standard deviations of revisions, fitted interquartile contrasts, literal mention counts and the exploratory comparison of newly named occupations. The numerical inputs contain occupation-level public statistics. No workers or platform users are identified.

## Data and timing

- `primary_panel.csv` contains one row per occupation. `growth_*` is annual growth in wage-and-salary employment in log percentage points. `mention_*` indicates an exact word match in any staffing explanation for that occupation and release. Occupations missing from the table receive zero. The script documents the exposure scales and uses 2021 employment weights.
- The [analysis plan and configuration at the pre-estimation commit](https://github.com/charliehuo6-blip/bls-ai-forecast-revisions/tree/14128730fda96863e2bc70e16e61f2699ee7ba67) identify the prespecified revision models. The literal-word and scale comparisons were added after estimation.
- The agency provides the [projection archive](https://www.bls.gov/emp/data/projections-archive.htm), [current tables](https://www.bls.gov/emp/tables.htm) and the [2023–33 occupational case studies](https://doi.org/10.21916/mlr.2025.1).
