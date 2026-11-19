"""Reproduce the primary estimates in Are Official Forecasts Pricing In AI?
Download primary_panel.csv beside this script. Run with Python 3 and
numpy, scipy and pandas installed: python reproduce.py

Each row is a public occupational category. The five growth columns contain
annual log percentage points from the BLS wage-and-salary matrix. Mention
columns flag an explicit occurrence of AI or artificial intelligence in
BLS Table 1.12. They measure literal wording. Source workbooks are available
from https://www.bls.gov/emp/data/projections-archive.htm and
https://www.bls.gov/emp/tables.htm. All ranks and controls in this panel were
fixed for the primary model at the protocol commit 1412873. The scaled
observed and theoretical ranks are in units of ten index points.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

panel = pd.read_csv(Path(__file__).with_name("primary_panel.csv"))
vintages = ["2021-31", "2022-32", "2023-33", "2024-34", "2025-35"]
continuous = ["observed_per10", "theory_per10", "log_employment_2021",
              "log_wage_uncensored_design_term", "wage_censored_indicator",
              "wage_missing_indicator"]
major = pd.get_dummies(panel.major_group_code, drop_first=True, dtype=float)
education = pd.get_dummies(panel.typical_education_needed_for_entry, dtype=float)
education = education.drop(columns=["High school diploma or equivalent"])
controls = pd.concat([panel[continuous], major, education], axis=1)
design = np.ones((len(panel), 1))
for column in controls:
    candidate = np.column_stack([design, controls[column]])
    if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
        design = candidate
groups = panel.minor_group_code.to_numpy()
clusters = np.unique(groups)
n, k = design.shape
assert (n, k, len(clusters)) == (641, 35, 91)
weight = panel.analytic_weight_2021.to_numpy()
weighted_design = np.sqrt(weight)[:, None] * design
bread = np.linalg.inv(weighted_design.T @ weighted_design)
rank = panel.observed_per10.to_numpy()
order = np.argsort(rank)
cumulative = np.cumsum(weight[order]) / sum(weight)
q25, q75 = rank[order][np.searchsorted(cumulative, [.25, .75])]
print("New release | observed slope | 95% CR1 CI | revision SD | IQR contrast")
for old, new in zip(vintages, vintages[1:]):
    revision = (panel["growth_" + new] - panel["growth_" + old]).to_numpy()
    beta = np.linalg.lstsq(weighted_design, np.sqrt(weight) * revision, rcond=None)[0]
    residual = revision - design @ beta
    scores = np.array([design[groups == group].T @ (weight * residual)[groups == group]
                       for group in clusters])
    correction = len(clusters) / (len(clusters) - 1) * (n - 1) / (n - k)
    covariance = correction * bread @ scores.T @ scores @ bread
    half_width = t.ppf(.975, len(clusters) - 1) * np.sqrt(covariance[1, 1])
    sd = np.sqrt(np.average((revision - np.average(revision, weights=weight)) ** 2, weights=weight))
    print(new, *(f"{value:.6f}" for value in (beta[1], beta[1] - half_width,
          beta[1] + half_width, sd, beta[1] * (q75 - q25))))
print("Literal mentions across releases:", [int(panel["mention_" + v].sum()) for v in vintages])
risk = panel[panel["mention_2024-34"] == 0]
entry = risk["mention_2025-35"]
entry_design = np.column_stack([np.ones(len(risk)), entry, risk.theory_per10])
gap = np.linalg.lstsq(entry_design, 10 * risk.observed_per10, rcond=None)[0][1]
print(f"New mentions: {entry.sum()} of {len(risk)}; theory-adjusted usage-rank gap: {gap:.2f}")
