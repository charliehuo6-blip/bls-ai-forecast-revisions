"""Reproduce the six final-update slopes and the paired mode-weight contrast."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from exploratory import design, fit, intervals

base = Path(__file__).resolve().parent
panel = pd.read_csv(base / "primary_panel.csv")
controls = design(panel)[:, 2:]
modes = pd.read_csv(base / "mode_inputs.csv").set_index("occupation_code")
keep = panel.occupation_code.isin(modes.index)
frame = panel.loc[keep]
modes = modes.loc[frame.occupation_code].copy()
modes["half"] = modes.auto + .5*modes.aug
modes["equal"] = modes.auto + modes.aug
# The equal-mode score is exactly the fraction covered. Integer rounding removes
# numerical noise before reranking rational coverage fractions with tied values.
modes["equal"] = (modes.equal*modes.tasks_matched).round()/modes.tasks_matched
weights = frame.analytic_weight_2021.to_numpy()
groups = frame.minor_group_code.to_numpy()
y = (frame["growth_2025-35"] - frame["growth_2024-34"]).to_numpy()[:, None]

results, paired = {}, {}
for name in ["auto", "aug", "half", "equal", "anthropic_observed", "microsoft_applicability"]:
    source = modes[name].round(12)
    rank = source.rank(method="average").to_numpy()*10/len(source)
    x = np.column_stack([np.ones(len(frame)), rank])
    for column in controls[keep, :].T:
        candidate = np.column_stack([x, column])
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            x = candidate
    beta, scores = fit(x, y, weights, groups)
    results[name] = intervals(beta, scores, *x.shape)[0]
    if name in ["equal", "half"]:
        paired[name] = beta, scores, x.shape

b1, s1, dims = paired["equal"]
b0, s0, dims0 = paired["half"]
assert dims == dims0
results["equal_minus_half"] = intervals(b1-b0, s1-s0, *dims)[0]
print(json.dumps(dict(n=len(frame), clusters=len(np.unique(groups)), regressors=dims[1],
   spearman_half_published=spearmanr(modes.half.round(12), modes.anthropic_observed).statistic,
   matched_pairs=int(modes.tasks_matched.sum()), listed_pairs_in_retained=int(modes.tasks_listed.sum()),
   slopes=results), indent=2))
