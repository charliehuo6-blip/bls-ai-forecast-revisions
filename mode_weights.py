"""Reproduce Table C1. Add --check-ranks for the later Claude sample and rank comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from exploratory import design, fit, intervals

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check-ranks", action="store_true", help="Also report the post hoc comparison in Appendix C.")
args = parser.parse_args()
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

def fit_rank(sample, rank, control_columns):
    x = np.column_stack([np.ones(len(sample)), rank])
    for column in control_columns.T:
        candidate = np.column_stack([x, column])
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            x = candidate
    y = (sample["growth_2025-35"] - sample["growth_2024-34"]).to_numpy()[:, None]
    weights = sample.analytic_weight_2021.to_numpy()
    groups = sample.minor_group_code.to_numpy()
    beta, scores = fit(x, y, weights, groups)
    return beta, scores, x.shape


results, paired = {}, {}
matched_controls = controls[keep, :]
for name in ["auto", "aug", "half", "equal", "anthropic_observed", "microsoft_applicability"]:
    source = modes[name].round(12)
    rank = source.rank(method="average").to_numpy()*10/len(source)
    beta, scores, dims = fit_rank(frame, rank, matched_controls)
    results[name] = intervals(beta, scores, *dims)[0]
    if name in ["equal", "half"]:
        paired[name] = beta, scores, dims

b1, s1, dims = paired["equal"]
b0, s0, dims0 = paired["half"]
assert dims == dims0
results["equal_minus_half"] = intervals(b1-b0, s1-s0, *dims)[0]
report = dict(n=len(frame), clusters=frame.minor_group_code.nunique(), regressors=dims[1],
    spearman_half_published=spearmanr(modes.half.round(12), modes.anthropic_observed).statistic,
    matched_pairs=int(modes.tasks_matched.sum()), listed_pairs_in_retained=int(modes.tasks_listed.sum()),
    slopes=results)

if args.check_ranks:
    comparison = pd.read_csv(base / "comparison_inputs.csv").set_index("occupation_code", verify_integrity=True)
    original = comparison.loc[panel.occupation_code, "claude_rank_per10"].to_numpy()
    claude = modes.anthropic_observed.round(12)
    # The original scale counts scores strictly below each score. Holding the
    # formula fixed isolates the change in reference population from tie handling.
    strict_rank = (claude.rank(method="min").to_numpy() - 1) * 10 / len(claude)
    rank_check = {}
    for label, sample, ranks, columns in [
        ("full_original_ranks", panel, original, controls),
        ("matched_original_ranks", frame, original[keep], matched_controls),
        ("matched_strict_below", frame, strict_rank, matched_controls),
    ]:
        beta, scores, dims = fit_rank(sample, ranks, columns)
        rank_check[label] = intervals(beta, scores, *dims)[0]
    rank_check["matched_average_ranks"] = results["anthropic_observed"]
    rank_check["matched_zero_scores"] = int(claude.eq(0).sum())
    report["post_hoc_claude_rank_check"] = rank_check

print(json.dumps(report, indent=2))
