"""Reproduce the exploratory platform-rank tie check in Appendix B."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exploratory import design, fit, intervals, permutation


BASE = Path(__file__).resolve().parent
SOURCES = ['anthropic_observed', 'microsoft_applicability']


def platform_ranks(source_scores, use_midranks):
    """Rank among each platform's mapped occupations, including those outside the panel."""
    counts = source_scores.count()
    if use_midranks:
        positions = source_scores.rank(method='average')
    else:
        positions = source_scores.rank(method='min') - 1
    return 100 * positions / counts


def analyze(panel, vintages):
    x = design(panel)
    weights = panel.analytic_weight_2021.to_numpy()
    groups = panel.minor_group_code.to_numpy()
    levels = panel[['growth_' + vintage for vintage in vintages]].to_numpy()
    level_beta, level_scores = fit(x, levels, weights, groups)
    revision_beta, revision_scores = fit(x, np.diff(levels, axis=1), weights, groups)
    np.testing.assert_allclose(np.diff(level_beta), revision_beta, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.diff(level_scores, axis=1), revision_scores, atol=1e-12, rtol=0)

    risk = panel.loc[panel['mention_2024-34'].eq(0)]
    entry = risk['mention_2025-35'].eq(1)
    gap = permutation(panel)
    gap.update(n_new=int(entry.sum()), n_other=int((~entry).sum()),
               mean_new=float(risk.loc[entry, 'observed_per10'].mean() * 10),
               mean_other=float(risk.loc[~entry, 'observed_per10'].mean() * 10))
    return {'levels': dict(zip(vintages, intervals(level_beta, level_scores, *x.shape))),
            'revisions': dict(zip(vintages[1:], intervals(revision_beta, revision_scores, *x.shape))),
            'new_mentions': gap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write a copy of the JSON results here.')
    args = parser.parse_args()
    panel = pd.read_csv(BASE / 'primary_panel.csv').set_index('occupation_code', verify_integrity=True)
    comparison = pd.read_csv(BASE / 'comparison_inputs.csv').set_index('occupation_code', verify_integrity=True)
    scores = pd.read_csv(BASE / 'usage_reference.csv', float_precision='round_trip').set_index('occupation_code', verify_integrity=True)[SOURCES]
    assert len(scores) == 831 and len(panel) == 641 and panel.minor_group_code.nunique() == 91
    originals = platform_ranks(scores, use_midranks=False)
    alternate = platform_ranks(scores, use_midranks=True)
    # The input retains unscored occupations as missing. A published zero is a score.
    primary_scores = scores.loc[panel.index]
    assert primary_scores.notna().all().all()
    np.testing.assert_allclose(primary_scores.anthropic_observed, comparison.loc[panel.index, 'anthropic_observed'], atol=1e-14, rtol=0)
    np.testing.assert_allclose(originals.loc[panel.index, 'anthropic_observed'] / 10, comparison.loc[panel.index, 'claude_rank_per10'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(originals.loc[panel.index].mean(axis=1) / 10, panel.observed_per10, atol=1e-12, rtol=0)
    report = {'sample': {'n': len(panel), 'clusters': int(panel.minor_group_code.nunique()),
                         'reference_nonmissing': scores.count().to_dict(),
                         'reference_zeros': scores.eq(0).sum().to_dict(),
                         'sample_zeros': primary_scores.eq(0).sum().to_dict()}}
    vintages = [name.removeprefix('growth_') for name in panel if name.startswith('growth_')]
    for label, ranks in [('strict_below', originals), ('midrank', alternate)]:
        usage = ranks.loc[panel.index].mean(axis=1, skipna=False) / 10
        report[label] = analyze(panel.assign(observed_per10=usage), vintages)
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
