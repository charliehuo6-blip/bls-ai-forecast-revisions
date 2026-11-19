"""Exploratory analyses for the BLS projections paper. Run beside primary_panel.csv and comparison_inputs.csv.
Usage: python exploratory.py [--output results.json]
All four vintage updates share the same occupational sample and controls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def design(frame):
    controls = frame[["observed_per10", "theory_per10", "log_employment_2021",
                      "log_wage_uncensored_design_term", "wage_censored_indicator",
                      "wage_missing_indicator"]]
    major = pd.get_dummies(frame.major_group_code, drop_first=True, dtype=float)
    educ = pd.get_dummies(frame.typical_education_needed_for_entry, dtype=float)
    controls = pd.concat([controls, major, educ.drop(columns="High school diploma or equivalent")], axis=1)
    x = np.ones((len(frame), 1))
    for col in controls:
        candidate = np.column_stack([x, controls[col]])
        if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
            x = candidate
    assert x.shape == (641, 35)
    return x


def fit(x, y, weight, groups):
    # All outcomes in a column share an identical regression design.
    b = np.linalg.lstsq(np.sqrt(weight)[:, None]*x, np.sqrt(weight)[:, None]*y, rcond=None)[0]
    residual = y - x @ b
    group_ids = np.unique(groups)
    effect_weight = (np.linalg.solve(x.T @ (weight[:, None]*x), x.T)[1] * weight)
    scores = np.array([(effect_weight[groups == g, None]*residual[groups == g]).sum(axis=0) for g in group_ids])
    return b[1], scores


def intervals(beta, scores, n, k):
    m = scores.shape[0]
    cov = scores.T @ scores * (m / (m-1)) * ((n-1)/(n-k))
    se = np.sqrt(np.diag(cov))
    half = t.ppf(.975, m-1)*se
    return [dict(estimate=b, se=s, lower=l, upper=u, p=2*t.sf(abs(b/s),m-1))
            for b, s, l, u in zip(beta, se, beta-half, beta+half)]


def permutation(frame, draws=99999, seed=20261119):
    # FWL slope for a candidate set. Partial out intercept and theoretical rank.
    risk = frame.loc[frame["mention_2024-34"].eq(0)].copy()
    entry = risk["mention_2025-35"].to_numpy().astype(bool)
    y = risk.observed_per10.to_numpy()*10
    theory = risk.theory_per10.to_numpy()
    z = theory - theory.mean()
    zz = z @ z
    resid = y - y.mean() - z * (z @ y)/zz
    n, k = len(y), int(entry.sum())
    assert (n,k) == (612,14)
    def slope(idx):
        return resid[idx].sum(axis=-1)/(k - k*k/n - z[idx].sum(axis=-1)**2/zz)
    observed = slope(np.flatnonzero(entry))
    raw = np.linalg.lstsq(np.column_stack([np.ones(n),entry,theory]),y,rcond=None)[0][1]
    np.testing.assert_allclose(raw,observed)
    rng = np.random.default_rng(seed)
    bins = pd.qcut(theory,10,labels=False,duplicates="drop")
    strata = [(ids,int(entry[ids].sum())) for b in np.unique(bins) if (ids:=np.flatnonzero(bins==b)).size]
    def simulate(strata):
        count = 0
        for start in range(0,draws,500):
            size = min(500,draws-start)
            idx=[]
            for ids, count_in_stratum in strata:
                if count_in_stratum:
                    keys = rng.random((size,len(ids)))
                    idx.append(ids[np.argpartition(keys,count_in_stratum-1,axis=1)[:,:count_in_stratum]])
            b = slope(np.concatenate(idx,axis=1))
            count += int((abs(b)>=abs(observed)-1e-11).sum())
        return dict(draws=draws,as_or_more_extreme=count,two_sided_p=(count+1)/(draws+1))
    return dict(adjusted_gap=observed,seed=seed,unrestricted=simulate([(np.arange(n),k)]),
                within_theory_deciles=simulate(strata))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--output",type=Path)
    output = args.parse_args().output
    f = pd.read_csv(Path(__file__).with_name("primary_panel.csv"))
    v = [col.split("_",1)[1] for col in f if col.startswith("growth_")]
    w = f.analytic_weight_2021.to_numpy()
    g = f.minor_group_code.to_numpy()
    x = design(f)
    levels = f[["growth_"+j for j in v]].to_numpy()
    n,k = x.shape
    bl, sl = fit(x,levels,w,g)
    dr = np.diff(levels,axis=1)
    br, sr = fit(x,dr,w,g)
    np.testing.assert_allclose(np.diff(bl),br,atol=1e-12)
    # The stacked design is block diagonal with all controls interacted by update.
    # Scores align on the same minor groups across updates. CR1 uses stacked N and K.
    contrast = np.array([-1,.5,.5,0])
    contrast_ci = intervals(np.array([br@contrast]),(sr@contrast)[:,None],4*n,4*k)[0]
    reduced_x = np.delete(x,2,axis=1)
    bu,su = fit(reduced_x,dr,w,g)
    unconditional_levels,uls=fit(reduced_x,levels,w,g)
    us=f.observed_per10; theory=f.theory_per10
    cov=np.cov(us,theory,aweights=w,ddof=0)
    comparison = pd.read_csv(Path(__file__).with_name("comparison_inputs.csv"))
    assert comparison.occupation_code.equals(f.occupation_code)
    raw_x = np.column_stack([np.ones(n),10*comparison.anthropic_observed])
    benchmark = []
    for j, vintage in enumerate(v):
        # Convert annual log percentage points back to a ten-year simple percentage.
        y = 100*np.expm1(levels[:,j:j+1]/10)
        base = comparison["base_"+vintage].to_numpy()
        beta, scores = fit(raw_x,y,base,g)
        benchmark.extend(intervals(beta,scores,n,2))
    report=dict(vintages=v,raw_coverage_benchmark=benchmark,level_slopes=intervals(bl,sl,n,k),
        timing_contrast=contrast_ci,no_theory_revision=intervals(bu,su,n,k-1),
        no_theory_levels=intervals(unconditional_levels,uls,n,k-1),
        weighted_corr=cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]),unweighted_corr=us.corr(theory),
        entry_permutation=permutation(f))
    result=json.dumps(report,indent=2)+'\n'
    if output:output.write_text(result)
    print(result)


if __name__ == "__main__":
    main()
