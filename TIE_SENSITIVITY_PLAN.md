# Sensitivity to tied platform-exposure scores

This exploratory check follows the Appendix C rank diagnostic in manuscript commit `9f60baab2aecb89d785e4dfe86b935bd9841a04b`. The original results and the Appendix C comparison were already known. The 15 original revision specifications keep their original definitions.

## One change in the comparison

Rank each observed platform score among all nonmissing occupations in its original, directly mapped BLS reference population, before selecting the common 641 occupations. If L observations score lower, t have exactly the same score and N are observed, the primary rank is 100 L/N. The alternative is the average one-based rank, 100 [L + (t + 1)/2]/N. Apply this rule to Claude and Copilot separately and average the two ranks. Keep theoretical ranks, sample, outcomes, controls, weights, clusters, and mention labels fixed. Report the source denominators and the Claude zero count in the 641 occupations. Validate the reconstructed original ranks against the replication panel before fitting. The 584-occupation Appendix C ranking population remains separate.

## Outcomes and inference

1. Fit the four consecutive revision regressions and the five Panel B forecast-level regressions on the original 641-occupation panel, with the same baseline employment weights, full controls, and minor-group CR1 inference (95% intervals, t with 90 df). Report all estimates and intervals. Verify that adjacent level-slope differences reproduce the revision slopes. Panel A uses raw coverage and does not require reranking.
2. Among the 612 previously unnamed occupations, regress the new usage rank on the indicator for gaining an AI mention and the original theoretical rank, with equal occupation weights. Report the entrant count, both mean ranks, adjusted gap, HC3 standard error and 95% t interval. Repeat both existing two-sided 99,999-draw permutation tests with seed 20261119, unrestricted and within the original theoretical-rank deciles. Retain the plus-one p-value correction and report extreme counts.
3. Run the original rank rule through the same code as a reference. Compare the signs and uncertainty of the four revisions, the timing of the point-estimate minimum, the concentration of new mentions, and the five level slopes. Report any change in interpretation. These results are exploratory and do not enter the family of original primary tests.
