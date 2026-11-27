# Sensitivity to tied platform-exposure scores

This exploratory check follows the Appendix C rank diagnostic. The original results and the Appendix C comparison were already known. The 15 original revision specifications keep their original definitions.

## One change in the comparison

Each observed platform score is ranked among all nonmissing occupations in its original, directly mapped BLS reference population, before selection of the common 641 occupations. If L observations score lower, t have exactly the same score and N are observed, the primary rank is 100 L/N. The alternative is the average one-based rank, 100 [L + (t + 1)/2]/N. Each rule applies to Claude and Copilot separately, followed by averaging the two ranks. Theoretical ranks, sample, outcomes, controls, weights, clusters, and mention labels remain fixed. The report includes the source denominators and the Claude zero count in the 641 occupations. The reconstructed original ranks must agree with the replication panel before estimation. The 584-occupation Appendix C ranking population remains separate.

## Outcomes and inference

1. The four consecutive revision regressions and the five Panel B forecast-level regressions use the original 641-occupation panel, with the same baseline employment weights, full controls, and minor-group CR1 inference (95% intervals, t with 90 df). All estimates and intervals are reported. Adjacent level-slope differences must reproduce the revision slopes. Panel A uses raw coverage and does not require reranking.
2. Among the 612 previously unnamed occupations, the new usage rank is regressed on the indicator for gaining an AI mention and the original theoretical rank, with equal occupation weights. The report includes the entrant count, both mean ranks, adjusted gap, HC3 standard error and 95% t interval. Both existing two-sided 99,999-draw permutation tests use seed 20261119, unrestricted and within the original theoretical-rank deciles, with the plus-one p-value correction and extreme counts reported.
3. The original rank rule runs through the same code as a reference. The comparison covers the signs and uncertainty of the four revisions, the timing of the point-estimate minimum, the concentration of new mentions, the five level slopes, and any change in interpretation. These results are exploratory and do not enter the family of original primary tests.
