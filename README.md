# Are Official Forecasts Pricing In AI?

Replication materials for *Are Official Forecasts Pricing In AI? Evidence from BLS Occupational Projections, 2021–2035* by Charlie Huo, Swarthmore College.

**[Read the paper (PDF)](paper.pdf)**

The paper follows the same 641 occupations through five BLS releases. It compares growth projections and their revisions with occupation-level measures of AI use, and tracks when BLS explicitly names AI in its staffing explanations.

## Reproduce the results

From a local copy of this repository, run the following with Python 3.10 or later.

```sh
python -m pip install numpy pandas scipy
python reproduce.py
python exploratory.py --output exploratory_results.json
python mode_weights.py --check-ranks
python tie_sensitivity.py --output tie_results.json
```

- `reproduce.py` prints the four primary revision slopes with clustered confidence intervals, the spread of revisions, interquartile contrasts and release-level literal mention counts.
- `exploratory.py` produces the estimates and intervals plotted in **both panels of Figure 1**. It also computes the 2024–34 bridge between them (Table A1), slopes omitting theoretical exposure, and the single contrast between the two middle updates and the first. That contrast clusters across all four updates by minor occupational group.
- For newly named occupations, the exploratory script gives the equal-weight, theory-adjusted rank difference and HC3 interval. It runs **99,999 draws for each of two permutation tests**, one unrestricted and one preserving entry counts within theoretical-rank deciles (seed 20261119). The whole script took about **3 seconds** in the test environment with single-threaded OpenBLAS. Runtime varies by machine. JSON contains estimates used for the figures, rather than rendered figures.

The [supplemental plan](https://github.com/charliehuo6-blip/bls-ai-forecast-revisions/commit/5583b0323f5dc1226c71d9561d46fd2ad71b3425) precedes estimation of the mode comparisons. `mode_weights.py` reproduces all six final-update slopes and the paired change in Table C1 from `mode_inputs.csv`. To regenerate the input from task-level data, run `python rebuild_modes.py`. It fetches the two pinned Economic Index files, verifies their SHA-256 checksums and applies the study crosswalk in `mode_mapping.csv`. Task statements originate in the U.S. Department of Labor [O*NET database](https://www.onetcenter.org/license_db.html). Task penetration comes from the [March 2026 study appendix](https://assets.claude.com/e5f77fc0e77c0185110b5e4b909602791ae76eae.pdf).

The optional `--check-ranks` output separates the change in the Claude-only sample from changes in the reference population and the treatment of ties (Appendix C). The [exploratory tie-sensitivity plan](TIE_SENSITIVITY_PLAN.md) was committed after that diagnostic and before the combined-platform analysis. `tie_sensitivity.py` reproduces the Appendix B revision row and Tables B1–B2. It rebuilds the original platform ranks as a check, reranks both observed measures on the same reference populations, and repeats all nine slopes and both 99,999-draw mention tests under each rule. The theoretical control and the permutation strata stay fixed.

## Inputs and timing

`primary_panel.csv` contains 641 occupational categories. Each `growth_*` column is annual log growth in wage-and-salary employment, in percentage points. Each `mention_*` column flags an explicit mention of “AI” or “artificial intelligence” in that occupation’s Table 1.12 entries, counting an occupation absent from the table as zero. `comparison_inputs.csv` gives the matched Claude exposure share, Claude rank in units of ten rank points, and base employment (thousands of jobs) for each vintage. The Claude rank comes from the frozen source distribution before restriction to these 641 occupations. `usage_reference.csv` contains the two directly mapped platform scores for all 831 tracked occupations before that restriction. Empty values mean no score was mapped (755 scored occupations for Claude and 784 for Copilot). A numeric zero is a published score. Source credits and reuse terms appear below.

The [pre-estimation commit](https://github.com/charliehuo6-blip/bls-ai-forecast-revisions/tree/14128730fda96863e2bc70e16e61f2699ee7ba67) contains the plan, configuration and frozen-file inventory. Figure 1, the specification bridge, timing contrast and textual comparisons are exploratory. The manuscript supplement includes the full 15-specification revision results and the staffing-bound checks.

## Data sources, attribution and licenses

| Source | Input and terms |
| --- | --- |
| U.S. Bureau of Labor Statistics | National Employment Matrix, occupational projections, Table 1.12, baseline controls and occupational crosswalks. BLS identifies its published statistics as public domain and asks users to credit the agency. [Projection archive](https://www.bls.gov/emp/data/projections-archive.htm) · [Current tables](https://www.bls.gov/emp/tables.htm) · [BLS reuse policy](https://www.bls.gov/bls/linksite.htm) |
| Anthropic Economic Index | Maxim Massenkoff and Peter McCrory (2026), *Labor market impacts of AI: A new measure and early evidence*. The source is the occupation-level job exposure file. This project maps occupations and derives ranks and a cross-platform mean. Anthropic identifies its **data as CC BY** and its code as MIT. [Study](https://www.anthropic.com/research/labor-market-impacts) · [Source file](https://huggingface.co/datasets/Anthropic/EconomicIndex/blob/7a1cc3c17cf7b0f88fc5c63f59059db751a9abae/labor_market_impacts/job_exposure.csv) · [Upstream license statement](https://huggingface.co/datasets/Anthropic/EconomicIndex#license) |
| Microsoft Research | Kiran Tomlinson, Sonia Jaffe, Will Wang, Scott Counts and Siddharth Suri (2025), *Working with AI*. The v1.1 occupation applicability scores are mapped to BLS occupations, ranked and averaged with the Claude rank. [Source and attribution](https://github.com/microsoft/working-with-ai/tree/v1.1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Theoretical exposure | Source measures by Edward Felten, Manav Raj and Robert Seamans (2021), Tyna Eloundou, Sam Manning, Pamela Mishkin and Daniel Rock (2024), and Andrea Eisfeldt, Gregor Schubert and Miao Ben Zhang (2023). The panel contains the derived theoretical rank. Consult each upstream source for its terms. [AIOE](https://github.com/AIOE-Data/AIOE) · [GPTs are GPTs](https://github.com/openai/GPTs-are-GPTs) · [Generative AI and Firm Values](https://github.com/gschubert/genai-firm-values-data) |

The [MIT License](LICENSE) covers this repository’s original Python code. Source data and derived data remain subject to the applicable upstream terms.
