"""Build matched occupation components from the Economic Index and the study crosswalk."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

base = Path(__file__).resolve().parent
output = base
from urllib.request import urlopen
import hashlib
sources = {'task_penetration.csv': ('https://huggingface.co/datasets/Anthropic/EconomicIndex/raw/0771328fb2cd2d492686260a7ac098628a03b2bc/labor_market_impacts/task_penetration.csv', '85bee872db1d55d3e9a7f4e89da5ae4a5d59aa8ec875d728fbf4b7d820984616'), 'onet_task_statements.csv': ('https://huggingface.co/datasets/Anthropic/EconomicIndex/raw/4e5f69fab13a9c3f7d0dcc31a6f893a9818738f6/release_2025_03_27/onet_task_statements.csv', '82e4c418dc08475ef1263ad1e8c66ffa3357291370cd35ee1065ecea590597ec')}
cache = base / "mode_source_cache"
cache.mkdir(exist_ok=True)
for filename, (url, checksum) in sources.items():
    path = cache / filename
    if not path.exists():
        with urlopen(url, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("Upstream file differs from the registered source: " + filename)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Cached file differs from the registered source: " + filename)
mapping = pd.read_csv(base / "mode_mapping.csv")
normalize = lambda text: text.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
tasks = pd.read_csv(cache / "onet_task_statements.csv")
scores = pd.read_csv(cache / "task_penetration.csv")
scores["key"] = normalize(scores.task)
assert scores.groupby("key").penetration.nunique().max() == 1
scores = scores.drop_duplicates("key")
crosswalk = mapping[["onet_code","ep_code"]].drop_duplicates()
crosswalk = crosswalk.loc[~crosswalk.onet_code.duplicated(keep=False)]
tasks = tasks.merge(crosswalk, left_on="O*NET-SOC Code", right_on="onet_code", validate="many_to_one")
tasks["key"] = normalize(tasks.Task)
tasks = tasks.drop_duplicates(["ep_code", "key"])
panel_codes = pd.read_csv(output / "primary_panel.csv").occupation_code
tasks = tasks.loc[tasks.ep_code.isin(panel_codes)].copy()
tasks = tasks.merge(scores[["key", "penetration"]], on="key", how="left", validate="many_to_one")
matched = tasks.dropna(subset=["penetration"]).copy()
p = matched.penetration
positive = p.gt(0)
assert (p[positive].between(.5, 1)).all() and (p >= 0).all()
matched["auto"] = np.where(positive, 2*p - 1, 0)
matched["aug"] = np.where(positive, 2*(1-p), 0)
np.testing.assert_allclose(matched.auto + .5*matched.aug, p)
np.testing.assert_allclose(matched.auto + matched.aug, positive.astype(float))
out = matched.groupby("ep_code")[["auto","aug"]].mean()
out["tasks_matched"] = matched.groupby("ep_code").size()
out["tasks_listed"] = tasks.groupby("ep_code").size()
out = out.join(mapping[["ep_code","anthropic_observed","microsoft_applicability"]].drop_duplicates().set_index("ep_code"), validate="one_to_one")
assert out.notna().all().all()
out.index.name = "occupation_code"
out.to_csv(output / "mode_inputs.csv")
print(json.dumps(dict(occupations=len(out), listed_pairs=len(tasks), matched_pairs=len(matched),
                      listed_occupations=tasks.ep_code.nunique(), scored_zero=int((p==0).sum()))))
