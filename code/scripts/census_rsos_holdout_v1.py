"""Bounded metadata-only feasibility pass; cannot download photos or declare G6."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import pandas as pd
from robird.io import atomic_write_json, sha256_file
from robird.rsos_suite_v1 import finish, marker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    directory = Path(config["data_root"])/"holdout_metadata"
    directory.mkdir(parents=True, exist_ok=True)
    exclude_obs, exclude_users = set(), set()
    exclusions = {}
    for relative in config["exclusion_manifests"]:
        path = root/relative
        exclusions[relative] = sha256_file(path)
        df = pd.read_csv(path)
        if "observation_id" in df:
            exclude_obs.update(df.observation_id.dropna().astype(int))
        if "observer_id" in df:
            exclude_users.update(df.observer_id.dropna().astype(int))
    manifest = pd.read_csv(root/"code/data/manifests/development_photos_v5_3.csv")
    taxa = manifest[["taxon_id","class_index"]].drop_duplicates().sort_values("class_index")
    reports, artifacts = [], []
    for taxon_id in taxa.taxon_id.astype(int):
        done_path = directory/f"taxon_{taxon_id}.json"
        done = marker(done_path, args.contract)
        if done:
            reports.append(done["result"])
            artifacts.append(done_path)
            continue
        candidates, pages, observed_ids = [], [], set()
        for page in range(1, 6):
            params = dict(taxon_id=taxon_id, quality_grade="research", photos="true",
                          photo_license="cc0,cc-by,cc-by-sa", created_d2="2026-09-08T00:00:00Z",
                          order_by="id", order="desc", per_page=200, page=page)
            path = directory/f"response_{taxon_id}_{page}.json"
            page_marker = marker(path.with_suffix(".done.json"), args.contract)
            if page_marker:
                response = json.loads(path.read_text(encoding="utf-8"))
            else:
                url = "https://api.inaturalist.org/v1/observations?"+urlencode(params)
                for attempt in range(3):
                    try:
                        req = Request(url, headers={"User-Agent":"ROBird-RSOS academic metadata feasibility"})
                        with urlopen(req, timeout=45) as handle:
                            response = json.load(handle)
                        if not isinstance(response.get("results"), list):
                            raise ValueError("Missing API results")
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep([2,5][attempt])
                atomic_write_json(path, response)
                finish(path.with_suffix(".done.json"), args.contract, [path], params=params)
                time.sleep(1)
            pages += [path, path.with_suffix(".done.json")]
            for obs in response["results"]:
                obs_id = int(obs["id"])
                if obs_id in observed_ids:
                    continue
                observed_ids.add(obs_id)
                observer = int(obs["user"]["id"])
                if (obs_id in exclude_obs or observer in exclude_users
                        or int(obs.get("taxon",{}).get("id",-1)) != taxon_id
                        or obs.get("quality_grade") != "research"):
                    continue
                photos = sorted([p for p in obs.get("photos", [])
                                 if p.get("license_code") in ("cc0","cc-by","cc-by-sa")
                                 and p.get("url") and p.get("attribution")], key=lambda p:int(p["id"]))
                if len(photos) < 2:
                    continue
                candidates.append(dict(observation_id=obs_id, observer_id=observer, taxon_id=taxon_id,
                                       created_at=obs.get("created_at"), observed_on=obs.get("observed_on"),
                                       photos=photos[:5]))
            if len(response["results"]) < 200:
                break
        unique_users = len({c["observer_id"] for c in candidates})
        result = dict(taxon_id=taxon_id, candidate_groups=len(candidates),
                      independent_observers=unique_users, shortfall=max(0,10-unique_users),
                      candidates=candidates)
        finish(done_path, args.contract, pages, result=result)
        reports.append(result)
        artifacts.append(done_path)
        print(json.dumps(dict(taxon_id=taxon_id, independent_observers=unique_users)), flush=True)
    shortfalls = [{k:r[k] for k in ("taxon_id","independent_observers","shortfall")}
                  for r in reports if r["shortfall"]]
    finish(directory/"done.json", args.contract, artifacts,
           status="METADATA_CENSUS_COMPLETE_NO_G6_DECISION", taxa=len(reports),
           individually_feasible=sum(r["shortfall"] == 0 for r in reports),
           shortfalls=shortfalls, global_assignment_evaluated=False,
           exclusion_hashes=exclusions, images_downloaded=0,
           formal_class_table_written=False, original_g6_pass=False)


if __name__ == "__main__":
    main()
