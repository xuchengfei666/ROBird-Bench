"""Isolated serial-suite primitives; all large artifacts live under the new E: root."""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from robird.budget_metrics_v2_1 import group_statistics, summarize_run, validate_records
from robird.data import ObservationFeatureDataset, collate_observations
from robird.evaluation import evaluate_loader
from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.models import DeepSetsClassifier, ModelOutput, build_model, masked_mean
from robird.training import seed_everything, train_one_epoch, validate


class ProbabilityMLP(DeepSetsClassifier):
    def forward(self, features, mask):
        probabilities = self.rho(self.phi(features)).softmax(dim=-1)
        pooled = masked_mean(probabilities, mask)
        return ModelOutput(logits=pooled.clamp_min(1e-12).log())


def make_model(name, dim):
    config = dict(name=name, feature_dim=dim, hidden_dim=512,
                  num_heads=4, num_layers=2, dropout=.1)
    if name == "probability_mlp":
        return ProbabilityMLP(dim, 512, 100, .1)
    return build_model(config, 100)


def atomic_binary(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def marker(path, contract):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["contract"] != contract:
        raise RuntimeError(f"Contract mismatch: {path}")
    for artifact, digest in value["artifacts"].items():
        if sha256_file(Path(artifact)) != digest:
            raise RuntimeError(f"Finished artifact changed: {artifact}")
    return value


def finish(path, contract, artifacts=(), **extra):
    value = dict(contract=contract, artifacts={str(Path(p).resolve()): sha256_file(Path(p))
                                             for p in artifacts}, **extra)
    atomic_write_json(Path(path), value, refuse_if_exists=True)
    return value


def features_resnet(manifest, directory, contract, batch_size=32):
    directory = Path(directory)
    complete = marker(directory/"done.json", contract)
    if complete:
        return np.load(directory/"features.npy", mmap_mode="r")
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50
    torch.hub.set_dir(str(directory.parent/"model_cache"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval().cuda()
    model.fc = nn.Identity()
    transform = weights.transforms()
    chunks = []
    for start in range(0, len(manifest), 512):
        part = manifest.iloc[start:start+512]
        path = directory/f"chunk_{start:06d}.npy"
        done = marker(path.with_suffix(".json"), contract)
        if not done:
            values = []
            for pos in range(0, len(part), batch_size):
                images = []
                for row in part.iloc[pos:pos+batch_size].itertuples():
                    data = Path(row.local_path).read_bytes()
                    if hashlib.sha256(data).hexdigest() != row.sha256:
                        raise RuntimeError(f"Image bytes changed: {row.photo_id}")
                    with Image.open(io.BytesIO(data)) as image:
                        images.append(transform(image.convert("RGB")))
                with torch.no_grad():
                    feat = model(torch.stack(images).cuda())
                    feat = nn.functional.normalize(feat, dim=1).cpu().numpy()
                values.append(feat)
            value = np.concatenate(values)
            if not np.isfinite(value).all() or value.shape != (len(part), 2048):
                raise RuntimeError("Invalid ResNet feature shard")
            atomic_binary(path, lambda f: np.save(f, value))
            finish(path.with_suffix(".json"), contract, [path],
                   start=start, photo_ids=part.photo_id.astype(int).tolist())
        elif done["photo_ids"] != part.photo_id.astype(int).tolist():
            raise RuntimeError("Feature shard photo-order mismatch")
        chunks.append(path)
    result = np.concatenate([np.load(p) for p in chunks])
    atomic_binary(directory/"features.npy", lambda f: np.save(f, result))
    index = pd.DataFrame(dict(photo_id=manifest.photo_id.tolist(), feature_row=np.arange(len(manifest))))
    atomic_write_csv(directory/"index.csv", index)
    weight_path = Path(torch.hub.get_dir())/"checkpoints"/Path(weights.url).name
    finish(directory/"done.json", contract,
           [directory/"features.npy", directory/"index.csv", weight_path],
           weights="ResNet50_Weights.IMAGENET1K_V2", weight_url=weights.url,
           torchvision_version=torchvision.__version__, transform=str(transform),
           shape=list(result.shape), l2_normalized=True)
    del model
    torch.cuda.empty_cache()
    return result


def dataset(manifest, features, split, budget, seed, mode):
    index = pd.DataFrame(dict(photo_id=manifest.photo_id.tolist(), feature_row=np.arange(len(manifest))))
    return ObservationFeatureDataset(manifest, features, index, split, budget, seed, mode, 32)


def loader(data, shuffle=False):
    return DataLoader(data, batch_size=64, shuffle=shuffle, num_workers=0,
                      pin_memory=True, collate_fn=collate_observations)


def train_run(manifest, features, spec, directory, contract, reuse=None, expected_hashes=None):
    directory = Path(directory)
    complete = marker(directory/"train_done.json", contract)
    if complete:
        return Path(complete["checkpoint"])
    seed = spec["seed"]
    seed_everything(seed)
    model = make_model(spec["model"], features.shape[1]).cuda()
    if reuse:
        payload = torch.load(reuse, map_location="cpu", weights_only=False)
        provenance = payload["provenance"]
        if provenance["seed"] != seed or provenance["training_budget"] is not None:
            raise RuntimeError("Legacy seed/budget mismatch")
        if provenance["model_config"]["name"] != spec["model"]:
            raise RuntimeError("Legacy model mismatch")
        for key, value in expected_hashes.items():
            if provenance.get(key) != value:
                raise RuntimeError(f"Legacy input provenance mismatch: {key}")
        model.load_state_dict(payload["model_state"], strict=True)
        finish(directory/"train_done.json", contract, [reuse],
               checkpoint=str(reuse), reused=True, spec=spec)
        return Path(reuse)
    train_data = dataset(manifest, features, "train", spec["budget"], seed, "random")
    val_data = dataset(manifest, features, "validation", None, seed, "exhaustive")
    train_loader, val_loader = loader(train_data, True), loader(val_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    criterion = nn.CrossEntropyLoss()
    last_path, best_path = directory/"latest.pth", directory/"best.pth"
    history, start, stale, best_loss = [], 0, 0, float("inf")
    if last_path.exists():
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if last["contract"] != contract or last["spec"] != spec:
            raise RuntimeError("Resume contract mismatch")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer"])
        torch.set_rng_state(last["rng_cpu"])
        torch.cuda.set_rng_state_all(last["rng_cuda"])
        np.random.set_state(last["rng_numpy"])
        random.setstate(last["rng_python"])
        history, start, stale, best_loss = last["history"], last["epoch"]+1, last["stale"], last["best_loss"]
    for epoch in range(start, 60):
        if stale >= 10:
            break
        train_data.set_epoch(epoch)
        loss = train_one_epoch(model, train_loader, optimizer, criterion, torch.device("cuda"))
        validation = validate(model, val_loader, torch.device("cuda"))
        if not np.isfinite([loss, validation["loss"]]).all():
            raise RuntimeError("Nonfinite training/validation")
        history.append(dict(epoch=epoch, train_loss=loss, **validation))
        if validation["loss"] < best_loss:
            best_loss, stale = validation["loss"], 0
            atomic_binary(best_path, lambda f: torch.save(
                dict(contract=contract, spec=spec, epoch=epoch, model_state=model.state_dict(),
                     validation=validation), f))
        else:
            stale += 1
        snapshot = dict(
            contract=contract, spec=spec, epoch=epoch, model_state=model.state_dict(),
            optimizer=optimizer.state_dict(), history=history, stale=stale, best_loss=best_loss,
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
            rng_numpy=np.random.get_state(), rng_python=random.getstate())
        atomic_binary(last_path, lambda f: torch.save(snapshot, f))
    atomic_write_json(directory/"history.json", history)
    finish(directory/"train_done.json", contract, [best_path, directory/"history.json"],
           checkpoint=str(best_path), reused=False, spec=spec)
    del model, optimizer
    torch.cuda.empty_cache()
    return best_path


def prediction_frame(model, manifest, features):
    records = []
    for k in range(1, 6):
        data = dataset(manifest, features, "development_test", k, 20260819, "exhaustive")
        if len(data):
            records.extend(evaluate_loader(model, loader(data), torch.device("cuda")))
    return pd.DataFrame([dict(
        observation_id=r["observation_id"], budget=r["budget"], label=r["label"],
        prediction=r["prediction"], photo_ids=";".join(map(str, r["photo_ids"])),
        probabilities_json=json.dumps(r["probabilities"].tolist(), separators=(",",":")))
        for r in records])


def evaluate_run(manifest, features, spec, checkpoint, directory, report_dir, contract):
    directory, report_dir = Path(directory), Path(report_dir)
    complete = marker(report_dir/"eval_done.json", contract)
    if complete:
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(spec["model"], features.shape[1]).cuda().eval()
    model.load_state_dict(payload["model_state"], strict=True)
    raw_path = directory/"predictions.csv"
    if raw_path.exists() and marker(directory/"predictions_done.json", contract):
        frame = pd.read_csv(raw_path)
    else:
        frame = prediction_frame(model, manifest, features)
        atomic_write_csv(raw_path, frame)
        finish(directory/"predictions_done.json", contract, [raw_path, checkpoint])
    groups = validate_records(frame, manifest[manifest.split == "development_test"])
    summary, budgets, edges = summarize_run(groups, repeats=2000, seed=20260908)
    atomic_write_json(report_dir/"metrics.json", dict(spec=spec, **summary))
    atomic_write_csv(report_dir/"groups.csv", budgets)
    atomic_write_csv(report_dir/"nested.csv", edges)
    finish(report_dir/"eval_done.json", contract,
           [report_dir/"metrics.json", report_dir/"groups.csv", report_dir/"nested.csv", raw_path])
    del model
    torch.cuda.empty_cache()


def shuffle_membership(manifest, seed):
    test = manifest[manifest.split == "development_test"].copy()
    test["n"] = test.groupby("observation_id").photo_id.transform("size")
    rng = np.random.default_rng(seed)
    shuffled, mapping = [], []
    for (_, cardinality), part in test.groupby(["taxon_id","n"], sort=True):
        ids = np.sort(part.observation_id.unique())
        if len(ids) < 3:
            continue
        rng.shuffle(ids)
        photos = {}
        for obs in ids:
            sub = part[part.observation_id == obs].sort_values("photo_id")
            photos[int(obs)] = sub.iloc[rng.permutation(len(sub))].to_dict("records")
        for i, obs in enumerate(ids):
            for position in range(int(cardinality)):
                donor = int(ids[(i+1+position%(len(ids)-1)) % len(ids)])
                row = dict(photos[donor][position])
                mapping.append(dict(slot_observation_id=int(obs), photo_id=int(row["photo_id"]),
                                    source_observation_id=donor, source_observer_id=int(row["observer_id"])))
                # Deliberate sentinel: this is a mixed pseudo-set, not a real observer.
                row.update(observation_id=int(obs), observer_id=-1)
                shuffled.append(row)
    return pd.DataFrame(shuffled).drop(columns="n"), pd.DataFrame(mapping)


def e3_run(manifest, features, spec, checkpoint, directory, report_dir, contract):
    directory, report_dir = Path(directory), Path(report_dir)
    complete = marker(report_dir/"e3_done.json", contract)
    if complete:
        return
    model = make_model(spec["model"], features.shape[1]).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"])
    real = pd.read_csv(report_dir/"groups.csv")
    pieces, artifacts = [], []
    photo_to_row = dict(zip(manifest.photo_id.astype(int), range(len(manifest))))
    for repeat in range(20):
        done_path = report_dir/f"e3_repeat_{repeat:02d}.json"
        done = marker(done_path, contract)
        if done:
            pieces.extend(done["comparisons"])
            artifacts.append(done_path)
            continue
        pseudo, mapping = shuffle_membership(manifest, 2026090800+repeat)
        aligned = np.asarray(features[[photo_to_row[int(p)] for p in pseudo.photo_id]])
        frame = prediction_frame(model, pseudo, aligned)
        groups = validate_records(frame, pseudo, num_classes=100)
        stats, _ = group_statistics(groups)
        raw_path = directory/f"e3_repeat_{repeat:02d}_predictions.csv"
        mapping_path = directory/f"e3_repeat_{repeat:02d}_mapping.csv"
        atomic_write_csv(raw_path, frame)
        atomic_write_csv(mapping_path, mapping)
        comparisons = []
        for k, part in stats.groupby("budget"):
            paired = real[(real.budget == k) & real.observation_id.isin(part.observation_id)]
            if set(paired.observation_id) != set(part.observation_id):
                raise RuntimeError("E3 cohort mismatch")
            a = float(paired.groupby("label").expected_accuracy.mean().mean())
            b = float(part.groupby("label").expected_accuracy.mean().mean())
            comparisons.append(dict(repeat=repeat, budget=int(k), groups=len(part),
                                    taxa=part.label.nunique(), macro_real=a, macro_shuffled=b,
                                    difference=b-a))
        finish(done_path, contract, [raw_path, mapping_path],
               comparisons=comparisons, uncertainty="randomization_distribution_not_confidence_interval")
        pieces.extend(comparisons)
        artifacts.append(done_path)
    atomic_write_csv(report_dir/"e3_randomizations.csv", pd.DataFrame(pieces))
    finish(report_dir/"e3_done.json", contract, [*artifacts, report_dir/"e3_randomizations.csv"])
    del model
    torch.cuda.empty_cache()


def group_similarity(manifest, features):
    # Descriptive covariate, not a quality/optical cause and never a tuning input.
    rows = []
    for obs, part in manifest.assign(feature_row=np.arange(len(manifest))).groupby("observation_id"):
        x = np.asarray(features[part.feature_row])
        sim = x @ x.T
        rows.append(dict(observation_id=int(obs), label=int(part.class_index.iloc[0]),
                         mean_pair_cosine=float(sim[np.triu_indices(len(x), 1)].mean()),
                         mean_area=float((part.width*part.height).mean())))
    return pd.DataFrame(rows)
