#!/usr/bin/env python3
"""Build ``dataset.pt`` files from raw image folders, in the format the zoo
trainers consume.

The raw Train / Test images from the TANS repo are laid out per task as::

    <src>/<task>/{tr,va,te}/<class>/<image files>

    Train: https://www.dropbox.com/s/mvkyb7qsdmx5cud/raw_m_train.tar.gz?dl=0
    Test:  https://www.dropbox.com/s/jaiq173z0fruzw4/raw_m_test.tar.gz?dl=0

For every task this writes::

    <out>/<task>/dataset.pt
      = {"trainset", "valset", "testset"} -> CachedDataset with
          .data     float32 [N, 3, 32, 32] normalized to [-1, 1]
          .targets  long    [N]
          .transform = None

which is exactly what ``train_cnn3_datasetpt_zoo.py`` and
``train_resnet18slim_datasetpt_zoo.py`` (and the WeightCLIP dataset encoder) load.

Preprocessing:  Resize((32,32)) -> RGB -> ToTensor([0,1]) -> Normalize(0.5,0.5) => [-1,1].
Class indices are aligned across tr/va/te using the train split's class ordering.

CIFAR-10 is a Test task but is *not* part of the raw TANS download, so pass
``--with-cifar10`` (or list ``cifar10`` in ``--only``) to materialize it from
torchvision in the same format -- no manual download needed.

Usage:
    # Train zoos (CNN + ResNet share the same images)
    python scripts/build_dataset_pts.py --src /path/Train --out /path/dataset_pts

    # Test / OOD (includes cifar10, fetched from torchvision)
    python scripts/build_dataset_pts.py --src /path/Test --out /path/dataset_pts --with-cifar10

    # cifar10 only
    python scripts/build_dataset_pts.py --out /path/dataset_pts --with-cifar10
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder


# dataset.pt payload types
#
# These MUST live at module top level so that instances pickle as
# ``__main__.CachedDataset`` / ``__main__.GrayscaleToRGB`` when this file is run
# as a script.  Those are the exact class paths every loader in this repo
# registers when unpickling dataset.pt (see sane.data.datasets.image_set_dataset,
# sane.loss.alignment, and the zoo trainers).  Do not move them into a package.
class CachedDataset(Dataset):
    def __init__(self) -> None:
        self.transform = None
        self.data: torch.Tensor = torch.empty(0, 3, 32, 32)
        self.targets: torch.Tensor = torch.empty(0, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, idx: int):
        img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


class GrayscaleToRGB:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x


SPLIT_MAP = [("tr", "trainset"), ("va", "valset"), ("te", "testset")]
TF = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                                   # [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
])


def canonical_name(name: str) -> str:
    """Normalize a raw folder name to the repo's dash convention (``ct_images`` -> ``ct-images``).

    Matches ``canonical_dataset_name`` in the trainers so the built folders are
    found by ``--dataset-pt-root``.
    """
    return name.strip().lower().replace(" ", "-").replace("_", "-")


def _cache(data: torch.Tensor, targets: torch.Tensor) -> CachedDataset:
    cd = CachedDataset()
    cd.data = data.float()
    cd.targets = targets.long()
    return cd


def _tensors_from_dataset(ds: Dataset, remap: dict[int, int] | None, workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(ds, batch_size=512, num_workers=workers, shuffle=False)
    data, targets = [], []
    for xb, yb in loader:
        data.append(xb)
        if remap is None:
            targets.append(yb.long())
        else:
            targets.append(torch.tensor([remap[int(t)] for t in yb], dtype=torch.long))
    if not data:
        return torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)
    return torch.cat(data), torch.cat(targets)


def build_split(split_dir: Path, canon: dict[str, int], workers: int) -> CachedDataset | None:
    if not split_dir.is_dir():
        return None
    ds = ImageFolder(str(split_dir), transform=TF)   # default loader -> PIL.convert("RGB")
    remap = {i: canon[c] for i, c in enumerate(ds.classes)}   # align to train class order
    data, targets = _tensors_from_dataset(ds, remap, workers)
    return _cache(data, targets) if len(targets) else None


def _save(payload: dict, out_dir: Path, num_classes: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "dataset.pt")
    return {"num_classes": num_classes, **{k: int(len(payload[k])) for k in payload}}


def build_task(task_dir: Path, out_dir: Path, workers: int) -> dict | None:
    tr_dir = task_dir / "tr"
    if not tr_dir.is_dir():
        return None
    canon = {name: i for i, name in enumerate(sorted(p.name for p in tr_dir.iterdir() if p.is_dir()))}
    payload = {}
    for src_split, key in SPLIT_MAP:
        cd = build_split(task_dir / src_split, canon, workers)
        if cd is not None and len(cd) > 0:
            payload[key] = cd
    if "trainset" not in payload:
        return None
    return _save(payload, out_dir, len(canon))


def build_cifar10(out_dir: Path, workers: int, val_frac: float = 0.1, seed: int = 0) -> dict:
    """Materialize CIFAR-10 from torchvision into the same dataset.pt format.

    trainset/valset come from the 50k train split (deterministic ``val_frac``
    hold-out); testset is the 10k test split.
    """
    download_root = out_dir.parent / "_cifar10_download"
    train = CIFAR10(str(download_root), train=True, download=True, transform=TF)
    test = CIFAR10(str(download_root), train=False, download=True, transform=TF)

    xtr, ytr = _tensors_from_dataset(train, None, workers)
    xte, yte = _tensors_from_dataset(test, None, workers)

    perm = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed))
    n_val = int(len(ytr) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    payload = {"trainset": _cache(xtr[tr_idx], ytr[tr_idx]), "valset": _cache(xtr[val_idx], ytr[val_idx]), "testset": _cache(xte, yte)}
    return _save(payload, out_dir, num_classes=10)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="Raw root: <src>/<task>/{tr,va,te}/<class>/*. Omit to build only cifar10.")
    ap.add_argument("--out", required=True, help="Output root: <out>/<task>/dataset.pt")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers for encoding images")
    ap.add_argument("--only", default=None, help="Comma-separated task names to build (matches raw or normalized names)")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild even if dataset.pt exists")
    ap.add_argument("--with-cifar10", action="store_true", help="Also build cifar10/dataset.pt from torchvision")
    ap.add_argument("--raw-names", action="store_true", help="Keep raw folder names; default normalizes to the ct-images dash form")
    ap.add_argument("--limit", type=int, default=None, help="(debug) only the first N tasks")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    only = set(args.only.split(",")) if args.only else None

    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    tasks: list[tuple[str, Path]] = []
    if args.src:
        src = Path(args.src)
        for d in sorted(p for p in src.iterdir() if p.is_dir()):
            name = d.name if args.raw_names else canonical_name(d.name)
            if only is not None and d.name not in only and name not in only:
                continue
            tasks.append((name, d))
    if args.limit:
        tasks = tasks[: args.limit]

    build_cifar = args.with_cifar10 or (only is not None and "cifar10" in only)
    print(f"{len(tasks)} raw task(s){' + cifar10' if build_cifar else ''} -> {out}")

    def record(name: str, info: dict | None, note: str) -> None:
        if info is not None:
            manifest[name] = info
        out.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps(manifest, indent=2))
        print(note)

    for i, (name, task_dir) in enumerate(tasks):
        od = out / name
        if (od / "dataset.pt").exists() and not args.overwrite:
            print(f"[{i + 1}/{len(tasks)}] {name}  (exists, skip)")
            continue
        try:
            info = build_task(task_dir, od, args.workers)
            if info is None:
                print(f"[{i + 1}/{len(tasks)}] {name}  SKIP (no tr split under {task_dir})")
                continue
            record(name, info, f"[{i + 1}/{len(tasks)}] {name}  classes={info['num_classes']} "
                               f"tr={info.get('trainset')} va={info.get('valset')} te={info.get('testset')}")
        except Exception:  # noqa: BLE001
            print(f"[{i + 1}/{len(tasks)}] {name}  ERROR\n{traceback.format_exc()}")

    if build_cifar:
        od = out / "cifar10"
        if (od / "dataset.pt").exists() and not args.overwrite:
            print("cifar10  (exists, skip)")
        else:
            info = build_cifar10(od, args.workers)
            record("cifar10", info, f"cifar10  classes={info['num_classes']} "
                                    f"tr={info['trainset']} va={info['valset']} te={info['testset']}")

    print(f"\nDone. {len(manifest)} datasets in manifest -> {mpath}")


if __name__ == "__main__":
    main()
