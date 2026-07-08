import re
from pathlib import Path


_DATASET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"fashionmnist|fashion[-_]?mnist|f[-_]?mnist"), "fmnist"),
    (re.compile(r"cifar100"), "cifar100"),
    (re.compile(r"cifar10"), "cifar10"),
    (re.compile(r"stl10|(^|[^a-z0-9])stl([^a-z0-9]|$)"), "stl10"),
    (re.compile(r"svhn"), "svhn"),
    (re.compile(r"usps"), "usps"),
    (re.compile(r"mnist"), "mnist"),
)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def infer_logical_dataset_name(zoo_path: str) -> str:
    path = Path(zoo_path)
    candidates: list[str] = []
    current = path
    for _ in range(3):
        if current.name:
            candidates.append(current.name.lower())
        parent = current.parent
        if parent == current:
            break
        current = parent

    for candidate in candidates:
        for pattern, canonical_name in _DATASET_PATTERNS:
            if pattern.search(candidate):
                return canonical_name

    if len(path.parts) >= 3:
        return _normalize_name(path.parent.parent.name)
    return _normalize_name(path.name)