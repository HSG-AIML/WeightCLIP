#!/usr/bin/env bash
# Run dataset_to_model.py across multiple seeds and aggregate mean±std.
#
# Usage:
#   scripts/multi_seed_d2m.sh -s "0 1 2 3 4" -- --mode memory_bank \
#       --sane-ckpt /path/to/checkpoint.pt --arch resnet18slim
#
# Defaults to seeds 0..4 if -s/--seeds is omitted. Everything after `--` is
# forwarded to dataset_to_model.py verbatim. The script overrides --seed and
# --output-dir per run; do not pass those after `--`.
set -euo pipefail

SEEDS="0 1 2 3 4"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--seeds) SEEDS="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1 (separate dataset_to_model args after --)" >&2
            exit 1 ;;
    esac
done
EXTRA_ARGS=("$@")

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
    echo "No dataset_to_model.py args provided. Pass them after --." >&2
    exit 1
fi

OUT_ROOT=$(mktemp -d -t d2m_multiseed_XXXXXX)

# Share one embedding cache across all seeds. If the user already passed
# --cache-dir we honor it. Otherwise we derive the standard
# dataset_to_model.py default: <sane_ckpt_parent>/dataset_to_model_<arch>/embedding_cache
# so any cache from previous runs is picked up.
USER_PASSED_CACHE_DIR=0
SANE_CKPT=""
ARCH="cnn3"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[i]}" in
        --cache-dir) USER_PASSED_CACHE_DIR=1 ;;
        --sane-ckpt) SANE_CKPT="${EXTRA_ARGS[i+1]:-}" ;;
        --arch)      ARCH="${EXTRA_ARGS[i+1]:-cnn3}" ;;
    esac
done

CACHE_ARGS=()
SHARED_CACHE=""
if [[ $USER_PASSED_CACHE_DIR -eq 0 ]]; then
    if [[ -n "$SANE_CKPT" ]]; then
        if [[ -d "$SANE_CKPT" ]]; then
            CKPT_FILE="$SANE_CKPT/checkpoint.pt"
        else
            CKPT_FILE="$SANE_CKPT"
        fi
        SHARED_CACHE="$(dirname "$CKPT_FILE")/dataset_to_model_${ARCH}/embedding_cache"
    else
        SHARED_CACHE="$OUT_ROOT/embedding_cache"
    fi
    mkdir -p "$SHARED_CACHE"
    CACHE_ARGS=(--cache-dir "$SHARED_CACHE")
fi

echo "Output root: $OUT_ROOT"
echo "Seeds:       $SEEDS"
if [[ -n "$SHARED_CACHE" ]]; then
    echo "Shared cache: $SHARED_CACHE"
fi
echo "Forwarded:   ${EXTRA_ARGS[*]}"

for s in $SEEDS; do
    OUT="$OUT_ROOT/seed_$s"
    echo
    echo "=== Seed $s ==> $OUT ==="
    python3 scripts/dataset_to_model.py "${EXTRA_ARGS[@]}" "${CACHE_ARGS[@]}" --seed "$s" --output-dir "$OUT"
done

python3 - "$OUT_ROOT" $SEEDS <<'PY'
import json, sys
from pathlib import Path
from statistics import mean, stdev

root = Path(sys.argv[1])
seeds = sys.argv[2:]
METRICS = ("pre_mean", "epoch_1_mean", "post_mean")
SECTIONS = ("direct_decode", "neighbour", "memory_bank", "scratch")

agg = {}  # ds -> metric -> [values]
for s in seeds:
    path = root / f"seed_{s}" / "results.json"
    if not path.exists():
        print(f"[warn] missing {path}", file=sys.stderr)
        continue
    payload = json.loads(path.read_text())
    for section_name in SECTIONS:
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        for ds, stats in section.get("datasets", {}).items():
            if not isinstance(stats, dict):
                continue
            entry = agg.setdefault(ds, {m: [] for m in METRICS})
            for m in METRICS:
                if m in stats:
                    entry[m].append(float(stats[m]))

def fmt(vs):
    if not vs:
        return "    n/a    "
    if len(vs) == 1:
        return f"{vs[0]:6.2f}± 0.00"
    return f"{mean(vs):6.2f}±{stdev(vs):5.2f}"

print()
print(f"Seeds: {','.join(seeds)}")
print(f"{'Dataset':<28}  {'pre':>13}  {'ep1':>13}  {'final':>13}")
print("-" * 76)
grand = {m: [] for m in METRICS}
for ds in sorted(agg):
    row = agg[ds]
    print(f"{ds:<28}  {fmt(row['pre_mean']):>13}  {fmt(row['epoch_1_mean']):>13}  {fmt(row['post_mean']):>13}")
    for m in METRICS:
        grand[m].extend(row[m])
print("-" * 76)
print(f"{'AVERAGE':<28}  {fmt(grand['pre_mean']):>13}  {fmt(grand['epoch_1_mean']):>13}  {fmt(grand['post_mean']):>13}")

(root / "aggregate.json").write_text(json.dumps({
    "seeds": seeds,
    "per_dataset": {
        ds: {m: {"values": vs,
                 "mean": (mean(vs) if vs else None),
                 "std": (stdev(vs) if len(vs) > 1 else 0.0)}
             for m, vs in row.items()}
        for ds, row in agg.items()
    },
    "grand_average": {
        m: {"mean": (mean(vs) if vs else None),
            "std": (stdev(vs) if len(vs) > 1 else 0.0)}
        for m, vs in grand.items()
    },
}, indent=2))
print()
print(f"Aggregate saved to {root}/aggregate.json")
PY
