#!/usr/bin/env bash
# run_mvtec_table10.sh — LogSAD reimplementation on MVTec AD (few-shot: 1/2/4-shot)
# Reproduces Table 10 of the LogSAD paper (CVPR 2025)
# Results saved to outputs/MVTec_AD/  (mirrors outputs/MVTec_LOCO/ structure)
set -euo pipefail

cd "$(dirname "$0")"

DATASET_PATH="${DATASET_PATH:-/home/gaya6/LogSAD/datasets/MVTec_AD}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
DEFAULT_PYTHON="/home/gaya6/miniconda3/envs/logsad/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
  else
    PYTHON_BIN="python"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/MVTec_AD}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/logs}"
RESULT_FILE="${RESULT_FILE:-${OUTPUT_DIR}/results.txt}"

CATEGORIES=(
  bottle
  cable
  capsule
  carpet
  grid
  hazelnut
  leather
  metal_nut
  pill
  screw
  tile
  toothbrush
  transistor
  wood
  zipper
)

DISPLAY_NAMES=(
  "Bottle"
  "Cable"
  "Capsule"
  "Carpet"
  "Grid"
  "Hazelnut"
  "Leather"
  "Metal Nut"
  "Pill"
  "Screw"
  "Tile"
  "Toothbrush"
  "Transistor"
  "Wood"
  "Zipper"
)

PROTOCOLS=(
  "1-shot"
  "2-shot"
  "4-shot"
)

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
for protocol in "${PROTOCOLS[@]}"; do
  mkdir -p "${LOG_ROOT}/${protocol}"
done

# ---------------------------------------------------------------------------
# extract_metrics <category> <log_file>
# Prints: img_f1  img_auroc  pix_f1  pix_auroc
# ---------------------------------------------------------------------------
extract_metrics() {
  local category="$1"
  local log_file="$2"

  "${PYTHON_BIN}" - "$category" "$log_file" <<'PY'
import re, sys

category, log_file = sys.argv[1], sys.argv[2]
with open(log_file, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

row = None
for line in text.splitlines():
    if re.search(rf"\|\s*{re.escape(category)}\s*\|", line):
        row = line

if row is None:
    raise SystemExit(f"Could not find metric row for '{category}'. Check {log_file}")

cells = [c.strip() for c in row.strip().strip("|").split("|")]
# cells: [category, k_shots, img_f1, img_auroc, pix_f1, pix_auroc]
if len(cells) < 6:
    raise SystemExit(f"Cannot parse metrics from: {row}")

print(cells[2], cells[3], cells[4], cells[5])
PY
}

# ---------------------------------------------------------------------------
# format_table  <all values...>
# Values layout: for each protocol, for each category: img_f1 img_auroc pix_f1 pix_auroc
# Produces two tables (image-level, pixel-level) written to RESULT_FILE.
# ---------------------------------------------------------------------------
format_table() {
  local tmp_file
  tmp_file="$(mktemp "${RESULT_FILE}.tmp.XXXXXX")"

  "${PYTHON_BIN}" - "$tmp_file" "$@" <<'PY'
import sys

result_file = sys.argv[1]
values      = sys.argv[2:]

protocols = ["1-shot", "2-shot", "4-shot"]
categories = [
    "Bottle", "Cable", "Capsule", "Carpet", "Grid",
    "Hazelnut", "Leather", "Metal Nut", "Pill", "Screw",
    "Tile", "Toothbrush", "Transistor", "Wood", "Zipper",
]
n_metrics = 4  # img_f1, img_auroc, pix_f1, pix_auroc

expected = len(protocols) * len(categories) * n_metrics
if len(values) != expected:
    raise SystemExit(f"Expected {expected} values, got {len(values)}")

def fmt(v):
    return f"{float(v):.1f}"

data = {}
cursor = 0
for protocol in protocols:
    data[protocol] = {}
    for cat in categories:
        data[protocol][cat] = tuple(values[cursor:cursor + n_metrics])
        cursor += n_metrics

def build_table(title, col_idx_pair, metric_headers):
    header1 = ["Protocol"] + [cat for cat in categories for _ in range(2)] + ["Average", ""]
    header2 = [""] + metric_headers * (len(categories) + 1)

    rows = []
    for protocol in protocols:
        row = [protocol]
        vals0, vals1 = [], []
        for cat in categories:
            v0 = float(data[protocol][cat][col_idx_pair[0]])
            v1 = float(data[protocol][cat][col_idx_pair[1]])
            vals0.append(v0)
            vals1.append(v1)
            row.extend([fmt(v0), fmt(v1)])
        row.extend([fmt(sum(vals0) / len(vals0)), fmt(sum(vals1) / len(vals1))])
        rows.append(row)

    all_rows = [header1, header2] + rows
    widths = [max(len(str(r[c])) for r in all_rows) for c in range(len(header1))]

    def line(items):
        return "| " + " | ".join(str(items[i]).ljust(widths[i]) for i in range(len(items))) + " |"

    def sep():
        return "|-" + "-|-".join("-" * w for w in widths) + "-|"

    return "\n".join([title, "", line(header1), sep(), line(header2), sep(),
                      *(line(r) for r in rows)])

img_table = build_table(
    "Table 10-A. Image-level F1-max and AUROC on MVTec AD (few-shot protocols).",
    (0, 1), ["F1-max", "AUROC"],
)
pix_table = build_table(
    "Table 10-B. Pixel-level F1-max and AUROC on MVTec AD (few-shot protocols).",
    (2, 3), ["F1-max", "AUROC"],
)

with open(result_file, "w", encoding="utf-8") as f:
    f.write(img_table + "\n\n" + pix_table + "\n")
PY

  mv "${tmp_file}" "${RESULT_FILE}"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ALL_VALUES=()

for shot in 1 2 4; do
  protocol="${shot}-shot"
  echo "Running LogSAD ${protocol} on MVTec AD."
  for i in "${!CATEGORIES[@]}"; do
    category="${CATEGORIES[$i]}"
    display="${DISPLAY_NAMES[$i]}"
    log_file="${LOG_ROOT}/${protocol}/${category}.log"

    echo "  [${protocol}] ${display}"
    "${PYTHON_BIN}" -u evaluation_MVTec.py \
      --module_path model_ensemble_few_shot_visa \
      --category    "${category}" \
      --dataset_path "${DATASET_PATH}" \
      --k_shot      "${shot}" \
      > "${log_file}" 2>&1

    read -r img_f1 img_auroc pix_f1 pix_auroc \
      < <(extract_metrics "${category}" "${log_file}")
    ALL_VALUES+=("${img_f1}" "${img_auroc}" "${pix_f1}" "${pix_auroc}")
  done
done

format_table "${ALL_VALUES[@]}"

echo
echo "Saved summary : ${RESULT_FILE}"
echo "Saved logs    : ${LOG_ROOT}/{1-shot,2-shot,4-shot}/"
