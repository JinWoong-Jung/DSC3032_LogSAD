#!/usr/bin/env bash
# OWN CODE (Group 6): Full LOCO evaluation runner — not in the original LogSAD repo.
# Runs all 5 categories × {1/2/4-shot + full-data} and writes results to outputs/MVTec_LOCO/.
# Reproduces Table 9 of the LogSAD paper (CVPR 2025).
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/MVTec_LOCO}"
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/MVTec_LOCO}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_DIR}/logs}"
RESULT_FILE="${RESULT_FILE:-${OUTPUT_DIR}/results.txt}"

CATEGORIES=(
  breakfast_box
  juice_bottle
  pushpins
  screw_bag
  splicing_connectors
)

DISPLAY_NAMES=(
  "Breakfast Box"
  "Juice Bottle"
  "Pushpins"
  "Screw Bag"
  "Splicing Connectors"
)

PROTOCOLS=(
  "1-shot"
  "2-shot"
  "4-shot"
  "full-data"
)

mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
for protocol in "${PROTOCOLS[@]}"; do
  mkdir -p "${LOG_ROOT}/${protocol}"
done

extract_image_metrics() {
  local category="$1"
  local log_file="$2"

  "${PYTHON_BIN}" - "$category" "$log_file" <<'PY'
import re
import sys

category, log_file = sys.argv[1], sys.argv[2]
with open(log_file, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

row = None
for line in text.splitlines():
    if re.search(rf"\|\s*{re.escape(category)}\s*\|", line):
        row = line

if row is None:
    raise SystemExit(f"Could not find final metric row for {category}. Check {log_file}")

cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
if len(cells) < 4:
    raise SystemExit(f"Could not parse F1/AUROC image metrics from row: {row}")

print(cells[2], cells[3])
PY
}

format_table() {
  local tmp_file
  tmp_file="$(mktemp "${RESULT_FILE}.tmp.XXXXXX")"

  "${PYTHON_BIN}" - "$tmp_file" "$@" <<'PY'
import sys

result_file = sys.argv[1]
values = sys.argv[2:]

protocols = ["1-shot", "2-shot", "4-shot", "full-data"]
categories = [
    "Breakfast Box",
    "Juice Bottle",
    "Pushpins",
    "Screw Bag",
    "Splicing Connectors",
]

expected = len(protocols) * len(categories) * 2
if len(values) != expected:
    raise SystemExit(f"Expected {expected} metric values, got {len(values)}")

def fmt(value: float) -> str:
    return f"{value:.1f}"

rows = []
cursor = 0
for protocol in protocols:
    row = [protocol]
    f1_values = []
    auroc_values = []
    for _ in categories:
        f1 = float(values[cursor])
        auroc = float(values[cursor + 1])
        cursor += 2
        f1_values.append(f1)
        auroc_values.append(auroc)
        row.extend([fmt(f1), fmt(auroc)])
    row.extend([fmt(sum(f1_values) / len(f1_values)), fmt(sum(auroc_values) / len(auroc_values))])
    rows.append(row)

header1 = ["Protocol"]
for category in categories:
    header1.extend([category, ""])
header1.extend(["Average", ""])

header2 = [""]
for _ in categories + ["Average"]:
    header2.extend(["F1-max", "AUROC"])

all_rows = [header1, header2, *rows]
widths = [
    max(len(str(row[col])) for row in all_rows)
    for col in range(len(header1))
]

def line(items):
    return "| " + " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(items)) + " |"

def sep():
    return "|-" + "-|-".join("-" * width for width in widths) + "-|"

title = "Table 9. Image-level F1-max and AUROC results on MVTec LOCO in few-shot and full-data protocols."
table = "\n".join([title, "", line(header1), sep(), line(header2), sep(), *(line(row) for row in rows)])

with open(result_file, "w", encoding="utf-8") as f:
    f.write(table + "\n")
PY

  mv "${tmp_file}" "${RESULT_FILE}"
}

ALL_VALUES=()

for shot in 1 2 4; do
  protocol="${shot}-shot"
  echo "Running LogSAD ${protocol} on MVTec LOCO."
  for i in "${!CATEGORIES[@]}"; do
    category="${CATEGORIES[$i]}"
    display="${DISPLAY_NAMES[$i]}"
    log_file="${LOG_ROOT}/${protocol}/${category}.log"

    echo "  [${protocol}] ${display}"
    "${PYTHON_BIN}" -u evaluation.py \
      --module_path model_ensemble_few_shot \
      --category "${category}" \
      --dataset_path "${DATASET_PATH}" \
      --k_shot "${shot}" \
      > "${log_file}" 2>&1

    read -r f1 auroc < <(extract_image_metrics "${category}" "${log_file}")
    ALL_VALUES+=("${f1}" "${auroc}")
  done
done

echo "Running LogSAD full-data on MVTec LOCO."
for i in "${!CATEGORIES[@]}"; do
  category="${CATEGORIES[$i]}"
  display="${DISPLAY_NAMES[$i]}"
  coreset_log="${LOG_ROOT}/full-data/${category}_coreset.log"
  eval_log="${LOG_ROOT}/full-data/${category}.log"

  echo "  [full-data] ${display}: compute coreset"
  "${PYTHON_BIN}" -u compute_coreset.py \
    --module_path model_ensemble \
    --category "${category}" \
    --dataset_path "${DATASET_PATH}" \
    > "${coreset_log}" 2>&1

  echo "  [full-data] ${display}: evaluate"
  "${PYTHON_BIN}" -u evaluation.py \
    --module_path model_ensemble \
    --category "${category}" \
    --dataset_path "${DATASET_PATH}" \
    --k_shot 4 \
    > "${eval_log}" 2>&1

  read -r f1 auroc < <(extract_image_metrics "${category}" "${eval_log}")
  ALL_VALUES+=("${f1}" "${auroc}")
done

format_table "${ALL_VALUES[@]}"

echo
echo "Saved summary: ${RESULT_FILE}"
echo "Saved logs under: ${LOG_ROOT}/{1-shot,2-shot,4-shot,full-data}/"
