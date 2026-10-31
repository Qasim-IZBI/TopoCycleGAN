#!/bin/bash
#SBATCH --job-name=topo_field_table
#SBATCH --output=logs_topo/field_table_%j.out
#SBATCH --error=logs_topo/field_table_%j.err

#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1

# The stain-field tables (paper/tab1_fields.py): the field the selection
# protocol chose per dataset (main paper, Tab. 1) and the one it ranked last
# (Supp. Tab. S1), read from the trained runs' own training_meta.json, with the
# audits' AUROCs. Standard library only -- seconds; also fine on the login node:
#   bash tab1_fields.sh
#
#   AUDIT        the audit whose decision.json the chosen fields came from
#   WORST_AUDIT  topo-worst-field's --out (recommended_<marker>.env)
#
#   cd paper/slurm && mkdir -p logs_topo
#   sbatch --export=ALL,AUDIT=$R/<audit dir>,WORST_AUDIT=$R/<worst-field dir> tab1_fields.sh
#
# Output: ${OUT}/fields.csv -- give it to the paper's fill_results.py --manual.

set -eo pipefail

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
PAPER="${REPO_ROOT}/paper"
[ -d "$PAPER" ] || { echo "ERROR: no paper/ under REPO_ROOT=${REPO_ROOT}" >&2; exit 1; }
OUT=${OUT:-${ROOT}/paper_tables}
mkdir -p "$OUT"

ARGS=(--root "$ROOT" --out "${OUT}/fields.csv")
[ -n "${AUDIT:-}" ] && ARGS+=(--audit "$AUDIT")
[ -n "${WORST_AUDIT:-}" ] && ARGS+=(--worst-audit "$WORST_AUDIT")
[ -n "${RUN:-}" ] && ARGS+=(--run "$RUN")
python3 "${PAPER}/tab1_fields.py" "${ARGS[@]}"
