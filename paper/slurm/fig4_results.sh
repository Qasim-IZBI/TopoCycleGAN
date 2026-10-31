#!/bin/bash
#SBATCH --job-name=topo_results_fig
#SBATCH --output=logs_topo/results_figure_%j.out
#SBATCH --error=logs_topo/results_figure_%j.err

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1

# The results figure (paper/fig4_results.py): every seed of every model, per
# dataset -- SSIM, LPIPS, FID, nuclei recall, task MAE (positive-nuclei % for
# the IHC markers, CPA for VS) and BM0 per nucleus.
#
# First the ensemble reports it reads (tools/ensemble_report.py; Cellpose-SAM
# nuclei), then the figure:
#   MIST markers  valA, test half   -> ${EVAL_ROOT}/ensemble_report/valA_test_cpsam/
#   BCI and VS    testA, whole      -> ${EVAL_ROOT}/ensemble_report/testA_all_cpsam/
# (VS has no nuclei: it only brings SSIM, LPIPS, FID and CPA MAE.) The reports
# also feed the paper's fill_results.py. SKIP_REPORT=1 plots existing reports.
#
#   cd paper/slurm && mkdir -p logs_topo
#   sbatch fig4_results.sh
#   sbatch --export=ALL,EVAL_ROOT=$R/eval_layout1_stop20k,FORMAT=png fig4_results.sh
#
# Output: ${OUT}/results.${FORMAT} -- copy to the paper's fig/results.pdf.

set -eo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
PAPER="${REPO_ROOT}/paper"
[ -d "$PAPER" ] || { echo "ERROR: no paper/ under REPO_ROOT=${REPO_ROOT}" >&2; exit 1; }
EVAL_ROOT=${EVAL_ROOT:-${ROOT}/eval_layout1_stop20k}
DETECTOR=${DETECTOR:-cpsam}
FORMAT=${FORMAT:-pdf}
OUT=${OUT:-${ROOT}/figure_results}
REPORTS="${EVAL_ROOT}/ensemble_report"
mkdir -p "$OUT"

if [ "${SKIP_REPORT:-0}" != "1" ]; then
    python "${REPO_ROOT}/tools/ensemble_report.py" --eval-root "$EVAL_ROOT" \
        --markers Ki67 ER HER2 PR --detector "$DETECTOR"
    python "${REPO_ROOT}/tools/ensemble_report.py" --eval-root "$EVAL_ROOT" \
        --markers BCI VS --split testA --detector "$DETECTOR"
fi

MEMBERS=()
for d in "valA_test_${DETECTOR}" "testA_all_${DETECTOR}"; do
    f="${REPORTS}/${d}/members.csv"
    if [ -f "$f" ]; then MEMBERS+=("$f"); else echo "[skip] no ${f}"; fi
done
(( ${#MEMBERS[@]} )) || { echo "ERROR: no ensemble report under ${REPORTS}" >&2; exit 1; }

python "${PAPER}/fig4_results.py" --members "${MEMBERS[@]}" --out "${OUT}/results.${FORMAT}"

echo
echo "To pull it back:  scp <cluster>:${OUT}/results.${FORMAT} <paper>/fig/results.pdf"
