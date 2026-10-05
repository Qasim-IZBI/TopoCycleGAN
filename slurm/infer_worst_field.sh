#!/bin/bash
#SBATCH --job-name=topo_infer_worst
#SBATCH --output=logs_topo/infer_worstfield_%A_%a.out
#SBATCH --error=logs_topo/infer_worstfield_%A_%a.err

#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Cell 0 is not trained by sweep_worst_field.sh -- the baseline is independent
# of the field, and the one in Outputs_<marker> is already inferred there.
#SBATCH --array=1-12

# Validation inference for the field audit's negative control: task N infers
# the model sweep_worst_field.sh task N trained, from the same output root.
#
# This is infer_sweep.sh with BASE pointed at the worst-field root. The field
# itself is not needed: it is baked into the checkpoint's config.
#
#   sbatch --export=ALL,MARKER=ER infer_worst_field.sh
#
#   # the per-channel arm only trained the ph_cyc=1 cells:
#   sbatch --export=ALL,MARKER=ER,PH_CYC_SPLIT=1 \
#          --array=2,3,5,6,8,9,11,12 infer_worst_field.sh
#
#   # VS and BCI trained on paula, so infer there too:
#   sbatch --partition=paula --export=ALL,MARKER=VS  infer_worst_field.sh
#   sbatch --partition=paula --export=ALL,MARKER=BCI infer_worst_field.sh
#
# Every other infer_sweep.sh knob (SPLIT, VAL_A, SUBDIR, LIMIT, TILES_<M>)
# passes straight through.

: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"

PH_CYC_SPLIT=${PH_CYC_SPLIT:-0}
case "$PH_CYC_SPLIT" in
    0) variant="cycmerged" ;;
    1) variant="cycsplit" ;;
    *) echo "ERROR: PH_CYC_SPLIT must be 0 or 1, got '${PH_CYC_SPLIT}'" >&2; exit 1 ;;
esac

# Kept in step with the root sweep_worst_field.sh trains into.
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_worstfield_${variant}}

echo "FIELD CONTROL: inferring the audit's worst-ranked field"
echo "  ph_cyc      ${variant} (PH_CYC_SPLIT=${PH_CYC_SPLIT})"
echo "  output root ${BASE}"

SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/infer_sweep.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"
source "${SLURM_DIR}/infer_sweep.sh"
