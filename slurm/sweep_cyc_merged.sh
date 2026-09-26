#!/bin/bash
#SBATCH --job-name=topo_cycmerged
#SBATCH --output=logs_topo/cycmerged_%A_%a.out
#SBATCH --error=logs_topo/cycmerged_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Only the cells where ph_cyc is actually on -- see the note below.
#SBATCH --array=2,3,5,6,8,9,11,12

# The grid again with the IHC cycle term on ONE MERGED FIELD: the same
# stain1+stain2 field the ph_trans terms are computed on, rather than the two
# deconvolved channels compared separately.
#
# READ THIS FIRST -- it decides whether you need this script at all.
#
# The merged field is what topo-train does by DEFAULT: ph_cyc_split is off
# (models.py), _sweep_common.sh defaults PH_CYC_SPLIT=0, and the audit's
# recommended_<marker>.env does not write that variable. With the flag off the
# cycle term runs
#     paired_diagram_loss(dgm_rec_B, dgm_real_B)
# on field module "B" -- cfg.field_B, merged by cfg.combine -- which is the
# very module the trans terms use for fake_B. Per-channel comparison is the
# OPT-IN, via --ph-cyc-split.
#
# So a sweep submitted without PH_CYC_SPLIT=1 in the environment has already
# trained the merged-field cycle term, and re-running it here would just spend
# a second 48h slot per cell reproducing it. The one way it could have been on
# is `sbatch --export=ALL` carrying a PH_CYC_SPLIT=1 out of the login shell.
# Check before submitting -- the training log prints the whole command line:
#
#   grep -c ph-cyc-split logs_topo/<the sweep's>.out    # 0 = already merged
#   echo $PH_CYC_SPLIT                                  # empty/0 = already merged
#
# If those say the flag was on, this is the script that redoes those cells with
# it off. If they say it was off, you already have this sweep.
#
# This pins PH_CYC_SPLIT=0 rather than leaning on the default, so an exported
# 1 cannot reach it through --export=ALL -- which is exactly the accident that
# would have produced the split runs in the first place.
#
#   mkdir -p logs_topo                      # SLURM will not create it for you
#   sbatch --export=ALL,MARKER=ER   sweep_cyc_merged.sh
#   sbatch --export=ALL,MARKER=BCI --partition=paula sweep_cyc_merged.sh
#   sbatch --export=ALL,MARKER=VS  --partition=paula sweep_cyc_merged.sh
#
#   # on what the audit chose, as the original sweeps were submitted:
#   sbatch --export=ALL,MARKER=BCI,AUDIT_DIR=/work2/bz66izin-TopoCG/field_audit_v2 \
#          --partition=paula sweep_cyc_merged.sh
#
# VS and BCI train on paula (sweep_vs.sh / sweep_bci.sh); the #SBATCH header
# here says clara for the MIST markers and --partition overrides it.
#
# ONLY 8 OF THE 13 CELLS ARE AFFECTED. ph_cyc_split changes nothing in a cell
# whose ph_cyc is 0, so tasks 0, 1, 4, 7 and 10 would be bit-for-bit repeats of
# runs you already have. The array above is the 8 cells with ph_cyc=1. Pass
# --array=0-12 to fill the grid out anyway; the five extra cells are there to
# be copied from the original sweep, not retrained.
#
# Results go to their own BASE. _sweep_common.sh resumes from OUTPUT when it
# finds checkpoints there, so sharing a directory with the original sweep would
# not produce a second run -- it would continue the first one under a different
# loss, and the checkpoint would describe neither.

MARKER=${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}

# The point of this sweep. Set before _sweep_common.sh is sourced, so its
# ${PH_CYC_SPLIT:-0} keeps this value, and an inherited 1 cannot win.
PH_CYC_SPLIT=0

# Separate output root: these are a different loss from the runs already in
# Outputs_<marker>, and must not resume them or be mistaken for them later.
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_cycmerged}

# Sibling scripts live next to this one. SLURM runs a batch script from a copy
# in its spool directory, so BASH_SOURCE cannot locate them -- the directory
# sbatch was called from can. The fallback keeps a submit from the repo root
# working too.
SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/_sweep_common.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"

echo "IHC cycle term: MERGED field (ph_cyc_split=0)"
echo "output root:    ${BASE}"

source "${SLURM_DIR}/_sweep_common.sh"
