#!/bin/bash
#SBATCH --job-name=topo_cycsplit
#SBATCH --output=logs_topo/cycsplit_%A_%a.out
#SBATCH --error=logs_topo/cycsplit_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# Only the cells where ph_cyc is actually on -- see the note below.
#SBATCH --array=2,3,5,6,8,9,11,12

# The grid with the IHC cycle term compared PER DECONVOLVED STAIN CHANNEL:
# rec_B against real_B once for each of the two stains, rather than once on the
# merged stain1+stain2 field.
#
# This is the arm that has never been trained. The sweeps already in
# Outputs_<marker> ran with ph_cyc_split off -- the default -- so they are the
# merged-field arm, and this is what they get compared against. Nothing else
# changes: same grid, same schedule, same fields, same vectors, same
# TOPO_EVERY. If you find yourself tempted to raise TOPO_EVERY to claw back the
# extra cost below, don't -- it would make the two arms incomparable, which is
# the only reason either of them is being run.
#
#   mkdir -p logs_topo                      # SLURM will not create it for you
#   sbatch --export=ALL,MARKER=ER   sweep_cyc_split.sh
#   sbatch --export=ALL,MARKER=BCI --partition=paula sweep_cyc_split.sh
#   sbatch --export=ALL,MARKER=VS  --partition=paula sweep_cyc_split.sh
#
#   # on what the audit chose, as the original sweeps were submitted -- pass
#   # the SAME AUDIT_DIR the merged arm used, or the arms differ by their
#   # fields as well as by this flag and neither result means anything:
#   sbatch --export=ALL,MARKER=BCI,AUDIT_DIR=/work2/bz66izin-TopoCG/field_audit_v2 \
#          --partition=paula sweep_cyc_split.sh
#
# VS and BCI train on paula (sweep_vs.sh / sweep_bci.sh); the #SBATCH header
# here says clara for the MIST markers and --partition overrides it.
#
# EXPECT A SECOND SUBMIT. The split term computes two extra diagrams per step,
# ~+50% persistence cost, so a cell runs ~53h against this partition's 48h
# limit and will be killed at the wall short of 400k steps. That is survivable
# and needs no flag: the trainer writes step_latest.pt every 1000 steps and
# resume_if_exists() restarts from the furthest-ahead checkpoint, with the PH
# schedule restored from the checkpointed step buffer. So when a task times
# out, submit the SAME LINE AGAIN and it continues where it stopped -- at most
# 1000 steps are repeated. Check what actually finished before resubmitting:
#
#   sacct -n -X -j <jobid> -o JobID,State | grep TIMEOUT
#   grep -c . logs_topo/cycsplit_<jobid>_*.out
#
# ONLY 8 OF THE 13 CELLS ARE AFFECTED. ph_cyc_split changes nothing in a cell
# whose ph_cyc is 0, so tasks 0, 1, 4, 7 and 10 are already answered by the
# merged sweep and are not repeated here. Pass --array=0-12 to fill the grid
# out anyway; those five cells would be bit-for-bit repeats.
#
# Results go to their own BASE. RUN_NAME is a function of the grid cell alone,
# and the common body resumes from OUTPUT when it finds checkpoints there, so
# sharing a root with the merged sweep would not start a second run -- it would
# continue the merged one under a different loss, and the checkpoint would
# describe neither.

MARKER=${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}

# The point of this sweep. Set before _sweep_common.sh is sourced, so its
# ${PH_CYC_SPLIT:-0} keeps this value rather than the default.
PH_CYC_SPLIT=1

# Separate output root: a different loss from the runs in Outputs_<marker>,
# which must not be resumed or later mistaken for these.
MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')
BASE=${BASE:-/work2/bz66izin-TopoCG/Outputs_${MARKER_LC}_cycsplit}

# Sibling scripts live next to this one. SLURM runs a batch script from a copy
# in its spool directory, so BASH_SOURCE cannot locate them -- the directory
# sbatch was called from can. The fallback keeps a submit from the repo root
# working too.
SLURM_DIR=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
[ -f "${SLURM_DIR}/_sweep_common.sh" ] || SLURM_DIR="${SLURM_DIR}/slurm"

echo "IHC cycle term: PER STAIN CHANNEL (ph_cyc_split=1)"
echo "output root:    ${BASE}"
echo "note:           ~+50% persistence cost; expect a timeout and one resubmit"

source "${SLURM_DIR}/_sweep_common.sh"
