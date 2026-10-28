# The training grid for the paper: the single source of truth for the cells.
#
# Sourced by _sweep_common.sh (training) and infer_sweep.sh (inference), so the
# two always agree on what task N means. Not executable on its own.
#
#   lambda_cycle  fixed at 10 (the zoo's CycleGAN default)
#   lambda_topo   {2e-4, 2e-3, 2e-2, 2e-1}
#   ph_cyc        {0, 1}
#   ph_trans      {0, 1}
#
# The full factorial is 16, but ph_cyc=0 AND ph_trans=0 switches off every
# topological term, so lambda_topo is inert there -- those 4 collapse to a single
# cell, which IS the vanilla CycleGAN baseline. 13 cells remain.
#
# The weights are one decade apart and chosen against the measured gradient
# ratio: at lambda_topo=1 the PH term carries ~4160x the cycle gradient, so
# 2e-4 is parity, 2e-3 is ~8x, 2e-2 is ~83x and 2e-1 is ~830x. The low end has
# to be at parity -- if the best cell sat on the grid's edge there would be no
# way to tell whether the search went low enough.
#
# Task 0 is that baseline. It runs the same code path as every other cell --
# same data, schedule, capacity and optimiser, only the loss differs -- so it is
# a fair comparison rather than a re-implementation.
#
# The field hyperparameters are NOT swept: they were fixed beforehand by
# topo-validate-fields on held-out validation tiles (field_A stain1/stain2,
# field_B stain1+stain2 summed, downsample 2, dims 0 1, projection birth).
#
# Format: lambda_topo:ph_cyc:ph_trans
#
# GRID=trans picks the paper's rerun grid instead: ph_trans alone at lambda_topo
# 1e-4 .. 1e-1, one decade apart (tasks 0-3), trained with the layout term so
# no cell is lost to inversion. Everything that sources this file -- training,
# inference, evaluation -- must see the same GRID.
#
# GRID=ablate is the rerun's step 2, at the lambda_topo the trans grid chose
# (1e-2, selection_sweep_cpsam): task 0 ph_trans alone, task 1 + ph_cyc.
# Task 0 on the audit's field IS the trans grid's cell 2, so under sweep_layout
# only task 1 is trained; task 0 is the worst-field control
# (sweep_worst_field.sh, its own SWEEP_ROOT).

case "${GRID:-full}" in
full)
CELLS=(
  "0:0:0"
  "0.0002:0:1"
  "0.0002:1:0"
  "0.0002:1:1"
  "0.002:0:1"
  "0.002:1:0"
  "0.002:1:1"
  "0.02:0:1"
  "0.02:1:0"
  "0.02:1:1"
  "0.2:0:1"
  "0.2:1:0"
  "0.2:1:1"
)
;;
trans)
CELLS=(
  "0.0001:0:1"
  "0.001:0:1"
  "0.01:0:1"
  "0.1:0:1"
)
;;
ablate)
CELLS=(
  "0.01:0:1"
  "0.01:1:1"
)
;;
*)
echo "ERROR: GRID=${GRID}; full, trans or ablate" >&2
return 1 2>/dev/null || exit 1
;;
esac

# A submitter's --export=ALL,LAMBDA_TOPO=... overrides the cell. Remember that
# override once, here, rather than reading it back off LAMBDA_TOPO inside
# grid_select: the first call would otherwise assign the cell's own value and
# every later call would treat it as an override, so a loop over the grid --
# which is how compare.sh enumerates the cells -- would report task 0's weight
# for all 13. One array task calling grid_select once never saw it.
_GRID_LAMBDA_TOPO=${LAMBDA_TOPO:-}
_GRID_LAMBDA_CYCLE=${LAMBDA_CYCLE:-}

grid_select() {
    local id=$1
    if ! (( id < ${#CELLS[@]} )); then
        echo "task ${id} is past the end of the ${#CELLS[@]}-cell grid" >&2
        return 1
    fi
    IFS=: read -r CELL_TOPO PH_CYC PH_TRANS <<< "${CELLS[$id]}"
    LAMBDA_CYCLE=${_GRID_LAMBDA_CYCLE:-10}
    LAMBDA_TOPO=${_GRID_LAMBDA_TOPO:-$CELL_TOPO}
    RUN_NAME="${MARKER}_lt${LAMBDA_TOPO}_cyc${PH_CYC}_trans${PH_TRANS}"
    if [ "$PH_CYC" = "0" ] && [ "$PH_TRANS" = "0" ]; then
        RUN_NAME="${MARKER}_baseline_cyclegan"
    fi
}
