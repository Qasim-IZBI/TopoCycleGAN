# The training grid for the paper: the single source of truth for the cells.
#
# Sourced by _sweep_common.sh (training) and infer_sweep.sh (inference), so the
# two always agree on what task N means. Not executable on its own.
#
#   lambda_cycle  fixed at 10 (the zoo's CycleGAN default)
#   lambda_topo   {2e-4, 2e-2, 2e-1}
#   ph_cyc        {0, 1}
#   ph_trans      {0, 1}
#
# The full factorial is 12, but ph_cyc=0 AND ph_trans=0 switches off every
# topological term, so lambda_topo is inert there -- those 3 collapse to a single
# cell, which IS the vanilla CycleGAN baseline. 10 cells remain.
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

CELLS=(
  "0:0:0"
  "0.0002:0:1"
  "0.0002:1:0"
  "0.0002:1:1"
  "0.02:0:1"
  "0.02:1:0"
  "0.02:1:1"
  "0.2:0:1"
  "0.2:1:0"
  "0.2:1:1"
)

grid_select() {
    local id=$1
    if ! (( id < ${#CELLS[@]} )); then
        echo "task ${id} is past the end of the ${#CELLS[@]}-cell grid" >&2
        return 1
    fi
    IFS=: read -r CELL_TOPO PH_CYC PH_TRANS <<< "${CELLS[$id]}"
    LAMBDA_CYCLE=${LAMBDA_CYCLE:-10}
    LAMBDA_TOPO=${LAMBDA_TOPO:-$CELL_TOPO}
    RUN_NAME="${MARKER}_lt${LAMBDA_TOPO}_cyc${PH_CYC}_trans${PH_TRANS}"
    if [ "$PH_CYC" = "0" ] && [ "$PH_TRANS" = "0" ]; then
        RUN_NAME="${MARKER}_baseline_cyclegan"
    fi
}
