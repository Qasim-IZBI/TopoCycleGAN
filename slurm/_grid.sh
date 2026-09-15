# The sweep grid: the single source of truth for which cells exist.
#
# Sourced by _sweep_common.sh (training) and infer_sweep.sh (validation), so the
# two always agree on what task N means. Not executable on its own.
#
#   lambda_cycle  {10, 0}          CycleGAN L1 cycle weight
#   ph_cyc        {0, 1}           cycle-topology weight
#   ph_trans      {0, 1}           translation-topology weight
#   lambda_topo   {2e-4, 2e-2}     overall PH scale
#   field_B       hematoxylin/dab | dab/hematoxylin | dab+hematoxylin (sum)
#
# field_A is fixed at hematoxylin/eosin. The '/' forms deconvolve; a bare stain
# name projects and does NOT separate stains -- see fields.StainField.
#
# The full factorial is 48, but ph_cyc=0 AND ph_trans=0 switches off every
# topological term, making lambda_topo and both fields inert -- those 12 collapse
# to one anchor run per lambda_cycle. The 38 that remain are listed here.
#
#   tasks  0-18  lambda_cycle=10  (the critical path)
#   tasks 19-37  lambda_cycle=0   (does topology substitute for L1 cycle?)
#
# Format: lambda_cycle:ph_cyc:ph_trans:lambda_topo:field_B:field_combine

CELLS=(
  "10:0:0:0:hematoxylin/dab:max"   # anchor: no PH, fields inert
  "10:0:1:0.0002:hematoxylin/dab:max"
  "10:0:1:0.0002:dab/hematoxylin:max"
  "10:0:1:0.0002:dab+hematoxylin:sum"
  "10:0:1:0.02:hematoxylin/dab:max"
  "10:0:1:0.02:dab/hematoxylin:max"
  "10:0:1:0.02:dab+hematoxylin:sum"
  "10:1:0:0.0002:hematoxylin/dab:max"
  "10:1:0:0.0002:dab/hematoxylin:max"
  "10:1:0:0.0002:dab+hematoxylin:sum"
  "10:1:0:0.02:hematoxylin/dab:max"
  "10:1:0:0.02:dab/hematoxylin:max"
  "10:1:0:0.02:dab+hematoxylin:sum"
  "10:1:1:0.0002:hematoxylin/dab:max"
  "10:1:1:0.0002:dab/hematoxylin:max"
  "10:1:1:0.0002:dab+hematoxylin:sum"
  "10:1:1:0.02:hematoxylin/dab:max"
  "10:1:1:0.02:dab/hematoxylin:max"
  "10:1:1:0.02:dab+hematoxylin:sum"
  "0:0:0:0:hematoxylin/dab:max"    # anchor: no PH, fields inert
  "0:0:1:0.0002:hematoxylin/dab:max"
  "0:0:1:0.0002:dab/hematoxylin:max"
  "0:0:1:0.0002:dab+hematoxylin:sum"
  "0:0:1:0.02:hematoxylin/dab:max"
  "0:0:1:0.02:dab/hematoxylin:max"
  "0:0:1:0.02:dab+hematoxylin:sum"
  "0:1:0:0.0002:hematoxylin/dab:max"
  "0:1:0:0.0002:dab/hematoxylin:max"
  "0:1:0:0.0002:dab+hematoxylin:sum"
  "0:1:0:0.02:hematoxylin/dab:max"
  "0:1:0:0.02:dab/hematoxylin:max"
  "0:1:0:0.02:dab+hematoxylin:sum"
  "0:1:1:0.0002:hematoxylin/dab:max"
  "0:1:1:0.0002:dab/hematoxylin:max"
  "0:1:1:0.0002:dab+hematoxylin:sum"
  "0:1:1:0.02:hematoxylin/dab:max"
  "0:1:1:0.02:dab/hematoxylin:max"
  "0:1:1:0.02:dab+hematoxylin:sum"
)

# Resolve one cell into the variables both scripts use, RUN_NAME included.
# Environment overrides still win, so --export=ALL,LAMBDA_TOPO=0 works as before.
grid_select() {
    local id=$1
    if ! (( id < ${#CELLS[@]} )); then
        echo "task ${id} is past the end of the ${#CELLS[@]}-cell grid" >&2
        return 1
    fi
    IFS=: read -r LAMBDA_CYCLE PH_CYC PH_TRANS CELL_TOPO CELL_FIELD_B CELL_COMBINE \
        <<< "${CELLS[$id]}"
    LAMBDA_TOPO=${LAMBDA_TOPO:-$CELL_TOPO}
    FIELD_A=${FIELD_A:-hematoxylin/eosin}
    FIELD_B=${FIELD_B:-$CELL_FIELD_B}
    FIELD_COMBINE=${FIELD_COMBINE:-$CELL_COMBINE}
    FIELD_B_TAG=${FIELD_B//\//-}          # '/' is not safe in a directory name
    RUN_NAME="${MARKER}_lc${LAMBDA_CYCLE}_lt${LAMBDA_TOPO}_cyc${PH_CYC}_trans${PH_TRANS}_${FIELD_B_TAG}"
}
