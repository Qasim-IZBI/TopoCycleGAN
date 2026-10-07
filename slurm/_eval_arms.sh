# What array task N of an evaluation ARM scores -- not executable on its own.
#
# Sourced by evaluate.sh (FID / SSIM / LPIPS) and evaluate_nuclei.sh, so both
# file a model's results under the same <arm>/<name> and task N means the same
# model in either. Expects MARKER, ROOT and SPLIT; sets ARM, NAME and PRED (and
# exits clean, with a note, for a task the arm never trained).
#
#   ARM=main                  0-12 the TopoCycleGAN sweep in Outputs_<marker>,
#                             13 cyclegan small, 14 dclgan small
#   ARM=cycsplit              the ph_cyc=1 cells of Outputs_<marker>_cycsplit
#   ARM=worstfield_cycmerged  cells 1-12 of Outputs_<marker>_worstfield_cycmerged
#   ARM=worstfield_cycsplit   the ph_cyc=1 cells of ..._worstfield_cycsplit
#   ARM=baselines             0-2 cyclegan, 3-5 dclgan, small/medium/large
#   ARM=identity              task 0: the H&E input itself, scored as if it
#                             were the prediction -- the "do nothing" model
#
# PRED and NAME set by the caller score any other directory, under ARM=custom.

MARKER_LC=$(echo "$MARKER" | tr 'A-Z' 'a-z')

# Tile root per marker: TILES_<MARKER> wins if set, else TILES -- the same
# lookup training and inference use.
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
tiles_root() {
    local var="TILES_$1"
    echo "${!var:-$TILES}"
}

preds_root_of() {
    if [ "$SPLIT" = "valA" ]; then echo "$1/preds"; else echo "$1/preds_${SPLIT}"; fi
}

# A task the arm never trained is not a failure: exit clean, say why.
not_in_arm() {
    echo "ARM=${ARM} task ${TASK_ID}: $1 -- nothing to evaluate"
    exit 0
}

if [ -n "${PRED:-}" ]; then
    NAME=${NAME:?PRED is set -- set NAME too, it names the output directory}
    ARM=${ARM:-custom}
else
    ARM=${ARM:-main}
    TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch, or set PRED and NAME}
    _grid_dir=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
    [ -f "${_grid_dir}/_grid.sh" ] || _grid_dir="${_grid_dir}/slurm"
    source "${_grid_dir}/_grid.sh"
    n_cells=${#CELLS[@]}

    # grid_cell <root>: task N is grid cell N of the sweep under <root>.
    grid_cell() {
        (( TASK_ID < n_cells )) || not_in_arm "past the ${n_cells}-cell grid"
        grid_select "$TASK_ID"
        NAME="$RUN_NAME"
        PRED="$(preds_root_of "$1")/${RUN_NAME}"
    }
    # baseline_run <model> <size>: one train_baseline.sh run.
    baseline_run() {
        NAME="${MARKER}_$1_$2"
        PRED="$(preds_root_of "${ROOT}/Outputs_${MARKER_LC}_$1_$2")/${NAME}"
    }

    case "$ARM" in
        main)
            if (( TASK_ID < n_cells )); then
                grid_cell "${ROOT}/Outputs_${MARKER_LC}"
            else
                case $(( TASK_ID - n_cells )) in
                    0) baseline_run cyclegan small ;;
                    1) baseline_run dclgan   small ;;
                    *) not_in_arm "past the last task ($(( n_cells + 1 )))" ;;
                esac
            fi
            ;;
        cycsplit|worstfield_cycmerged|worstfield_cycsplit)
            grid_cell "${ROOT}/Outputs_${MARKER_LC}_${ARM}"
            # Mirror what the training scripts skip: no arm here trains the
            # field-free cell 0, and a split arm trains no ph_cyc=0 cell.
            # `if`, not `[ ] && ...`: a false test as the last command run
            # here would become the exit status of `source`, and the callers'
            # set -e would end every valid split-arm task, silently.
            if [ "$TASK_ID" = "0" ]; then
                not_in_arm "cell 0 is field-free; ARM=main has it"
            fi
            if [ "${ARM%cycsplit}" != "$ARM" ] && [ "$PH_CYC" = "0" ]; then
                not_in_arm "ph_cyc=0, so splitting changes nothing; ARM=main has it"
            fi
            ;;
        baselines)
            case "$TASK_ID" in
                0) baseline_run cyclegan small  ;;
                1) baseline_run cyclegan medium ;;
                2) baseline_run cyclegan large  ;;
                3) baseline_run dclgan   small  ;;
                4) baseline_run dclgan   medium ;;
                5) baseline_run dclgan   large  ;;
                *) not_in_arm "baselines are tasks 0-5" ;;
            esac
            ;;
        identity)
            # The input as its own prediction. Any metric that rewards it is
            # measuring structure kept, not IHC made -- this row says how much.
            if [ "$TASK_ID" != "0" ]; then
                not_in_arm "identity is task 0 only"
            fi
            NAME="${MARKER}_identity"
            PRED="$(tiles_root "$MARKER")/${MARKER}/TrainValAB/${SPLIT}"
            ;;
        *)
            echo "ERROR: unknown ARM '${ARM}'" >&2
            echo "  main, cycsplit, worstfield_cycmerged, worstfield_cycsplit, baselines or identity" >&2
            echo "  (or set PRED and NAME for any other directory)" >&2
            exit 1
            ;;
    esac
fi

# Sourced: end on success whatever branch ran, so `source` itself never
# returns non-zero into a caller running under set -e.
:
