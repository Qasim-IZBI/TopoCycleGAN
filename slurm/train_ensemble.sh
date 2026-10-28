#!/bin/bash
#SBATCH --job-name=topo_ensemble
#SBATCH --output=logs_topo/ensemble_%A_%a.out
#SBATCH --error=logs_topo/ensemble_%A_%a.err

#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-29  # 3 models x 10 seeds; see below

# Deep ensembles of the chosen TopoCycleGAN and the two small baselines: ten
# members each, identical but for the seed. They give the seed variance of
# every result, how often each model fails to train, and the per-pixel
# disagreement the zoo's uncertainty.py turns into uncertainty maps.
#
# ONE ARRAY PER MARKER. Task N = model * 10 + seed:
#
#    0- 9  TopoCycleGAN, the configuration split_eval.sh chose
#          (lambda_topo 2e-3, ph_cyc 1, ph_trans 1, merged field; BEST_RUN)
#   10-19  CycleGAN small
#   20-29  DCLGAN small
#
# Seed 0 of each is the run that already exists -- task 0/10/20 links it into
# the ensemble layout and exits. Seeds 1-9 train.
#
# A new member is the seed-0 run repeated with another seed, and nothing else:
#   TopoCycleGAN  every setting -- field, stain vectors, schedule, lambdas,
#                 steps, lr -- is read off the seed-0 run's training_meta.json
#                 (tools/run_args.py) and passed to topo-train with --seed
#   baselines     train_baseline.sh at its defaults, as seed 0 was, with SEED
# `bash train_ensemble.sh status` confirms it afterwards, member by member.
#
#   mkdir -p logs_topo
#   for M in Ki67 ER HER2 PR; do
#       sbatch --export=ALL,MARKER=$M train_ensemble.sh
#   done
#   sbatch --partition=paula --export=ALL,MARKER=BCI train_ensemble.sh
#
#   # %N caps how many of one array run at once, if the queue needs sharing:
#   sbatch --array=0-29%10 --export=ALL,MARKER=ER train_ensemble.sh
#
# THE 48h WALL. Every member resumes from its own checkpoints, so a member
# killed at the wall continues when its task is submitted again. `status`
# lists every member short of its step budget and prints the --array that
# resubmits exactly those (finished members would exit at once anyway):
#
#   MARKERS="Ki67 ER HER2 PR BCI" bash train_ensemble.sh status
#
# Output: ${ENS_ROOT}/<marker>/<model>/results/seed<k>/{checkpoints,samples,...}
#         -- the same results/<run> layout as Outputs_*, so infer_sweep.sh and
#         infer_baseline.sh read a member with BASE=<...>/<model>, RUN_NAME=seed<k>.
#         seed0 is a link to the existing run (and preds/seed0 to its predictions).

set -eo pipefail

ROOT=${ROOT:-/work2/bz66izin-TopoCG}
ENS_ROOT=${ENS_ROOT:-${ROOT}/ensemble}
REPO_ROOT=${REPO_ROOT:-/home/sc.uni-leipzig.de/bz66izin/TopoCG_Project/TopoCycleGAN}
REPO=${REPO:-${REPO_ROOT}/slurm}
BEST_RUN=${BEST_RUN:-lt0.002_cyc1_trans1}   # the chosen cell, by its run-name suffix
BEST_ARM_ROOT=${BEST_ARM_ROOT:-}            # default Outputs_<marker> (merged, best field)
MODELS=(topo cyclegan dclgan)
PY3=${PY3:-python3}
# A schedule VARIANT of the TopoCycleGAN members, not a repeat: TOPO_START /
# TOPO_WARMUP override what seed 0 recorded. Seed 0 then trains too (the
# existing run has the old schedule), and the variant must live under its own
# ENS_ROOT so it never mixes with the real ensemble:
#
#   sbatch --array=0-9 --export=ALL,MARKER=Ki67,TOPO_START=0,ENS_ROOT=/work2/bz66izin-TopoCG/ensemble_start0 train_ensemble.sh
#
# LAMBDA_LAYOUT adds the layout-consistency term (topo_i2i/layout.py) that
# stops a run inverting tissue and lumen -- to all three models, so they stay
# comparable. LAYOUT_STOP switches it off after that many steps: the inversion
# is settled within ~3k steps, and kept on for a whole run the term costs
# nuclei precision and SSIM. It is still logged (loss_layout) after the stop.
#
#   sbatch --export=ALL,MARKER=Ki67,LAMBDA_LAYOUT=1,LAYOUT_STOP=20000,ENS_ROOT=/work2/bz66izin-TopoCG/ensemble_layout1_stop20k train_ensemble.sh
#
# STEPS cuts every member short -- a SCREEN: whether a run inverts is settled
# in its first few thousand steps, so 10k tells it at ~2.5% of a full run. A
# screen saves a checkpoint at its last step; infer it with infer_ensemble.sh
# given the same STEPS, then count the inverted members with
# tools/invert_check.py.
#
#   sbatch --array=0-29 --time=06:00:00 --export=ALL,MARKER=Ki67,LAMBDA_LAYOUT=1,STEPS=10000,ENS_ROOT=/work2/bz66izin-TopoCG/screen_layout1 train_ensemble.sh
#
# LINK_SEED0=1 links seed 0 under a layout variant too, when the chosen run was
# itself trained with the same layout term -- the rerun's sweep cells are (seed
# 0, LAMBDA_LAYOUT=1, LAYOUT_STOP=20000), so retraining it would repeat it:
#
#   sbatch --array=0-9 --export=ALL,MARKER=ER,LAMBDA_LAYOUT=1,LAYOUT_STOP=20000,LINK_SEED0=1,\
#       BEST_ARM_ROOT=/work2/bz66izin-TopoCG/sweep_layout/er,BEST_RUN=lt0.01_cyc0_trans1,\
#       ENS_ROOT=/work2/bz66izin-TopoCG/ensemble_layout1_stop20k train_ensemble.sh
#
# Its layout.json must record the same weight and stop, or the task refuses.
TOPO_START=${TOPO_START:-}
TOPO_WARMUP=${TOPO_WARMUP:-}
STEPS=${STEPS:-}
LAMBDA_LAYOUT=${LAMBDA_LAYOUT:-}
LAYOUT_STOP=${LAYOUT_STOP:-}
VARIANT=0
[ -n "${TOPO_START}${TOPO_WARMUP}${STEPS}${LAMBDA_LAYOUT}${LAYOUT_STOP}" ] && VARIANT=1
if [ -n "$LAYOUT_STOP" ] && [ -z "$LAMBDA_LAYOUT" ]; then
    echo "ERROR: LAYOUT_STOP needs LAMBDA_LAYOUT" >&2; exit 1
fi

# Where each model's seed-0 run and its predictions live.
seed0_paths() {   # <model> -> sets SRC_RUN, SRC_PREDS
    local m_lc; m_lc=$(echo "$MARKER" | tr 'A-Z' 'a-z')
    case "$1" in
        topo)
            local base=${BEST_ARM_ROOT:-${ROOT}/Outputs_${m_lc}}
            SRC_RUN="${base}/results/${MARKER}_${BEST_RUN}"
            SRC_PREDS="${base}/preds/${MARKER}_${BEST_RUN}" ;;
        cyclegan|dclgan)
            SRC_RUN="${ROOT}/Outputs_${m_lc}_$1_small/results/${MARKER}_$1_small"
            SRC_PREDS="${ROOT}/Outputs_${m_lc}_$1_small/preds/${MARKER}_$1_small" ;;
    esac
}

# How far a run has got: the newest numbered checkpoint, or -- further along
# between those 100k-step saves -- the last step the trainer logged to
# loss_log.csv (every log_steps; a resumed run continues from that point too).
last_step() {
    local f n=0 logged
    for f in "$1"/checkpoints/step_[0-9]*.pt; do
        [ -e "$f" ] || continue
        f=${f##*step_}; f=${f%.pt}
        (( f > n )) && n=$f
    done
    if [ -f "$1/loss_log.csv" ]; then
        logged=$(tail -n 1 "$1/loss_log.csv" | cut -d, -f1 | tr -d '\r')
        case "$logged" in ''|*[!0-9]*) ;; *) (( logged > n )) && n=$logged ;; esac
    fi
    echo "$n"
}

# -----------------------------
# status: every member's progress, and whether it is a true repeat of seed 0
# -----------------------------
if [ "${1:-}" = "status" ]; then
    # A member still queued has written nothing and shows as '-'; one running
    # shows its progress but is not finished. Resubmitting either would put a
    # second job on the same member directory, writing the same checkpoints.
    # So no resubmit line is printed while any ensemble job is in flight.
    in_flight=0
    if command -v squeue >/dev/null 2>&1; then
        in_flight=$(squeue -h -u "$USER" -n topo_ensemble -t PENDING,RUNNING,REQUEUED,CONFIGURING \
                        2>/dev/null | wc -l | tr -d " ")
    fi
    for MARKER in ${MARKERS:-${MARKER:?set MARKER or MARKERS}}; do
        echo "== ${MARKER}"
        resubmit=()
        for mi in 0 1 2; do
            model=${MODELS[$mi]}
            seed0_paths "$model"
            target=$(grep -o '"total_steps": *[0-9]*' "${SRC_RUN}/training_meta.json" 2>/dev/null \
                     | grep -o '[0-9]*$' || echo 400000)
            target=${STEPS:-$target}
            line="  ${model}"
            for k in 0 1 2 3 4 5 6 7 8 9; do
                run="${ENS_ROOT}/${MARKER}/${model}/results/seed${k}"
                if [ ! -d "$run" ]; then
                    line="${line}  s${k}:-"
                    resubmit+=( $(( mi * 10 + k )) )
                    continue
                fi
                step=$(last_step "$run")
                mark=""
                if [ "$k" != "0" ] && [ -f "${run}/training_meta.json" ] && \
                   ! "$PY3" "${REPO_ROOT}/tools/run_args.py" --compare \
                        "${SRC_RUN}/training_meta.json" "${run}/training_meta.json" >/dev/null 2>&1; then
                    mark="!"     # settings differ from seed 0 -- not a true repeat
                fi
                if (( step >= target )); then
                    line="${line}  s${k}:done${mark}"
                else
                    line="${line}  s${k}:$(( step / 1000 ))k${mark}"
                    resubmit+=( $(( mi * 10 + k )) )
                fi
            done
            echo "$line"
        done
        if (( ${#resubmit[@]} )) && (( in_flight > 0 )); then
            echo "  ${#resubmit[@]} member(s) not finished -- no resubmit while jobs are in flight"
        elif (( ${#resubmit[@]} )); then
            echo "  resubmit:  sbatch --array=$(IFS=,; echo "${resubmit[*]}") --export=ALL,MARKER=${MARKER} train_ensemble.sh"
        else
            echo "  all 30 members at their step budget"
        fi
    done
    if (( in_flight > 0 )); then
        echo "NOTE: ${in_flight} ensemble job line(s) still queued or running. Resubmitting now"
        echo "      would start a second job on a member that already has one; run status"
        echo "      again once squeue shows none -- then the resubmit lines are safe."
    fi
    echo "('-' not started, Nk = steps reached, ! = settings differ from seed 0:"
    echo " run tools/run_args.py --compare <seed0>/training_meta.json <member>/training_meta.json)"
    exit 0
fi

# -----------------------------
# one member
# -----------------------------
: "${MARKER:?set MARKER, e.g. --export=ALL,MARKER=ER}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch -- there is no array index}
mi=$(( TASK_ID / 10 )); SEED=$(( TASK_ID % 10 ))
(( mi < 3 )) || { echo "task ${TASK_ID}: tasks are 0-29"; exit 0; }
model=${MODELS[$mi]}
seed0_paths "$model"
MODEL_DIR="${ENS_ROOT}/${MARKER}/${model}"
OUTPUT="${MODEL_DIR}/results/seed${SEED}"

echo "ensemble ${MARKER} / ${model} / seed ${SEED}"
echo "  seed-0 run  ${SRC_RUN}"
echo "  member      ${OUTPUT}"

if [ ! -f "${SRC_RUN}/training_meta.json" ]; then
    echo "ERROR: the seed-0 run has no training_meta.json -- ${SRC_RUN}" >&2
    exit 1
fi

if (( VARIANT )); then
    if [ "$model" != "topo" ] && [ -n "${TOPO_START}${TOPO_WARMUP}" ]; then
        echo "TOPO_START/TOPO_WARMUP apply to TopoCycleGAN (tasks 0-9) only" >&2; exit 1
    fi
    [ "$ENS_ROOT" != "${ROOT}/ensemble" ] || {
        echo "ERROR: a variant needs its own ENS_ROOT, not ${ENS_ROOT}" >&2; exit 1; }
fi

LINK_SEED0=${LINK_SEED0:-0}
if [ "$SEED" = "0" ] && (( VARIANT )) && [ "$LINK_SEED0" = "1" ]; then
    [ "$model" = "topo" ] || { echo "LINK_SEED0 applies to TopoCycleGAN (tasks 0-9) only" >&2; exit 1; }
    [ -z "${TOPO_START}${TOPO_WARMUP}${STEPS}" ] || {
        echo "ERROR: LINK_SEED0 with TOPO_START/TOPO_WARMUP/STEPS -- the run is not that variant" >&2; exit 1; }
    "$PY3" - "${SRC_RUN}/layout.json" "${LAMBDA_LAYOUT:-0}" "${LAYOUT_STOP:-0}" <<'PYEOF' || {
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except OSError:
    sys.exit("no layout.json: the run was trained without the layout term")
ok = float(r["lambda_layout"]) == float(sys.argv[2]) and int(r.get("stop_step") or 0) == int(sys.argv[3])
sys.exit(0 if ok else "layout.json records lambda %s, stop %s" % (r["lambda_layout"], r.get("stop_step")))
PYEOF
        echo "ERROR: LINK_SEED0: ${SRC_RUN} was not trained with LAMBDA_LAYOUT=${LAMBDA_LAYOUT} LAYOUT_STOP=${LAYOUT_STOP:-none}" >&2
        exit 1; }
fi
if [ "$SEED" = "0" ] && { (( ! VARIANT )) || [ "$LINK_SEED0" = "1" ]; }; then
    # The existing run IS seed 0: link it, and its predictions, into the layout.
    mkdir -p "${MODEL_DIR}/results" "${MODEL_DIR}/preds"
    ln -sfn "$SRC_RUN" "$OUTPUT"
    echo "  linked seed 0: ${SRC_RUN}"
    if [ -d "$SRC_PREDS" ]; then
        ln -sfn "$SRC_PREDS" "${MODEL_DIR}/preds/seed0"
        echo "  and its predictions: ${SRC_PREDS}"
    else
        echo "  (no predictions at ${SRC_PREDS} yet -- seed 0 is inferred with the rest)"
    fi
    exit 0
fi

TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}

if [ "$model" != "topo" ]; then
    # train_baseline.sh at its defaults -- as seed 0 was -- with this seed,
    # into the ensemble layout. It resumes, loads the env and checks the data.
    # The tile roots go with it explicitly: it has defaults of its own, and the
    # two scripts must not be able to disagree about the training data.
    # A variant's STEPS and layout settings go with it; otherwise its defaults.
    MODEL=$model SIZE=small MARKER=$MARKER SEED=$SEED \
        BASE="$MODEL_DIR" RUN_NAME="seed${SEED}" \
        TILES="$TILES" TILES_BCI="$TILES_BCI" TILES_VS="$TILES_VS" \
        STEPS="${STEPS:-400000}" SAVE_STEPS="${STEPS:-100000}" LAMBDA_LAYOUT="$LAMBDA_LAYOUT" \
        LAYOUT_STOP="$LAYOUT_STOP" bash "${REPO}/train_baseline.sh"
    exit 0
fi

# ---- TopoCycleGAN: the seed-0 run's own settings, plus --seed
if command -v module >/dev/null 2>&1; then
    module purge
    module load Anaconda3/2025.06-1
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-topocg}"
fi
nvidia-smi || true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# Persistence is CPU-bound and single-threaded per image -- as in the sweep.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CPUS=${SLURM_CPUS_PER_TASK:-4}
NUM_WORKERS=$(( CPUS > 2 ? CPUS - 2 : 1 ))

var="TILES_${MARKER}"; TILE_ROOT=${!var:-$TILES}
DATA_DIR="${TILE_ROOT}/${MARKER}/TrainValAB"

mkdir -p "$OUTPUT"
RUN_ARGS=()
while IFS= read -r a; do RUN_ARGS+=("$a"); done < <(
    "$PY3" "${REPO_ROOT}/tools/run_args.py" "${SRC_RUN}/training_meta.json" \
        --stains-out "${OUTPUT}/stains.json")
(( ${#RUN_ARGS[@]} )) || { echo "ERROR: no settings read from ${SRC_RUN}" >&2; exit 1; }
# After seed 0's own flags, so these win (argparse keeps the last).
[ -n "$TOPO_START" ]  && RUN_ARGS+=(--topo-start-step "$TOPO_START")
[ -n "$TOPO_WARMUP" ] && RUN_ARGS+=(--topo-warmup-steps "$TOPO_WARMUP")
[ -n "$STEPS" ]         && RUN_ARGS+=(--steps "$STEPS" --save-steps "$STEPS")
[ -n "$LAMBDA_LAYOUT" ] && RUN_ARGS+=(--lambda-layout "$LAMBDA_LAYOUT")
[ -n "$LAYOUT_STOP" ]   && RUN_ARGS+=(--layout-stop "$LAYOUT_STOP")

echo "Running command:"
printf ' %q' topo-train --dataA "${DATA_DIR}/trainA/" --dataB "${DATA_DIR}/trainB/" \
    --output "$OUTPUT" --image-size 256 --num-workers "$NUM_WORKERS" --seed "$SEED" "${RUN_ARGS[@]}"
echo
topo-train --dataA "${DATA_DIR}/trainA/" --dataB "${DATA_DIR}/trainB/" \
    --output "$OUTPUT" --image-size 256 --num-workers "$NUM_WORKERS" --seed "$SEED" \
    "${RUN_ARGS[@]}"

echo "Done: ${MARKER} / ${model} / seed ${SEED}"
