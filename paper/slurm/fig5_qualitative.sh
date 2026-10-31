#!/bin/bash
#SBATCH --job-name=topo_qual_fig
#SBATCH --output=logs_topo/qual_figure_%j.out
#SBATCH --error=logs_topo/qual_figure_%j.err

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=clara
#SBATCH --exclude=clara[02,04-08]
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# The qualitative figure (paper/fig5_qualitative.py): one row per dataset -- H&E
# input, real target, TopoCycleGAN, CycleGAN, DCLGAN, CycleDiffusion -- every
# model on the same tile, from seed SEED's predictions in ENS_ROOT.
#
# One tile per dataset, by its path relative to the split directory:
#   TILE_Ki67, TILE_ER, TILE_HER2, TILE_PR, TILE_BCI, TILE_VS
# A dataset without a TILE_<M> is left out of the figure.
#
# Split per dataset: MIST markers valA (the test half of validation), BCI and
# VS their held-out testA. SPLIT_<M> overrides one. The real target is the
# same name in the matching B directory (valB / testB).
#
# CycleDiffusion has no ensemble predictions yet: CD_<M> is a prediction
# directory or a run directory / checkpoint (then it is run here, CD_STEPS DDIM
# steps); without it the column says n/a. Same for TOPO_<M>, CYC_<M>, DCL_<M>
# to override the ensemble's seed SEED.
#
#   cd paper/slurm && mkdir -p logs_topo
#   sbatch --export=ALL,TILE_Ki67=<t>.png,TILE_ER=<t>.png,TILE_HER2=<t>.png,TILE_PR=<t>.png,TILE_BCI=<t>.png,TILE_VS=<t>.png fig5_qualitative.sh
#   ... ,CD_Ki67=$R/Outputs_ki67_cyclediffusion_small/results/Ki67_cyclediffusion_small    # run CD here
#   ... ,CROP="64 64 128"          # the same 128 px window of every panel
#
# Output: ${OUT}/qual.${FORMAT}, qual_panels/ (every panel as PNG),
# qual_summary.json (each panel's source and layout r), qual_config.json.

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
ENS_ROOT=${ENS_ROOT:-${ROOT}/ensemble_layout1_stop20k}
SEED=${SEED:-0}
MARKERS=${MARKERS:-"Ki67 ER HER2 PR BCI VS"}
OUT=${OUT:-${ROOT}/figure_qual}
FORMAT=${FORMAT:-pdf}
TILES=${TILES:-${ROOT}/MIST_tiles}
TILES_BCI=${TILES_BCI:-${ROOT}/BCI_tiles}
TILES_VS=${TILES_VS:-${ROOT}/VS_tiles}
mkdir -p "$OUT"

CONFIG="${OUT}/qual_config.json"
export ROOT ENS_ROOT SEED MARKERS TILES TILES_BCI TILES_VS
python - "$CONFIG" <<'PY'
import json, os, sys
env = os.environ
models = [("TopoCycleGAN", "topo", "TOPO"), ("CycleGAN", "cyclegan", "CYC"),
          ("DCLGAN", "dclgan", "DCL"), ("CycleDiffusion", None, "CD")]
rows = []
for m in env["MARKERS"].split():
    tile = env.get("TILE_" + m)
    if not tile:
        print("[skip] %s: no TILE_%s" % (m, m)); continue
    split = env.get("SPLIT_" + m) or ("testA" if m in ("BCI", "VS") else "valA")
    root = env.get("TILES_" + m) or env["TILES"]
    he_dir = os.path.join(root, m, "TrainValAB", split)
    he, target = os.path.join(he_dir, tile), os.path.join(root, m, "TrainValAB", split[:-1] + "B", tile)
    for f in (he, target):
        if not os.path.isfile(f):
            sys.exit("no such tile: %s" % f)
    preds = "preds" if split == "valA" else "preds_" + split
    srcs = {}
    for name, model, var in models:
        src = env.get("%s_%s" % (var, m))
        if not src and model:
            src = os.path.join(env["ENS_ROOT"], m, model, preds, "seed" + env["SEED"])
        if src and os.path.exists(src):
            srcs[name] = src
        else:
            print("[n/a] %s / %s: %s" % (m, name, src or "not given"))
    rows.append({"label": "BCI" if m == "BCI" else m, "he": he, "target": target,
                 "he_root": he_dir, "models": srcs})
json.dump({"rows": rows, "models": [n for n, _, _ in models],
           "target_label": "Real IHC / SR"}, open(sys.argv[1], "w"), indent=2)
print("%d rows -> %s" % (len(rows), sys.argv[1]))
PY

python "${PAPER}/fig5_qualitative.py" --config "$CONFIG" --out "${OUT}/qual.${FORMAT}" \
    --cd-steps "${CD_STEPS:-50}" ${CROP:+--crop $CROP}

echo
echo "To pull it back:  tar czf figure_qual.tgz -C $(dirname "$OUT") $(basename "$OUT")"
