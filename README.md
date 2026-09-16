# topo-i2i

TopoGAN's topological loss ([Wang et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480120.pdf))
applied to unpaired virtual staining, layered on top of
[I2I-Stain-Zoo](https://github.com/HoehmeLab/I2I-Stain-Zoo) **without modifying it**.

## Install

```bash
pip install -e .          # pulls the zoo at a pinned commit, plus gudhi
```

## Preparing tiles

For datasets that already ship tiles (`trainA/ trainB/ valA/ valB/`) rather than
whole slides, where the zoo's WSI `tile.py` does not apply:

```bash
topo-crop --input  /work2/bz66izin-TopoCG/MIST/Ki67/TrainValAB/ \
          --output /work2/bz66izin-TopoCG/MIST_tiles/Ki67/TrainValAB/ \
          --tile_size 512 --resize_to 256
```

That is the command the sweep script's default paths assume: MIST Ki67 ships
1024x1024 tiles, cut into four 512x512 quadrants and resampled to 256x256.

It mirrors the directory tree, so the result feeds `topo-train` unchanged. Flag
names match the zoo's `tile.py`, so the two are interchangeable in a pipeline.

| flag | default | meaning |
|---|---|---|
| `--input` / `--output` | *required* | source root and mirrored destination |
| `--subdirs` | `trainA trainB valA valB` | subfolders to process |
| `--tile_size` | `512` | crop size taken from the source image |
| `--resize_to` | *= tile_size* | resample each crop to this size |
| `--overlap` | `0` | overlap between neighbouring crops, in source pixels |
| `--tissue_threshold` | `0.0` | drop crops below this tissue fraction (0 keeps everything) |
| `--white_level` | `220` | grayscale value above which a pixel counts as background |
| `--ext` | `.png` | output extension |
| `--num_workers` | all cores | |
| `--dry_run` | off | report what would be written without writing it |

**On scale.** `--resize_to` changes the microns per pixel of the output: cutting
512 and resizing to 256 halves the resolution relative to cutting 256 directly,
and the script prints the factor so it is on the record. The zoo's WSI pipeline
defaults to `tile_size=256, resize_to=None`, i.e. 256 px at the slide's native
resolution — to match that from 1024 px source tiles of comparable µm/px, use
`--tile_size 256` (16 crops per image) rather than 512→256. Resolution decides
whether adjacent nuclei remain separate connected components, so it changes what
the persistent-homology terms measure; a `--lambda-topo` tuned at one scale will
not transfer to another.

Any remainder on the right/bottom edge is dropped rather than padded, so every
tile covers real tissue at the same scale.

## Train

```bash
topo-train --dataA tiles/HE --dataB tiles/IHC --output runs/topo01 \
           --field-A hematoxylin --field-B dab+hematoxylin --field-combine max \
           --lambda-topo 1.0 --lambda-ph-cyc 1.0 --lambda-ph-trans 1.0 \
           --topo-downsample 2
```

`--lambda-topo 0` gives the plain CycleGAN baseline through the identical code path.

## All flags

`topo-train --help` is authoritative; this is the same surface with the reasoning.

### Data and run

| flag | default | meaning |
|---|---|---|
| `--dataA` | *required* | source-domain tiles (H&E) |
| `--dataB` | *required* | target-domain tiles (IHC / Sirius Red) |
| `--output` | `runs/topo` | run directory; checkpoints and samples go under it |
| `--steps` | `100000` | training steps |
| `--batch-size` | `4` | |
| `--image-size` | `256` | tile size fed to the model |
| `--num-workers` | `4` | dataloader workers |
| `--lr` | `2e-4` | Adam learning rate, betas fixed at (0.5, 0.999) |
| `--amp` | off | mixed precision |
| `--save-steps` | `25000` | permanent checkpoint interval (`step_latest.pt` is written every step) |
| `--log-steps` | `1000` | logging interval |

Training resumes automatically from `--output` if checkpoints are already there.

### Loss weights

| flag | default | meaning |
|---|---|---|
| `--lambda-topo` | `1.0` | overall scale on all PH terms; **`0` gives the plain CycleGAN baseline** through the identical code path |
| `--lambda-ph-cyc` | `1.0` | weight on the two cycle terms (`L_PH-cyc,H` + `L_PH-cyc,I`) |
| `--lambda-ph-trans` | `1.0` | weight on the two translation terms (`L_PH-trans,H` + `L_PH-trans,I`) |

Adversarial, cycle-L1 and identity weights come from `CycleGANConfig`
(`lambda_cycle=10.0`, `lambda_identity=0.5`) and are not exposed on this CLI.

### Which channels

| flag | default | meaning |
|---|---|---|
| `--preset` | `he-ihc` | `he-ihc` → A=`hematoxylin`, B=`dab+hematoxylin`; `he-sr` → A=`eosin`, B=`dab` |
| `--field-A` | from preset | override domain A: `gray`, a stain name, or `a+b` |
| `--field-B` | from preset | override domain B |
| `--field-combine` | from preset | how an `a+b` field merges: `max`, `sum`, `mean` |
| `--no-topo-invert` | off | keep the field as-is; by default it is negated so strong stain becomes the sublevel-set foreground |

The resolved fields are printed at startup as a `[fields]` line.

### What is compared

| flag | default | meaning |
|---|---|---|
| `--topo-dims` | `0 1` | homology dimensions: 0 connected components, 1 loops. Summed into one number |
| `--ph-cyc-split` | off | compare the IHC cycle term per stain channel (H and DAB separately) instead of on the merged field |
| `--topo-projection` | `auto` | `auto` = lifetime for H0, birth for H1; or force `birth` / `lifetime` / `death` everywhere |

### Schedule and cost

| flag | default | meaning |
|---|---|---|
| `--topo-start-step` | `0` | skip the PH terms until step N; no persistence is computed before it |
| `--topo-warmup-steps` | `0` | ramp the PH weight 0→1 over M steps after the start (0 = hard switch) |
| `--topo-every` | `1` | compute the PH terms only every n-th step |
| `--topo-max-images` | `0` | use only the first k images of each batch (0 = all) |
| `--topo-downsample` | `1` | average-pool the field by this factor before persistence |

Persistence is CPU-bound and unbatched, so these four are the levers that decide
how much the term costs. See **Cost** below for measured numbers.

### Extra logged values

Beyond the zoo's own losses, each log line carries `loss_ph_cyc_H`,
`loss_ph_cyc_I`, `loss_ph_trans_H`, `loss_ph_trans_I`, `loss_topo` (their
weighted sum) and `topo_scale` (the schedule multiplier).

### Sweep script environment

The sweep scripts read these via `--export=ALL,NAME=value`:
`DATA_DIR`, `DATA_A`, `DATA_B`, `BASE`, `CONDA_ENV`, `STEPS`, `BATCH_SIZE`,
`IMAGE_SIZE`, `PRESET`, `LAMBDA_TOPO`, `TOPO_DOWNSAMPLE`, `TOPO_EVERY`,
`SAVE_STEPS`, `PH_CYC_SPLIT`, `TOPO_START`, `TOPO_WARMUP`, `MARKER`, `REPO`. The three sweep weights come from the array index, not
the environment.

## Objective

```
L = L_adv + lambda_cycle * L_cyc + lambda_identity * L_idt
    + lambda_topo * ( lambda_ph_cyc  * (L_PH-cyc,H  + L_PH-cyc,I)
                    + lambda_ph_trans * (L_PH-trans,H + L_PH-trans,I) )
```

Persistent homology is computed on **stain-specific channels, never raw RGB**.
Domain A is H&E, domain B is IHC.

| `--field-*` spec | meaning |
|---|---|
| `gray` | luminance, no stain assumption |
| `hematoxylin/eosin` | deconvolve the pair, return the **first** stain's concentration with the second solved for and removed |
| `dab+hematoxylin` | deconvolve the pair and merge both channels with `--field-combine` |
| `hematoxylin` | bare projection onto one vector — **does not separate stains**, see below |

**Use `a/b`, not a bare stain name.** Stain vectors sit within ~37° of each
other, so projecting onto one returns most of the others:

```
field spec                 pure H pure DAB   pure E
hematoxylin                 1.000    0.800    0.864     <- projection: near-blind
hematoxylin/eosin           1.000   -0.077   -0.000     <- deconvolution
hematoxylin/dab             1.000   -0.000    0.297
```

A bare name measures something closer to total stain density than to one stain.
The bare form is kept only for completeness; every preset uses `a/b` or `a+b`.

`--preset` picks the pair of channels for a translation task; `--field-A`,
`--field-B` and `--field-combine` override any part of it.

| preset | domain A | domain B | rationale |
|---|---|---|---|
| `he-ihc` | `hematoxylin/eosin` | `dab+hematoxylin` (max) | DAB IHC (e.g. Ki67) carries a hematoxylin counterstain, so B merges both to get *all* nuclei, not only the positive ones |
| `he-sr` | `eosin/hematoxylin` | `dab/hematoxylin` | Sirius Red marks collagen, and eosin is the H&E channel that picks up the same collagen-rich stroma |

DAB alone sees only the *positive* nuclei, so the topology it measures changes
with proliferation index rather than with tissue structure; merging in the
hematoxylin counterstain gives all nuclei, which is the structure that should be
preserved by translation. The `a+b` path inverts the real stain matrix (the two
vectors plus their normalised cross-product residual, as QuPath does), so the
channels being merged are genuinely separated concentrations — a projection
would leave them correlated and make `max` close to meaningless.

Writing `x`=real H&E, `y_hat`=`G_H→I(x)`, `x_hat`=`G_I→H(y_hat)` for the forward
pass and `y`=real IHC, `x_hat'`=`G_I→H(y)`, `y_hat'`=`G_H→I(x_hat')` for the
reverse one:

| term | compares | fields |
|---|---|---|
| `L_PH-cyc,H` | `x` vs `x_hat` | H, H |
| `L_PH-cyc,I` | `y` vs `y_hat'` | H+DAB, H+DAB |
| `L_PH-trans,H` | `x` vs `y_hat` | H, **H+DAB** |
| `L_PH-trans,I` | `y` vs `x_hat'` | H+DAB, **H** |

All four are **paired**: term *i* compares image *i* with something generated
from image *i*, so no matching step is needed. The `trans` terms deliberately read
different stain fields on their two sides — H&E nuclei against IHC
nuclei-plus-chromogen — because the point is that nuclear topology visible in the
source must survive translation.

They are named `trans` (translation consistency) rather than `dist` because the
reference is each image's own source, not an unpaired real tile from the target
domain — this is not TopoGAN's distribution matching. TopoGAN's set-level OT loss remains available as
`losses.topological_loss` / `matched_diagram_loss` if you want to compare the two.

Each real domain's diagrams are computed once and shared between that domain's
cycle and distribution term.

## Estimating stain vectors

The vectors in `fields.STAIN_VECTORS` are literature defaults, fitted to nobody's
slides. For a model that has to work across staining combinations, estimate them
from the training tiles instead (Macenko et al., 2009 — the method behind
QuPath's *Estimate stain vectors → Auto*):

```bash
topo-estimate-stains --dataA tiles/ER/TrainValAB/trainA \
                     --dataB tiles/ER/TrainValAB/trainB \
                     --out runs/stains_ER.json
```

Estimate **once**, from TRAIN, and reuse the JSON everywhere downstream.
Re-estimating per run — or per batch — makes the loss non-stationary and puts a
discontinuity in the middle of a requeued job.

Once vectors come from data, stain *names* stop meaning anything, so field specs
become positional: `stain1/stain2`, `stain2/stain1`, `stain1+stain2`. Domain B's
pair is ordered against domain A's first stain, so channel 1 means the same thing
in both domains rather than depending on a per-domain heuristic; `--pin-shared`
goes further and forces them equal, trading fidelity to domain B's own colours
for exact comparability. The estimator warns if a domain's two stains come out
closer than 15 degrees, which is what a cohort with too little of one stain looks
like.

`slurm/estimate_and_validate.sh` does both steps per marker — estimate on train,
score field combinations on val — writing `stains_<marker>.json` (report these in
the paper) and `fields_<marker>.txt`.

## Choosing the fields before training

MIST ships registered H&E/IHC pairs, so the topological distance has a ground
truth to be tested against: a **true** pair must score below a **random** one.

**MIST pairs are serial sections**, which caps what this can show. Sections are
3-5 um apart and a nucleus is 5-10 um, so the two slides contain genuinely
different nuclei — exact nuclear correspondence is absent from the ground truth
itself. Only coarser architecture (glands, stroma, regional cellularity) carries
across. Measured AUROCs of ~0.6 on MIST may therefore be near the ceiling rather
than evidence the loss is broken, and the useful output is the *relative* ranking
of settings, which all face the same ceiling.

`--shuffle-within-slide` draws each wrong partner from the same source image.
Without it, a true pair also shares staining intensity, section thickness and
scanner with its partner, so the distance can score well by recognising the
specimen rather than the tissue — an unstratified AUROC is an upper bound on the
tissue-correspondence signal. If
it does not, that field choice carries no information about correspondence and
cannot teach `ph_trans` anything, however `lambda_topo` is set.

```bash
sbatch slurm/validate_fields.sh                  # all four markers
sbatch --export=ALL,MARKERS=ER slurm/validate_fields.sh                    # one marker
MARKERS=ER LIMIT=32 bash slurm/validate_fields.sh                          # locally

topo-validate-fields --dataA tiles/ER/TrainValAB/valA --dataB tiles/ER/TrainValAB/valB \
    --field-A hematoxylin/eosin \
    --field-B hematoxylin/dab dab/hematoxylin dab+hematoxylin
```

`slurm/validate_fields.sh` runs the full grid for each marker and writes
`field_validation/fields_<marker>.txt`. It requests **no GPU** — persistence is
CPU-only — and skips a marker whose `valA`/`valB` are missing.

`--dims-set` and `--topo-projection` take lists too, and are **nearly free** to
sweep: diagrams always carry both homology dimensions, and the projection only
changes how they are compared, so both reuse the cached diagrams. Only the field
specs and `--downsample` change the diagrams themselves and cost real time.

Keep `auto` in the projection list — it is what training uses, so without it the
winner cannot be compared against your current setting.

`--downsample` takes a list too, so resolution joins the cross product. That
matters for cost: persistence scales with pixel count, so if a coarser field
holds the same AUROC it is free signal — train at that `--topo-downsample`
instead. Raw distances are *not* comparable across factors (the sum runs over
more diagram points at finer resolution), but AUROC is, which is why it is the
column to steer by.

**How many tiles.** The 95% band at AUROC 0.60 is +/-0.069 at 128 tiles,
+/-0.035 at 512 and +/-0.012 at 4000. Note crops from one source image are not
independent samples — at four crops per image, 512 tiles is ~128 images — so the
effective sample is nearer the image count than the tile count.

Use ~128 for the screen: the 95% AUROC band is about
+/-0.013 there, enough to rank field choices, and the full 324-combination grid
takes ~3 minutes. 32 tiles gives +/-0.027, too noisy once 324 combinations get to
compete for the maximum. The whole 4000-tile val set buys +/-0.003 and costs
~1.6 h per marker, which cannot change which field you pick. Distances, not
diagrams, dominate that cost.

Tiles are sampled **at random** by default, because they are named
`<slide>_r<row>c<col>` — taking them in filename order (`--sample head`) would
draw every crop of the first few slides rather than a spread across the cohort.
The permutation is seeded, so `--offset` still carves an exactly disjoint second
sample **provided `--seed` is unchanged**; a different seed re-permutes and the
slices overlap.

Then re-score only the leading rows with `--offset 128` on a disjoint slice. At
128 tiles the top ten rows are statistically tied, so the screen's job is to
eliminate everything near 0.5 and surface a cluster of good cheap settings, not
to crown a winner.

It scores the cross product of everything given — a 3x3x3x3x3 sweep is 324
combinations and runs in seconds on 24 tiles — rather than committing GPU-weeks
to find out. With that many combinations the top row is partly luck, so the tool
says so and you should re-score the leaders on held-out tiles. The headline number is
AUROC — the probability a true pair scores below a shuffled one. 0.5 is useless,
1.0 is perfect separation. Tiles are matched by filename, with a loud warning if
the two directories share none.

Diagrams are computed once per image per spec and reused across combinations, so
adding specs is cheap relative to the first one.

## Validation inference

```bash
sbatch --export=ALL,MARKER=ER slurm/infer_sweep.sh                # all 38 cells
sbatch --export=ALL,MARKER=ER --array=0-18 slurm/infer_sweep.sh   # lambda_cycle=10 only
sbatch --export=ALL,MARKER=ER,LIMIT=8 --array=13 slurm/infer_sweep.sh
topo-infer --ckpt <ckpt> --data <tiles> --outdir <out>            # one checkpoint
```

`infer_sweep.sh` uses **the same `--array` as the training sweep**: task N infers
the model that training task N produced, because both resolve the cell through
`slurm/_grid.sh`, which holds the cell list and the `RUN_NAME` derivation once.
It prefers the highest numbered checkpoint and falls back to `step_latest.pt`; a
cell that has not trained yet prints a note and exits 0, so it does not show up
as a failed array task. Predictions go to `BASE/preds/<run>/`.

**The zoo's `i2i-inference` cannot load these checkpoints** — it rebuilds the
config with `CycleGANConfig(**saved_cfg)`, which rejects the extra `topo` field,
and then loads weights strictly, which rejects the `_topo_step` buffer. Both
failures are pinned by tests. `topo-infer` does the same job with the right
config class and otherwise follows the zoo's conventions: same transform, same
`[-1,1] -> [0,1]` tile writer, batch size 1, `.tif` output.

## Sweeping the PH weights

One sweep script per MIST marker — `slurm/sweep_ki67.sh`, `sweep_er.sh`,
`sweep_her2.sh`, `sweep_pr.sh` — each a SLURM array job over an explicit
38-cell grid:

| axis | values |
|---|---|
| `lambda_cycle` | 10, 0 |
| `lambda_ph_cyc` | 0, 1 |
| `lambda_ph_trans` | 0, 1 |
| `lambda_topo` | 2e-4, 2e-2 |
| `field_B` | `hematoxylin/dab`, `dab/hematoxylin`, `dab+hematoxylin` (sum) |

`field_A` is fixed at `hematoxylin/eosin`. The full factorial is 48, but
`ph_cyc=0` and `ph_trans=0` together switch off every topological term, making
`lambda_topo` and both fields inert — those 12 collapse to one anchor run per
`lambda_cycle`, leaving 38. The cell list is written out in
`slurm/_sweep_common.sh` rather than computed, so the duplicates are simply
absent and a task index past the end is refused.

Tasks 0–18 are `lambda_cycle=10` (the critical path); 19–37 are `lambda_cycle=0`,
testing whether topological terms can substitute for L1 cycle-consistency. Expect
that arm to struggle: the diagram distance compares multisets of scalar values
with no spatial anchoring, so it constrains how many features of what persistence
exist, never where — only `lambda_identity` would hold content in place.

```bash
sbatch slurm/sweep_ki67.sh                                 # all 27
sbatch --array=0-8,10-17,19-26 slurm/sweep_er.sh           # skip duplicate baselines
sbatch --array=13 slurm/sweep_her2.sh                      # one cell
```

The four differ only in their `#SBATCH` header and a `MARKER` assignment; the
grid, schedule and flags live once in `slurm/_sweep_common.sh`, which each
wrapper sources. `_sweep_common.sh` is not submittable on its own — it refuses
without `MARKER`. **Submit from the repository root** so `SLURM_SUBMIT_DIR`
locates it, or export `REPO=/path/to/TopoCycleGAN`.

`DATA_DIR` defaults to `/work2/bz66izin-TopoCG/MIST_tiles/${MARKER}/TrainValAB`
and `BASE` to `/work2/bz66izin-TopoCG/Outputs_<marker>` (lowercased), so each
marker writes to its own output tree:

```
Outputs_ki67/results/Ki67_lc10_lt0.0002_cyc1_trans1_hematoxylin-dab
Outputs_er/results/ER_lc10_lt0.0002_cyc1_trans1_hematoxylin-dab
Outputs_her2/results/HER2_...
Outputs_pr/results/PR_...
```

Run directories keep the marker prefix too, so a directory stays
self-describing if it is copied out of its tree.

Tasks 0, 9 and 18 all have both weights at zero, so they are the same baseline
three times — run one.

Defaults are 400,000 steps — 8 epochs of ~50k tiles, and ~41 h, so a cell
finishes inside one 48 h allocation without requeueing — with permanent
checkpoints every 100k steps (100k/200k/300k/400k). Each checkpoint is ~340 MB
(model plus both Adam states), so budget ~1.7 GB per run and ~170 GB for a full
4-marker sweep.

Create the log directory once before the first submit — SLURM will not make it,
and jobs fail with nowhere to report why:

```bash
mkdir -p logs_topo
```

The scripts target the `clara` partition, loads `Anaconda3/2025.06-1` and
activates the `topocg` conda environment (override with
`--export=ALL,CONDA_ENV=othername`). Each task writes to
`${BASE}/results/<preset>_cyc<w>_trans<w>/` and resumes from its own checkpoint
if requeued — the PH schedule resumes with it, since the step counter is a
checkpointed buffer.

Note the jobs are deliberately CPU-heavy: persistence is single-threaded per
image and runs while the GPU idles, so expect low GPU utilisation.

## The loss

| module | contents |
|---|---|
| `fields.py` | RGB → scalar field: luminance, single-vector projection, or full deconvolution of a stain pair |
| `persistence.py` | cubical persistence, differentiable w.r.t. the image |
| `losses.py` | Eq. 3 diagram distance, paired loss, Eq. 4/5 set-level OT loss |
| `models.py` | `TopoCycleGAN`, `TopoLossMixin` |

Differentiability comes for free: a sublevel-set filtration never invents values,
so every birth/death is the value at one *critical pixel*. gudhi supplies those
pixel indices (on detached numpy), and we gather them out of the live tensor —
autograd then routes each diagram point's gradient back to its pixel.

Two deviations from the paper, both deliberate:

- **No distance transform.** TopoGAN's generator emits binary masks and filters
  their DT. Ours emits stain RGB, so we filter a deconvolved stain channel
  instead — permitted by the paper's footnote 1, but birth times are then
  intensity values, not "gap to close" distances.
- **Hungarian instead of the OT LP** for Eq. 5. With uniform marginals and equal
  batch sizes the LP optimum is a permutation, so the assignment problem is
  equivalent and drops the POT dependency.

## Which projection

The diagram distance compares a 1-D projection of each diagram (TopoGAN Eq. 3
projects onto the birth axis, which is what reduces the matching to a sort).
`--topo-projection` chooses it:

| value | H0 | H1 |
|---|---|---|
| `auto` (default) | lifetime | birth |
| `birth` / `lifetime` / `death` | forced for every dimension | |

`auto` splits them deliberately. TopoGAN justifies dropping death times for
**H1**: on a distance transform a loop's birth is the gap that must be closed to
complete an almost-hole. That argument does not carry to **H0** on a stain field,
where a component's birth is just the value of a local minimum — birth-only would
compare peak stain intensities. Its lifetime instead measures how deep a blob is
before merging into its neighbour, which is the connectivity structure H0 is
meant to capture, and is the usual TDA robustness measure (noise gives
short-lived features).

Using deaths costs nothing measurable: gudhi already returns both endpoints of
every pair, so this is one extra subtraction reusing the same sort (0.26 ms vs
0.20 ms per call, against a ~30 ms persistence computation). Gradients flow the
same way and reach *more* pixels, since each pair now feeds both its birth and
its death pixel.

Note the zero-padding that matches unmatched points to the diagonal is exact for
`birth` and `lifetime` (a diagonal point has lifetime 0) but not for `death`,
which is offered for experiments rather than as a metric.

Full 2-D Wasserstein between diagrams is deliberately not offered: it makes the
matching a genuine assignment problem, measured at ~500 ms per pair against
0.2 ms, which is why TopoGAN projects to one axis in the first place.

## When the PH terms switch on

```bash
topo-train ... --topo-start-step 10000 --topo-warmup-steps 5000
```

Early in training the generator emits noise, and noise maximises the number of
critical points (1845 H0 + 3140 H1 on a 128x128 random field) — so the term is at
its largest, least meaningful and most expensive exactly when it can do least
good. `--topo-start-step N` skips it entirely until step N: no persistence is
computed at all, so those steps also run at baseline CycleGAN speed.
`--topo-warmup-steps M` then ramps the weight linearly from 0 to 1 over M steps
instead of switching it on in one jump, which spares Adam's moment estimates a
discontinuity in the loss.

The multiplier is logged each step as `topo_scale`, and the step counter is a
registered buffer, so it is checkpointed — a requeued SLURM job resumes its
schedule rather than restarting the warmup.

## Cost

Persistence is CPU-bound and unbatched, and there are now six fields to diagram
per step. Measured on a full generator step, batch 4 at 256x256, uniform noise
(worst case — real stain images have far fewer critical pairs):

| `--topo-downsample` | field size | per step |
|---|---|---|
| 1 | 256×256 | 2.37 s |
| 2 | 128×128 | 1.48 s |
| 4 | 64×64 | 1.21 s |

These are whole-step timings, so the generator forward/backward is included —
which is why the curve flattens rather than falling 4×. `--topo-every` and
`--topo-max-images` cut it further.

## Caveats

**`--lambda-topo` has no principled default.** The terms are in filtration units,
so their scale depends on `--field-A`/`--field-B`, and in the smoke test the four
PH terms summed to ~150 against a GAN term of ~3. Sweep before any real run.

**The cycle-topology terms overlap with the L1 cycle loss.** `lambda_cycle = 10`
already forces `rec_A ≈ real_A` pixel-wise, which implies matching topology. The
PH cycle terms may therefore add cost without adding much signal — worth an
ablation (`--lambda-ph-cyc 0`) before committing to them. The distribution terms
have no such overlap: nothing else in CycleGAN constrains unpaired topology.

## Standalone use

The loss has no dependency on the zoo:

```python
from topo_i2i import topological_loss, rgb_to_scalar_field
```

The vectors in `fields.STAIN_VECTORS` are literature values. Estimate the real
ones from your own slides — QuPath's *Analyze > Estimate stain vectors > Auto* —
and replace them before training; the deconvolution is only as good as those
numbers.

(A standalone Python reimplementation of that QuPath command, `qupath_stains.py`,
was removed in the commit following `b7aca07`; recover it from there if useful.)
