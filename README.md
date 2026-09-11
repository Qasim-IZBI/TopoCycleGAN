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
`TOPO_START`, `TOPO_WARMUP`, `MARKER`, `REPO`. The three sweep weights come from the array index, not
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
| `hematoxylin` | projection onto one stain vector (approximate — stains stay correlated) |
| `dab+hematoxylin` | full 3×3 deconvolution of the pair, channels merged by `--field-combine` |

`--preset` picks the pair of channels for a translation task; `--field-A`,
`--field-B` and `--field-combine` override any part of it.

| preset | domain A | domain B | rationale |
|---|---|---|---|
| `he-ihc` | `hematoxylin` | `dab+hematoxylin` (max) | DAB IHC (e.g. Ki67) carries a hematoxylin counterstain, so B merges both to get *all* nuclei, not only the positive ones |
| `he-sr` | `eosin` | `dab` | Sirius Red marks collagen, and eosin is the H&E channel that picks up the same collagen-rich stroma |

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

## Sweeping the PH weights

One sweep script per MIST marker — `slurm/sweep_ki67.sh`, `sweep_er.sh`,
`sweep_her2.sh`, `sweep_pr.sh` — each a SLURM array job over a 3x3x3 grid:

| axis | values | what it answers |
|---|---|---|
| `lambda_topo` | 2e-4, 2e-3, 2e-2 | overall PH scale |
| `lambda_ph_cyc` | 0, 0.5, 1 | is the cycle-topology family redundant with the L1 cycle loss? |
| `lambda_ph_trans` | 0, 0.5, 1 | is the translation-topology family pulling its weight? |

The zeros make the grid contain its own ablations: `ph_cyc=0` isolates the
translation terms, `ph_trans=0` isolates the cycle terms, and both zero is the
plain CycleGAN baseline.

On the scale axis: 2e-4 puts the PH gradient roughly on a par with the cycle
gradient, and 2e-2 leaves it ~80x larger. The loss sums over ~17k diagram points
while the other terms average over pixels, so its raw value (~600) is a reduction
artifact, not a measure of importance — matching gradients matters, matching
printed values does not.

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
and run directories are prefixed with the marker, so the four sweeps never
collide.

Tasks 0, 9 and 18 all have both weights at zero, so they are the same baseline
three times — run one.

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
