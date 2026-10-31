# Paper figures and tables

Everything that makes a figure or table of the CVPR 2027 paper, one Python
script and one SLURM script each, named after the paper's numbering (as of
31 Oct 2026; rename if the paper renumbers). Each `.py` runs by path from a
checkout with the `topocg` environment (`python paper/<script>.py --help`); each
`slurm/*.sh` documents its variables in its header.

| Paper | What | Python | SLURM | Output on the cluster |
|---|---|---|---|---|
| Fig. 1 (teaser) | one tile through every seed of an ensemble, without and with the layout term; layout *r* per output, inverted ones framed red | `fig1_seeds.py` | `slurm/fig1_seeds.sh` | `$R/figure_seeds/<M>_<model>_<tile>/` |
| Fig. 2 (method) | both translations, the four stain fields, PH diagrams and their projections, for a chosen H&E/IHC pair | `fig2_loss.py` | `slurm/fig2_loss.sh` | `$R/figure_loss/<M>/<tile>/` |
| Fig. 4 (results) | every seed of every model per dataset: SSIM, LPIPS, FID, nuclei recall, positive-nuclei % / CPA MAE, BM₀ | `fig4_results.py` | `slurm/fig4_results.sh` (also runs `tools/ensemble_report.py`) | `$R/figure_results/results.pdf` |
| Fig. 5 (qualitative) | one tile per dataset across all models, real target beside | `fig5_qualitative.py` | `slurm/fig5_qualitative.sh` | `$R/figure_qual/` |
| Tab. 1 + Supp. Tab. S1 | chosen and worst stain fields per dataset, from the runs' `training_meta.json` | `tab1_fields.py` | `slurm/tab1_fields.sh` | `$R/paper_tables/fields.csv` |

The numbers in the paper's other tables (inversion, detailed results) come
from `tools/ensemble_report.py`, `tools/nuclei_eval.py` / `tools/cpa_eval.py`
via the paper's own `fill_results.py`. Fig. 3 (layout *r* vs recall per seed)
has no script yet.

```bash
cd paper/slurm && mkdir -p logs_topo
sbatch --export=ALL,MARKER=Ki67,MODEL=cyclegan,TILE=<name>.png fig1_seeds.sh
sbatch --export=ALL,MARKER=Ki67,TILE=<name>.png fig2_loss.sh
sbatch --export=ALL,EVAL_ROOT=$R/eval_layout1_stop20k fig4_results.sh
sbatch --export=ALL,TILE_Ki67=<t>.png,TILE_ER=<t>.png,...,TILE_VS=<t>.png fig5_qualitative.sh
bash tab1_fields.sh        # AUDIT=..., WORST_AUDIT=... for the AUROC columns
```

Tests: `tests/test_figure.py`.
