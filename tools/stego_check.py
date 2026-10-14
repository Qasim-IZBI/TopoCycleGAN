"""Does a generator hide the H&E in its IHC? -- the CycleGAN steganography check.

A collapsed seed's IHC shows no nuclei, yet its cycle loss is as low as a
trained seed's: the reverse generator rebuilds nuclei it cannot see. That is
possible if the forward generator encodes the H&E in a low-amplitude signal the
eye -- and the discriminator -- ignore (Chu et al. 2017, "CycleGAN, a master of
steganography"). Such a code is fragile, so this perturbs the generated IHC
invisibly before the reverse generator and measures what is left of the
reconstruction:

    rec = G_B2A( perturb( G_A2B(he) ) )      L1(rec, he), in [0,1] units

for Gaussian noise of each --sigmas (std in [0,1] image units; 0.01 is ~2.5
grey levels) and, with --quantize, rounding to 8 bits as a saved tile is.

READING IT. Compare a collapsed seed with a trained one of the same model:
  hidden code     the collapsed seed's error climbs steeply already at
                  sigma 0.005-0.02, far more than the trained seed's
  no hidden code  both degrade alike and gently -- the reconstruction rests on
                  what is visible, and the collapse has another cause
`delta` is the rise over the run's own unperturbed error and `pert_l1` the
size of the perturbation itself: a generator whose reconstruction rests on
visible content has delta at most about pert_l1 (it passes the noise through
or removes it); a hidden code gives delta many times pert_l1. `ratio` is the
error over the unperturbed one.

    python tools/stego_check.py \\
        --run collapsed=topo:<ens>/Ki67/topo/results/seed4/checkpoints/step_400000.pt \\
        --run trained=topo:<ens>/Ki67/topo/results/seed6/checkpoints/step_400000.pt \\
        --data <tiles>/Ki67/TrainValAB/valA --limit 200 --quantize \\
        --out stego_ki67.csv --examples 4 --examples-dir stego_ki67/

MODEL is topo (topo-train checkpoints) or cyclegan / dclgan (the zoo's). Run it
on a GPU node; it falls back to CPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_any(model: str, ckpt_path: str, device):
    """A generator pair with forward_A2B / forward_B2A, in eval mode."""
    if model == "topo":
        from topo_i2i.inference import load_model
        return load_model(ckpt_path, device)

    from i2i_stain_zoo.models import CycleGAN, CycleGANConfig, DCLGAN, DCLGANConfig
    classes = {"cyclegan": (CycleGAN, CycleGANConfig), "dclgan": (DCLGAN, DCLGANConfig)}
    if model not in classes:
        raise SystemExit("unknown model %r: topo, cyclegan or dclgan" % model)
    cls, cfg_cls = classes[model]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("config")
    net = cls(cfg_cls(**saved) if saved else cfg_cls())
    net.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    print("[ckpt] %s %s (step %s)" % (model, os.path.basename(ckpt_path),
                                      ckpt.get("global_step", "?")))
    return net.to(device).eval()


def conditions(sigmas: List[float], quantize: bool) -> List[Tuple[str, float]]:
    out = [("clean", 0.0)] + [("noise", s) for s in sigmas if s > 0]
    if quantize:
        out.append(("q8", 0.0))
    return out


def perturb(fake: torch.Tensor, kind: str, sigma: float, gen: torch.Generator) -> torch.Tensor:
    """`fake` in [-1,1]; sigma in [0,1] image units, so 2*sigma here."""
    if kind == "clean":
        return fake
    if kind == "q8":
        return torch.round((fake + 1) * 127.5) / 127.5 - 1
    noise = torch.randn(fake.shape, generator=gen).to(fake.device)
    return (fake + 2 * sigma * noise).clamp(-1, 1)


def l1_01(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image mean absolute difference, in [0,1] units."""
    return (a - b).abs().mean(dim=(1, 2, 3)) / 2


def pick(paths: List[str], limit: int) -> List[str]:
    """`limit` tiles spread evenly over the sorted list -- every slide, not the first."""
    paths = sorted(paths)
    if not limit or len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, metavar="LABEL=MODEL:CKPT")
    p.add_argument("--data", required=True, help="directory of H&E tiles (domain A)")
    p.add_argument("--subdir", default="", help="as topo-infer --subdir")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1])
    p.add_argument("--quantize", action="store_true", help="also round the IHC to 8 bits")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV of the table")
    p.add_argument("--examples", type=int, default=0, help="tiles to save as image panels")
    p.add_argument("--examples-dir", default="stego_examples")
    p.add_argument("--example-sigma", type=float, default=0.02)
    args = p.parse_args()

    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from i2i_stain_zoo.datasets.single_domain_dataset import SingleDomainDataset
    from i2i_stain_zoo.datasets.transforms import default_train_transform
    from i2i_stain_zoo.utils import get_device
    from topo_i2i.inference import filter_subdir

    runs = []
    for spec in args.run:
        try:
            label, rest = spec.split("=", 1)
            model, ckpt = rest.split(":", 1)
        except ValueError:
            raise SystemExit("--run takes LABEL=MODEL:CKPT, got %r" % spec)
        runs.append((label, model, ckpt))

    device = get_device()
    dataset = SingleDomainDataset(args.data, transform=default_train_transform(args.image_size))
    if args.subdir:
        dataset.paths = filter_subdir(dataset.paths, args.subdir)
    dataset.paths = pick(dataset.paths, args.limit)
    if not dataset.paths:
        raise SystemExit("no tiles under %s" % args.data)
    print("[data] %d tiles from %s" % (len(dataset.paths), args.data))

    conds = conditions(args.sigmas, args.quantize)
    if args.examples:
        os.makedirs(args.examples_dir, exist_ok=True)

    rows = []
    for label, model, ckpt in runs:
        net = load_any(model, ckpt, device)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        err: Dict[str, List[torch.Tensor]] = {cond_key(k, s): [] for k, s in conds}
        pert: Dict[str, List[torch.Tensor]] = {cond_key(k, s): [] for k, s in conds}
        shown = 0
        with torch.no_grad():
            for b, (x, _) in enumerate(loader):
                x = x.to(device)
                fake = net.forward_A2B(x)
                recs = {}
                for i, (kind, sigma) in enumerate(conds):
                    key = cond_key(kind, sigma)
                    # The same noise for every run, so runs differ only in the model.
                    gen = torch.Generator().manual_seed(args.seed * 100003 + b * 101 + i)
                    pf = perturb(fake, kind, sigma, gen)
                    rec = net.forward_B2A(pf)
                    err[key].append(l1_01(rec, x).cpu())
                    pert[key].append(l1_01(pf, fake).cpu())
                    recs[(kind, sigma)] = rec
                if shown < args.examples:
                    ex = recs.get(("noise", args.example_sigma))
                    for j in range(min(args.examples - shown, x.shape[0])):
                        panel = [x[j], fake[j], recs[("clean", 0.0)][j]]
                        if ex is not None:
                            panel += [ex[j], ((ex[j] - x[j]).abs() - 1).clamp(-1, 1)]
                        img = (torch.stack(panel) + 1) / 2
                        save_image(img, os.path.join(args.examples_dir,
                                                     "%s_%02d.png" % (label, shown)), nrow=len(panel))
                        shown += 1

        base = torch.cat(err["clean"]).mean().item()
        print("\n== %s  (%s, %s)" % (label, model, ckpt))
        print("  %-12s %10s %10s %8s %10s" % ("condition", "recon_L1", "delta", "ratio", "pert_L1"))
        for kind, sigma in conds:
            key = cond_key(kind, sigma)
            e = torch.cat(err[key]).mean().item()
            d = torch.cat(pert[key]).mean().item()
            print("  %-12s %10.4f %+10.4f %8.2f %10.4f"
                  % (key, e, e - base, e / base if base else float("nan"), d))
            rows.append({"run": label, "model": model, "condition": kind, "sigma": sigma,
                         "tiles": len(dataset.paths), "recon_l1": e, "delta": e - base,
                         "ratio": e / base if base else "", "pert_l1": d, "ckpt": ckpt})
        del net

    if args.examples:
        print("\npanels in %s: H&E | IHC | rec clean | rec noise %.3g | |rec - H&E|"
              % (args.examples_dir, args.example_sigma))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("written %s" % args.out)


def cond_key(kind: str, sigma: float) -> str:
    return "noise %.3g" % sigma if kind == "noise" else kind


if __name__ == "__main__":
    main()
