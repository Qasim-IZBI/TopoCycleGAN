"""Run a trained TopoCycleGAN over a directory of tiles.

The zoo's `i2i-inference` cannot load these checkpoints: it rebuilds the config
with `CycleGANConfig(**saved_cfg)`, which rejects the extra `topo` field, and
then calls `load_state_dict(..., strict=True)`, which rejects the `_topo_step`
buffer. Both are checked in tests. This does the same job with the right config
class, and otherwise follows the zoo's conventions -- same transform, same
[-1,1] -> [0,1] tile writer, batch size 1, `.tif` output.

    topo-infer --ckpt runs/.../checkpoints/step_400000.pt \
               --data tiles/ER/TrainValAB/valA --outdir preds/valA --direction A2B

Tiles are found recursively and written under the same relative path, so a
nested <case>/images/<id>.tif layout comes back out as <case>/images/<id>.tif
and ids repeated across cases do not collide. That recursion also picks up a
sibling <case>/masks/ -- use --subdir images to keep the masks out.
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from i2i_stain_zoo.datasets.single_domain_dataset import SingleDomainDataset
from i2i_stain_zoo.datasets.transforms import default_train_transform
from i2i_stain_zoo.utils import get_device

from topo_i2i.models import TopoCycleGAN, TopoCycleGANConfig, TopoConfig


def save_tile(y: torch.Tensor, path: str) -> None:
    """Save a [-1,1] NCHW tile as an image in [0,1] -- the zoo's convention."""
    t = ((y.squeeze(0).detach().cpu().float().clamp(-1, 1) + 1.0) / 2.0).unsqueeze(0)
    save_image(t, path)


def load_model(ckpt_path: str, device) -> TopoCycleGAN:
    """Rebuild TopoCycleGAN from a checkpoint written by BaseTrainer."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("config")

    if saved:
        saved = dict(saved)
        topo = saved.pop("topo", None)          # nested dataclass -> rebuild it
        cfg = TopoCycleGANConfig(**saved)
        if topo:
            cfg.topo = TopoConfig(**topo)
    else:
        cfg = TopoCycleGANConfig()

    model = TopoCycleGAN(cfg)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.to(device).eval()
    step = ckpt.get("global_step", "?")
    print("[ckpt] %s (step %s)  lambda_topo=%s ph_cyc=%s ph_trans=%s A=%s B=%s"
          % (os.path.basename(ckpt_path), step, cfg.topo.lambda_topo,
             cfg.topo.lambda_ph_cyc, cfg.topo.lambda_ph_trans,
             cfg.topo.field_A, cfg.topo.field_B))
    return model


def filter_subdir(paths, subdir: str):
    """Keep only tiles whose immediate parent directory is named `subdir`.

    WSI tiling writes <case>/images/<id>.tif next to <case>/masks/<id>.tif, and
    the zoo's loader walks the whole tree -- without this it would translate the
    masks too, doubling the work and salting the output with predictions made
    from binary images.
    """
    return [p for p in paths if os.path.basename(os.path.dirname(p)) == subdir]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint to load")
    p.add_argument("--data", required=True, help="directory of input tiles")
    p.add_argument("--outdir", required=True)
    p.add_argument("--direction", choices=("A2B", "B2A"), default="A2B",
                   help="A2B translates H&E -> IHC (the default)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0,
                   help="stop after this many tiles (0 = all); useful for a quick look")
    p.add_argument("--ext", default=".tif", help="output extension")
    p.add_argument("--subdir", default="",
                   help="only use tiles whose parent directory has this name, "
                        "e.g. 'images' for a <case>/images/<id>.tif layout; "
                        "default: every image under --data")
    p.add_argument("--resume", action="store_true",
                   help="skip tiles that already exist in --outdir")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = get_device()
    model = load_model(args.ckpt, device)

    dataset = SingleDomainDataset(args.data,
                                  transform=default_train_transform(args.image_size))
    if args.subdir:
        found = len(dataset.paths)
        dataset.paths = filter_subdir(dataset.paths, args.subdir)
        if not dataset.paths:
            raise SystemExit("no tiles under %s have a parent directory named %r "
                             "(%d images found, all filtered out)"
                             % (args.data, args.subdir, found))
        print("[data] %d of %d images are in a %r directory"
              % (len(dataset.paths), found, args.subdir))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    os.makedirs(args.outdir, exist_ok=True)

    forward = model.forward_A2B if args.direction == "A2B" else model.forward_B2A
    written = skipped = 0
    with torch.no_grad():
        for i, (x, path) in enumerate(loader):
            if args.limit and written >= args.limit:
                break
            stem = os.path.splitext(os.path.relpath(path[0], args.data))[0]
            out = os.path.join(args.outdir, stem + args.ext)
            if args.resume and os.path.exists(out):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            save_tile(forward(x.to(device)), out)
            written += 1

    print("[%s] wrote %d tiles to %s%s"
          % (args.direction, written, os.path.abspath(args.outdir),
             ", skipped %d existing" % skipped if skipped else ""))


if __name__ == "__main__":
    main()
