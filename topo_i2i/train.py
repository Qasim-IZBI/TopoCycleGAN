"""Training entry point: the zoo's data + trainer, our model subclass.

Deliberately thin. Everything except the model class and the topo flags is the
zoo's own machinery, imported rather than copied, so this stays in step with the
pinned commit.

    topo-train --dataA tiles/HE --dataB tiles/IHC --output runs/topo01 \
               --lambda-topo 1.0 --field-A hematoxylin --field-B dab+hematoxylin \
               --topo-downsample 2
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from i2i_stain_zoo.datasets.transforms import default_train_transform
from i2i_stain_zoo.datasets.unpaired_dataset import UnpairedDataset
from i2i_stain_zoo.trainer.base_trainer import BaseTrainer

from topo_i2i.fields import FIELD_PRESETS, STAIN_VECTORS, resolve_fields
from topo_i2i.models import TopoConfig, TopoCycleGAN, TopoCycleGANConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataA", required=True, help="source domain tiles (H&E)")
    p.add_argument("--dataB", required=True, help="target domain tiles (Sirius Red)")
    p.add_argument("--output", default="runs/topo")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-steps", type=int, default=25_000)
    p.add_argument("--log-steps", type=int, default=1_000)

    g = p.add_argument_group("topological loss")
    g.add_argument("--lambda-topo", type=float, default=1.0,
                   help="overall scale; 0 disables PH entirely (CycleGAN baseline)")
    g.add_argument("--lambda-ph-cyc", type=float, default=1.0,
                   help="weight on the cycle-topology terms (L_PH-cyc,H + L_PH-cyc,I)")
    g.add_argument("--lambda-ph-trans", type=float, default=1.0,
                   help="weight on the translation terms (L_PH-trans,H + L_PH-trans,I)")
    g.add_argument("--preset", default="he-ihc", choices=sorted(FIELD_PRESETS),
                   help="which stain channels each domain is deconvolved onto: "
                        + "; ".join("%s -> %s / %s" % (k, v["field_A"], v["field_B"])
                                    for k, v in sorted(FIELD_PRESETS.items())))
    g.add_argument("--field-A", default=None, metavar="SPEC",
                   help="scalar field for the H&E domain: 'gray', one stain "
                        "name, or 'a+b' to deconvolve a pair and merge it "
                        "(known stains: %s)" % ", ".join(sorted(STAIN_VECTORS)))
    g.add_argument("--field-B", default=None, metavar="SPEC",
                   help="override the preset's domain-B field")
    g.add_argument("--field-combine", default=None, choices=("max", "sum", "mean"),
                   help="how an 'a+b' field merges its two channels")
    g.add_argument("--topo-dims", type=int, nargs="+", default=[0, 1],
                   help="homology dimensions: 0 components, 1 loops")
    g.add_argument("--topo-projection", default="auto",
                   choices=("auto", "birth", "lifetime", "death"),
                   help="which 1-D projection of the diagrams to compare; "
                        "'auto' uses lifetime for H0 and birth for H1")
    g.add_argument("--topo-every", type=int, default=1,
                   help="compute the PH terms every n-th step (cost control)")
    g.add_argument("--topo-max-images", type=int, default=0,
                   help="use only the first k images of the batch (0 = all)")
    g.add_argument("--topo-downsample", type=int, default=1,
                   help="average-pool the field before persistence")
    g.add_argument("--no-topo-invert", action="store_true",
                   help="keep the field as-is; by default it is negated so that "
                        "strong stain becomes the sublevel-set foreground")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnpairedDataset(
        root_A=args.dataA, root_B=args.dataB,
        transform=default_train_transform(image_size=args.image_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    fields = resolve_fields(args.preset, args.field_A, args.field_B, args.field_combine)
    print("[fields] preset=%s  A=%s  B=%s  combine=%s"
          % (args.preset, fields["field_A"], fields["field_B"], fields["combine"]))

    cfg = TopoCycleGANConfig(topo=TopoConfig(
        lambda_topo=args.lambda_topo,
        lambda_ph_cyc=args.lambda_ph_cyc,
        lambda_ph_trans=args.lambda_ph_trans,
        field_A=fields["field_A"],
        field_B=fields["field_B"],
        combine=fields["combine"],
        dims=tuple(args.topo_dims),
        projection=args.topo_projection,
        every_n_steps=args.topo_every,
        max_images=args.topo_max_images,
        downsample=args.topo_downsample,
        invert=not args.no_topo_invert,
    ))
    model = TopoCycleGAN(cfg).to(device)

    trainer = BaseTrainer(
        model=model, dataloader=loader, device=device,
        model_name="topo-cyclegan", lr=args.lr, use_amp=args.amp,
        save_dir=args.output + "/checkpoints",
        sample_dir=args.output + "/samples",
        save_steps=args.save_steps, log_steps=args.log_steps,
    )
    trainer.resume_if_exists()
    trainer.train(args.steps)


if __name__ == "__main__":
    main()
