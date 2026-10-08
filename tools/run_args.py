"""Rebuild the topo-train arguments of a finished run from its training_meta.json.

An ensemble member has to differ from the run it repeats in the seed and in
nothing else. Re-typing the field, the stain vectors and a dozen schedule
flags by hand invites a silent mismatch, so this reads what the run itself
recorded -- BaseTrainer writes the model config (topo settings included) and
the training block into training_meta.json when a run starts -- and prints the
matching topo-train flags, one per line, for a bash array:

    mapfile -t ARGS < <(python3 tools/run_args.py <run>/training_meta.json \\
                            --stains-out <dir>/stains.json)
    topo-train --dataA ... --dataB ... --output ... --seed 3 "${ARGS[@]}"

The stain vectors a run trained on live in its config; when it has any they
are written to --stains-out in topo-estimate-stains' format and passed as
--stains, so the new run deconvolves with exactly the same vectors. Data paths,
output, workers and seed are the caller's -- they are not settings of the run.

--compare A B instead checks two runs' recorded settings and prints every
difference outside the timing block, so a finished member can be confirmed to
be a true repeat. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys


def flags(meta: dict, stains_out: str) -> list:
    cfg = meta.get("config") or {}
    topo = cfg.get("topo") or {}
    tr = meta.get("training") or {}
    if not topo:
        sys.exit("no topo settings in this training_meta.json -- not a topo-train run?")

    out = []

    def add(flag, value):
        out.extend([flag, str(value)])

    add("--steps", tr["total_steps"])
    add("--batch-size", tr["batch_size"])
    add("--lr", tr["learning_rate"])
    add("--save-steps", tr["save_steps"])
    add("--log-steps", tr["log_steps"])
    if tr.get("amp"):
        out.append("--amp")
    add("--lambda-cycle", cfg["lambda_cycle"])
    add("--lambda-identity", cfg["lambda_identity"])

    add("--lambda-topo", topo["lambda_topo"])
    add("--lambda-ph-cyc", topo["lambda_ph_cyc"])
    add("--lambda-ph-trans", topo["lambda_ph_trans"])
    add("--field-A", topo["field_A"])
    add("--field-B", topo["field_B"])
    add("--field-combine", topo["combine"])
    out.append("--topo-dims")
    out.extend(str(d) for d in topo["dims"])
    add("--topo-projection", topo["projection"])
    add("--topo-start-step", topo["start_step"])
    add("--topo-warmup-steps", topo["warmup_steps"])
    add("--topo-every", topo["every_n_steps"])
    add("--topo-max-images", topo["max_images"])
    add("--topo-downsample", topo["downsample"])
    if topo.get("ph_cyc_split"):
        out.append("--ph-cyc-split")
    if not topo.get("invert", True):
        out.append("--no-topo-invert")

    va, vb = topo.get("vectors_A") or {}, topo.get("vectors_B") or {}
    if va or vb:
        if not stains_out:
            sys.exit("this run trained on estimated stain vectors -- give --stains-out")
        with open(stains_out, "w") as f:
            json.dump({"A": va, "B": vb}, f, indent=2)
        add("--stains", stains_out)
    return out


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("meta", nargs="?", help="the run's training_meta.json")
    p.add_argument("--stains-out", default=None)
    p.add_argument("--compare", nargs=2, metavar=("META_A", "META_B"),
                   help="print the recorded settings that differ between two runs")
    args = p.parse_args()

    if args.compare:
        a, b = (flatten(json.load(open(m))) for m in args.compare)
        diff = [(k, a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
                if not k.startswith("timing.") and a.get(k) != b.get(k)]
        for k, x, y in diff:
            print("%s: %s | %s" % (k, x, y))
        sys.exit(1 if diff else 0)

    if not args.meta:
        p.error("give a training_meta.json, or --compare A B")
    print("\n".join(flags(json.load(open(args.meta)), args.stains_out)))


if __name__ == "__main__":
    main()
