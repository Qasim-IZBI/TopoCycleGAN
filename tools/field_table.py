"""The stain-field tables of the paper, from what the runs were trained with.

Main paper: the field the selection protocol chose, per dataset. Supplementary:
the field it ranked last (the worst-field control). Both are read from the
runs' own training_meta.json -- the settings each model actually trained on,
not what a file says it should have -- and, when given, the audit's AUROCs:

    chosen   confirmation AUROC (slice 2) from <audit>/decision.json
    worst    screening AUROC (slice 1) from the comment topo-worst-field wrote
             into <worst-audit>/recommended_<marker>.env

    python tools/field_table.py --root /work2/bz66izin-TopoCG \\
        --audit $R/field_audit_v2 --worst-audit $R/field_worst_v2 --out fields.csv

fields.csv is key,value lines for the paper's fill_results.py (--manual takes
several files): fields.<marker>.<A|B|ds|hom|vec|auc> for the chosen field and
fieldsworst.<marker>.<...> for the worst. Also prints both tables, and warns
where a run's settings disagree with the audit's recommendation.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

MARKERS = ("Ki67", "ER", "HER2", "PR", "BCI", "VS")
SHORT = {"hematoxylin": r"H", "eosin": r"E", "dab": r"DAB", "sirius_red": r"SR",
         "stain1": r"$s_1$", "stain2": r"$s_2$", "gray": r"grey"}
DEFAULT_PROJECTION = {0: "lifetime", 1: "birth"}        # as topo_i2i.losses
PROJ_SHORT = {"lifetime": "lt", "birth": "b", "death": "d"}


def stain(name: str) -> str:
    return SHORT.get(name, name.replace("_", r"\_"))


def field_tex(spec: str, combine: str) -> str:
    """'hematoxylin/eosin' -> 'H|E' (channel H of the H-E deconvolution);
    'hematoxylin+dab' -> 'H+DAB' (both channels, summed) or max(H,DAB)."""
    if "/" in spec:
        a, b = spec.split("/", 1)
        return r"%s\,$|$\,%s" % (stain(a), stain(b))
    if "+" in spec:
        a, b = spec.split("+", 1)
        if combine == "sum":
            return r"%s\,+\,%s" % (stain(a), stain(b))
        op = r"$\max$" if combine == "max" else combine
        return r"%s(%s,\,%s)" % (op, stain(a), stain(b))
    return stain(spec)


def hom_tex(dims, projection: str) -> str:
    parts = []
    for d in dims:
        how = DEFAULT_PROJECTION.get(d, "birth") if projection in (None, "auto") else projection
        parts.append(r"$H_%d$\,%s" % (int(d), PROJ_SHORT.get(how, how)))
    return ", ".join(parts)


def run_meta(base: str, marker: str, run: str):
    path = os.path.join(base, marker.lower(), "results", "%s_%s" % (marker, run),
                        "training_meta.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as fh:
        return (json.load(fh).get("config") or {}).get("topo") or {}, path


def row_of(topo: dict) -> dict:
    vec = "est." if (topo.get("vectors_A") or topo.get("vectors_B")) else "lit."
    combine = topo.get("combine", "max")
    return {"A": field_tex(topo["field_A"], combine),
            "B": field_tex(topo["field_B"], combine),
            "ds": r"%d$\times$" % int(topo.get("downsample", 1)),
            "hom": hom_tex(topo.get("dims", [0, 1]), topo.get("projection", "auto")),
            "vec": vec}


def audit_choice(audit: str):
    """{marker: recommended candidate} from decision.json."""
    path = os.path.join(audit, "decision.json")
    if not os.path.exists(path):
        print("[warn] no %s -- chosen AUROCs left empty" % path, file=sys.stderr)
        return {}
    with open(path) as fh:
        return {d["marker"]: d.get("recommended") for d in json.load(fh)}


def worst_auroc(worst_audit: str, marker: str):
    path = os.path.join(worst_audit, "recommended_%s.env" % marker)
    if not os.path.exists(path):
        return None
    m = re.search(r"screen AUROC ([0-9.]+)", open(path).read())
    return float(m.group(1)) if m else None


def same_setting(topo: dict, rec: dict) -> bool:
    fa, fb, ds, dims, proj = rec["setting"]
    want_dims = [int(d) for d in str(dims).split(",")]
    return (topo.get("field_A") == fa and topo.get("field_B") == fb
            and int(topo.get("downsample", 1)) == int(ds)
            and [int(d) for d in topo.get("dims", [])] == want_dims
            and (topo.get("projection") or "auto") == proj)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/work2/bz66izin-TopoCG")
    p.add_argument("--chosen-root", default=None, help="default ROOT/sweep_layout")
    p.add_argument("--worst-root", default=None, help="default ROOT/worstfield_layout")
    p.add_argument("--run", default="lt0.01_cyc0_trans1", help="the cell, without the marker")
    p.add_argument("--audit", default=None, help="the audit the chosen fields came from")
    p.add_argument("--worst-audit", default=None, help="topo-worst-field's --out")
    p.add_argument("--markers", nargs="+", default=list(MARKERS))
    p.add_argument("--out", default="fields.csv")
    a = p.parse_args()
    chosen_root = a.chosen_root or os.path.join(a.root, "sweep_layout")
    worst_root = a.worst_root or os.path.join(a.root, "worstfield_layout")
    recs = audit_choice(a.audit) if a.audit else {}

    rows, table = [], {"fields": [], "fieldsworst": []}
    for m in a.markers:
        for pre, base in (("fields", chosen_root), ("fieldsworst", worst_root)):
            topo, path = run_meta(base, m, a.run)
            if topo is None:
                print("[skip] %s %s: no %s" % (pre, m, path), file=sys.stderr)
                continue
            r = row_of(topo)
            if pre == "fields":
                rec = recs.get(m)
                if rec:
                    r["auc"] = "%.3f" % rec["confirm_auroc"]
                    if not same_setting(topo, rec):
                        print("[WARN] %s: the run's field differs from %s's recommendation %s"
                              % (m, a.audit, rec["setting"]), file=sys.stderr)
            elif a.worst_audit:
                auc = worst_auroc(a.worst_audit, m)
                if auc is not None:
                    r["auc"] = "%.3f" % auc
            for k, v in r.items():
                rows.append(("%s.%s.%s" % (pre, m.lower(), k), v))
            table[pre].append((m, r))

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# written by tools/field_table.py"])
        w.writerows(rows)
    for pre, title in (("fields", "chosen (main paper)"), ("fieldsworst", "worst (supplementary)")):
        print("\n%s" % title)
        print("%-5s %-28s %-28s %-5s %-28s %-5s %s" % ("", "phi_A", "phi_B", "ds", "hom", "vec", "AUROC"))
        for m, r in table[pre]:
            print("%-5s %-28s %-28s %-5s %-28s %-5s %s" % (m, r["A"], r["B"], r["ds"], r["hom"],
                                                         r["vec"], r.get("auc", "-")))
    print("\n%d values -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
