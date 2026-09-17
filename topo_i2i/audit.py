"""Decide, before any GPU time is spent, which topological loss a dataset supports.

A field-validation table is easy to over-read: sweeping 324 combinations and
reporting the best AUROC reports the maximum of 324 correlated noisy statistics,
which is above 0.5 even when nothing is there. This module turns that table into
a decision by a rule fixed in advance, so the number that reaches the paper is
one nobody chose after seeing it.

The protocol, in three stages:

  screen   Run the whole field grid on slice 1 under the STRICT control (wrong
           partners drawn from the same source image), once per stain-vector
           source. Take the top TOP_K rows. Nothing here is reported as a result.

  confirm  Re-score exactly those TOP_K on slice 2, disjoint from slice 1 by
           construction (same seed, offset by the slice length). The candidates
           were fixed before slice 2 was touched, so these AUROCs are honest up
           to a Bonferroni correction over the candidates actually tested.

  decide   A candidate is SUPPORTED if its slice-2 AUROC clears 0.5 by more than
           z * SE, one-sided at ALPHA with Bonferroni over every candidate
           confirmed for that marker. SE is the Hanley-McNeil error under the
           null, which depends only on the sample size -- the error estimated at
           the observed AUROC shrinks as that value rises and hits zero at a
           perfect 1.0, which would certify a fluke. If any candidate is
           supported, the best confirmed one sets the training fields and
           ph_trans stays on. If none
           is, the dataset gets a ph_cyc-only grid: comparing an image with its
           own reconstruction needs no cross-domain correspondence, so it stays
           valid exactly where ph_trans does not.

The unstratified arm is not used to pick anything. It runs so the gap between it
and the strict arm can be reported: that difference is how much of an apparent
correspondence signal was really the model recognising which slide a tile came
from, and it is the number this protocol exists to expose.

A ph_cyc-only verdict is a prediction, not a dead end -- the training grid still
contains ph_trans cells, so it tests the prediction rather than assuming it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import NormalDist

from topo_i2i.validate_fields import auroc_se

# ---------------------------------------------------------------------------
# The pre-registered rule. Changing these after looking at results is exactly
# the thing the protocol exists to prevent, so they live here as named
# constants rather than as CLI flags with defaults.
# ---------------------------------------------------------------------------
TOP_K = 5          # candidates carried from screen to confirm, per vector source
ALPHA = 0.05       # one-sided; the alternative is directional (AUROC > 0.5)
CHANCE = 0.5

#: Where a cell's screen output lands: <audit>/<marker>/screen_<vectors>_<ctrl>.json
SCREEN = "screen_%s_%s.json"
#: and each confirmation: <audit>/<marker>/confirm_<vectors>_<k>.json
CONFIRM = "confirm_%s_%02d.json"

KEYS = ("field_A", "field_B", "downsample", "dims", "projection")


def setting_of(row: dict) -> tuple:
    """The identity of a combination, for comparing rows across slices."""
    return tuple(row[k] for k in KEYS)


def label(setting: tuple) -> str:
    fa, fb, ds, dims, proj = setting
    return "%s / %s  ds=%s dims=%s proj=%s" % (fa, fb, ds, dims, proj)


def z_threshold(n_candidates: int) -> float:
    """One-sided z for ALPHA, Bonferroni-corrected over the candidates tested."""
    n = max(1, n_candidates)
    return NormalDist().inv_cdf(1.0 - ALPHA / n)


def null_se(n_tiles: int) -> float:
    """Hanley-McNeil standard error UNDER THE NULL, AUROC = 0.5.

    The SE estimated at the observed AUROC is the wrong denominator for a test
    of "is this better than chance": it shrinks as the observed value moves away
    from 0.5, and collapses to exactly 0 at a perfect 1.0, which would certify a
    four-tile fluke. The null SE depends only on the sample size, which is what a
    hypothesis test should condition on.
    """
    return auroc_se(CHANCE, n_tiles, n_tiles)


def supported(auroc: float, n_tiles: int, n_candidates: int) -> bool:
    se = null_se(n_tiles)
    return (auroc - CHANCE) > z_threshold(n_candidates) * se


def read(path: str):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# stage 2 input: which combinations to re-score
# ---------------------------------------------------------------------------
def top_candidates(screen: dict, top_k: int = TOP_K):
    """The top_k rows by screening AUROC, best first.

    Ties are broken toward the cheaper setting (coarser downsample), so a
    candidate that survives costs as little as it can.
    """
    rows = sorted(screen["rows"],
                  key=lambda r: (-r["auroc"], -int(r["downsample"])))
    return rows[:top_k]


def plan_flags(row: dict) -> str:
    """The topo-validate-fields flags that re-score exactly this one row."""
    return (" ".join(["--field-A %s" % row["field_A"],
                      "--field-B %s" % row["field_B"],
                      "--downsample %d" % row["downsample"],
                      "--dims-set %s" % row["dims"],
                      "--topo-projection %s" % row["projection"]]))


# ---------------------------------------------------------------------------
# stage 3: the decision
# ---------------------------------------------------------------------------
def decide_marker(marker_dir: str, vector_sources) -> dict:
    """Apply the rule to one marker's screen and confirm files."""
    out = {"marker": os.path.basename(marker_dir.rstrip("/")),
           "candidates": [], "arms": {}, "verdict": None}

    # Everything confirmed for this marker shares one correction, because the
    # decision is made once over all of them.
    confirmed = []
    for vec in vector_sources:
        screen_path = os.path.join(marker_dir, SCREEN % (vec, "strict"))
        if not os.path.exists(screen_path):
            continue
        screen = read(screen_path)
        arm = {"screen_best": max(r["auroc"] for r in screen["rows"]),
               "n_tiles": screen["n_tiles"], "combinations": screen["combinations"],
               "groups": screen.get("groups"), "stains": screen.get("stains")}
        unstrat_path = os.path.join(marker_dir, SCREEN % (vec, "unstratified"))
        if os.path.exists(unstrat_path):
            unstrat = read(unstrat_path)
            arm["unstratified_best"] = max(r["auroc"] for r in unstrat["rows"])
            # How much of the apparent signal was specimen recognition.
            arm["specimen_recognition_delta"] = (arm["unstratified_best"]
                                                 - arm["screen_best"])
        out["arms"][vec] = arm

        for path in sorted(glob.glob(os.path.join(marker_dir,
                                                  "confirm_%s_*.json" % vec))):
            row = read(path)["rows"][0]
            conf = read(path)
            confirmed.append({
                "vectors": vec, "setting": setting_of(row),
                "confirm_auroc": row["auroc"], "confirm_se": row["auroc_se"],
                "confirm_n": conf["n_tiles"], "confirm_offset": conf["offset"],
                "field_combine": conf["field_combine"], "stains": conf.get("stains"),
            })

    z = z_threshold(len(confirmed))
    for c in confirmed:
        c["z_required"] = z
        c["null_se"] = null_se(c["confirm_n"])
        c["z_observed"] = (c["confirm_auroc"] - CHANCE) / c["null_se"]
        c["supported"] = supported(c["confirm_auroc"], c["confirm_n"], len(confirmed))
    out["candidates"] = confirmed
    out["n_confirmed"] = len(confirmed)
    out["z_required"] = z

    winners = [c for c in confirmed if c["supported"]]
    if winners:
        # Rank by the CONFIRMATION AUROC, never the screening one -- the
        # screening value is the number selection inflated.
        best = max(winners, key=lambda c: (c["confirm_auroc"], c["setting"][2]))
        out["verdict"] = "ph_trans_supported"
        out["recommended"] = best
    else:
        out["verdict"] = "ph_cyc_only"
        out["recommended"] = (max(confirmed, key=lambda c: c["confirm_auroc"])
                              if confirmed else None)
    return out


def env_lines(decision: dict, fallback_stains=None):
    """The shell assignments slurm/_sweep_common.sh sources."""
    rec = decision.get("recommended")
    lines = ["# Generated by topo-audit -- the pre-registered field-selection",
             "# protocol chose these. Do not hand-edit; re-run the audit.",
             "# marker=%s verdict=%s" % (decision["marker"], decision["verdict"])]
    if rec is None:
        lines.append("# no candidates were confirmed; nothing to recommend")
        return lines
    fa, fb, ds, dims, proj = rec["setting"]
    stains = rec.get("stains") or fallback_stains or ""
    # ${VAR:-default} rather than a bare assignment, so sourcing this fills in
    # only what the submitter left unset and an explicit --export still wins.
    setting = [("FIELD_A", fa), ("FIELD_B", fb),
               ("FIELD_COMBINE", rec["field_combine"]),
               ("TOPO_DOWNSAMPLE", str(int(ds))),
               ("TOPO_DIMS", " ".join(dims.split(","))),
               ("TOPO_PROJECTION", proj),
               # An empty STAINS means the literature table, which is what the
               # fixed-vector arm winning actually implies -- it must not be
               # left pointing at an estimate the decision rejected.
               ("STAINS", stains),
               ("PH_TRANS_SUPPORTED",
                "1" if decision["verdict"] == "ph_trans_supported" else "0")]
    lines += ['export %s="${%s:-%s}"' % (k, k, v) for k, v in setting]
    return lines


def format_report(decisions) -> str:
    w = []
    w.append("Field-validation audit -- pre-registered decision")
    w.append("=" * 72)
    w.append("")
    w.append("Rule fixed before the run: screen the full grid on slice 1 under the")
    w.append("strict within-image control, carry the top %d per vector source to a" % TOP_K)
    w.append("disjoint slice 2, and call a setting supported only if its slice-2")
    w.append("AUROC clears 0.5 one-sided at alpha=%.2f, Bonferroni-corrected over" % ALPHA)
    w.append("every candidate confirmed for that marker.")
    w.append("")

    for d in decisions:
        w.append("")
        w.append("-" * 72)
        w.append("%s -- %s" % (d["marker"], d["verdict"].upper().replace("_", " ")))
        w.append("-" * 72)
        w.append("  screening (slice 1, %d combinations on %d tiles)"
                 % (d["arms"][sorted(d["arms"])[0]]["combinations"],
                    d["arms"][sorted(d["arms"])[0]]["n_tiles"]))
        w.append("  %-9s %8s %14s %11s" % ("vectors", "strict", "unstratified",
                                           "inflation"))
        for vec, arm in sorted(d["arms"].items()):
            if "specimen_recognition_delta" in arm:
                w.append("  %-9s %8.3f %14.3f %11s"
                         % (vec, arm["screen_best"], arm["unstratified_best"],
                            "%+.3f" % arm["specimen_recognition_delta"]))
            else:
                w.append("  %-9s %8.3f %14s %11s" % (vec, arm["screen_best"], "-", "-"))
        w.append("  inflation = unstratified - strict: how much of the apparent")
        w.append("  signal was the model recognising which slide a tile came from.")
        w.append("  Negative means the strict control found no inflation to remove.")

        if d["candidates"]:
            width = max(len(label(c["setting"])) for c in d["candidates"])
            w.append("")
            w.append("  confirmation (slice 2, held out; z required %.2f over "
                     "%d candidates)" % (d["z_required"], d["n_confirmed"]))
            w.append("  %-9s %-*s %6s %7s %6s %4s"
                     % ("vectors", width, "setting", "n", "AUROC", "z", "ok"))
            for c in sorted(d["candidates"], key=lambda c: -c["confirm_auroc"]):
                w.append("  %-9s %-*s %6d %7.3f %6.2f %4s"
                         % (c["vectors"], width, label(c["setting"]),
                            c["confirm_n"], c["confirm_auroc"],
                            c["z_observed"], "yes" if c["supported"] else "no"))
        w.append("")
        rec = d.get("recommended")
        if d["verdict"] == "ph_trans_supported":
            w.append("  => ph_trans is supported. Train with:")
            w.append("       %s" % label(rec["setting"]))
            w.append("       vectors: %s" % (rec["stains"] or "literature table"))
            w.append("     Run the full grid; both PH terms are justified.")
        else:
            w.append("  => No setting survived the held-out slice, so there is no")
            w.append("     evidence that topology transfers between these domains at")
            w.append("     tile scale. ph_trans has nothing to learn from.")
            if rec:
                w.append("     (best confirmed: %.3f, z=%.2f, needed %.2f)"
                         % (rec["confirm_auroc"], rec["z_observed"], d["z_required"]))
            w.append("     Train the ph_cyc-only grid -- it compares an image with its")
            w.append("     own reconstruction in one field, so it needs no cross-domain")
            w.append("     correspondence and stays valid here. Keep the ph_trans cells")
            w.append("     in the grid: they now test this prediction instead of")
            w.append("     assuming it.")

    w.append("")
    w.append("=" * 72)
    supported_markers = [d["marker"] for d in decisions
                         if d["verdict"] == "ph_trans_supported"]
    w.append("ph_trans supported on: %s"
             % (", ".join(supported_markers) if supported_markers else "(none)"))
    w.append("ph_cyc-only on:        %s"
             % (", ".join(d["marker"] for d in decisions
                          if d["verdict"] == "ph_cyc_only") or "(none)"))

    # A setting that wins on one marker and nowhere else is probably noise.
    tally = {}
    for d in decisions:
        for c in d["candidates"]:
            if c["supported"]:
                tally.setdefault(c["setting"], []).append(d["marker"])
    if tally:
        w.append("")
        w.append("Settings supported on more than one marker (what to trust):")
        for setting, markers in sorted(tally.items(), key=lambda kv: -len(kv[1])):
            if len(markers) > 1:
                w.append("  %-52s %s" % (label(setting), ", ".join(markers)))
    return "\n".join(w) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="print the flags that re-score the top-K "
                                     "rows of a screening run")
    pl.add_argument("--screen", required=True, help="screen_*.json")
    pl.add_argument("--top-k", type=int, default=TOP_K)

    de = sub.add_parser("decide", help="apply the rule and write the recommendation")
    de.add_argument("--dir", required=True, help="audit directory, one subdir per marker")
    de.add_argument("--markers", nargs="*", default=None,
                    help="default: every subdirectory that has a screen file")
    de.add_argument("--vectors", nargs="+", default=["estimated", "fixed"])
    de.add_argument("--out", default=None, help="default: --dir")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "plan":
        for row in top_candidates(read(args.screen), args.top_k):
            print(plan_flags(row))
        return

    out_dir = args.out or args.dir
    markers = args.markers
    if not markers:
        markers = sorted(d for d in os.listdir(args.dir)
                         if glob.glob(os.path.join(args.dir, d, "screen_*.json")))
    if not markers:
        raise SystemExit("no screening results under %s -- did the array job run?"
                         % args.dir)

    decisions = [decide_marker(os.path.join(args.dir, m), args.vectors)
                 for m in markers]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "decision.json"), "w") as fh:
        json.dump([{k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in d.items()} for d in decisions], fh, indent=2,
                  default=list)
    report = format_report(decisions)
    with open(os.path.join(out_dir, "RECOMMENDATION.txt"), "w") as fh:
        fh.write(report)
    for d in decisions:
        path = os.path.join(out_dir, "recommended_%s.env" % d["marker"])
        with open(path, "w") as fh:
            fh.write("\n".join(env_lines(d)) + "\n")
    print(report)
    print("wrote %s/RECOMMENDATION.txt, decision.json and recommended_<marker>.env"
          % out_dir)


if __name__ == "__main__":
    main()
