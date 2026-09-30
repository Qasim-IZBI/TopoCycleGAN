"""Write the audit's WORST-scoring field setting out as a training env file.

The audit picks a field by ranking 324 combinations, then the sweep trains on
the winner. That leaves the obvious question unanswered: does the ranking
predict anything about training at all? The test is to train the other end of
the same table and see whether it does worse.

This reads a finished audit's screening table and emits `recommended_<M>.env`
files in exactly the format `slurm/_sweep_common.sh` already sources, so the
control sweep runs through the same path as the real one and differs only in
which row of the table it was handed.

    topo-worst-field --audit field_audit_vs2 --out field_worst_vs2

Two things to be careful about, both of which are why this prints the numbers
it picked rather than only writing the file:

* The minimum of 324 correlated noisy statistics is biased low, exactly as the
  maximum is biased high. The audit answers that for its winner by confirming
  it on a disjoint slice; the bottom row has no such confirmation, so its
  screening AUROC is an optimistic account of how bad it is.
* The lowest AUROC is not always the least informative setting. A value well
  below 0.5 is anti-correlated -- a true pair scores HIGHER than a shuffled one
  -- which is structure, not its absence, and a sign flip would recover it. The
  genuinely uninformative row is the one closest to 0.5. `--criterion` picks
  between the two; they are different experiments and the right one depends on
  whether you are testing "does the ranking matter" or "does topology matter".
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from topo_i2i.audit import KEYS, SCREEN, env_lines, label, read, setting_of


def rank_rows(rows, criterion: str):
    """Rows ordered worst-first under the chosen criterion."""
    if criterion == "lowest":
        return sorted(rows, key=lambda r: r["auroc"])
    if criterion == "uninformative":
        return sorted(rows, key=lambda r: abs(r["auroc"] - 0.5))
    raise ValueError("unknown criterion %r" % (criterion,))


def markers_under(audit_dir: str):
    return sorted(d for d in os.listdir(audit_dir)
                  if glob.glob(os.path.join(audit_dir, d, "screen_*.json")))


def pick(audit_dir: str, marker: str, vectors, criterion: str, rank: int):
    """(row, payload, vectors) for the marker's worst screened combination.

    Only the STRICT screen is read: the unstratified arm scores partly on
    specimen recognition, so its bottom row is not the field the audit would
    have rejected.
    """
    for vec in vectors:
        path = os.path.join(audit_dir, marker, SCREEN % (vec, "strict"))
        if not os.path.exists(path):
            continue
        payload = read(path)
        rows = payload.get("rows") or []
        if not rows:
            continue
        ordered = rank_rows(rows, criterion)
        if rank > len(ordered):
            raise SystemExit("%s: --rank %d but only %d combinations were screened"
                             % (marker, rank, len(ordered)))
        return ordered[rank - 1], payload, vec, ordered
    return None, None, None, None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", required=True,
                   help="a finished audit directory (the one run_audit.sh wrote)")
    p.add_argument("--out", required=True,
                   help="directory to write recommended_<marker>.env into; give it "
                        "its own, never the audit's, or the winner is overwritten")
    p.add_argument("--markers", nargs="*", default=None,
                   help="default: every marker with screening results")
    p.add_argument("--vectors", nargs="*", default=("estimated", "fixed"),
                   help="vector arms to look for, in order of preference")
    p.add_argument("--criterion", choices=("lowest", "uninformative"),
                   default="lowest",
                   help="lowest AUROC (the literal worst), or the row closest to "
                        "0.5 (the least informative). See the module docstring")
    p.add_argument("--rank", type=int, default=1,
                   help="take the Nth row from the bottom instead of the last")
    return p


def main() -> None:
    args = build_parser().parse_args()
    markers = args.markers or markers_under(args.audit)
    if not markers:
        raise SystemExit("no screening results under %s" % args.audit)

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for marker in markers:
        row, payload, vec, ordered = pick(args.audit, marker, args.vectors,
                                          args.criterion, args.rank)
        if row is None:
            print("[skip] %s: no strict screen under %s" % (marker, args.audit))
            continue

        setting = setting_of(row)
        best = ordered[-1] if args.criterion == "lowest" else max(
            ordered, key=lambda r: r["auroc"])
        # A decision-shaped dict so the env file is written by the audit's own
        # formatter -- the sweep must not be able to tell the two apart by
        # anything except the values.
        decision = {
            "marker": marker,
            # Deliberately not "ph_trans_supported": env_lines turns that into
            # PH_TRANS_SUPPORTED=0, and the sweep then logs that ph_trans is not
            # expected to help. For this control that note is the hypothesis.
            "verdict": "worst_field_control",
            "recommended": {"setting": setting,
                            "field_combine": payload.get("field_combine"),
                            "stains": payload.get("stains")},
        }
        lines = env_lines(decision)
        # env_lines opens with "the pre-registered protocol chose these", which
        # is the opposite of true here: the protocol RANKED THESE LAST. Leaving
        # that header on a control file would misrepresent it to anyone who
        # opens it later, so replace it rather than append a correction.
        lines[0] = "# Generated by topo-worst-field -- A NEGATIVE CONTROL."
        lines[1] = ("# The audit RANKED THIS SETTING LAST. It is here to be "
                    "trained against")
        lines.insert(2, "# the winner, not because anything recommends it. Do "
                        "not hand-edit.")
        how = ("lowest AUROC" if args.criterion == "lowest"
               else "AUROC nearest 0.5")
        lines.insert(4, "# picked by %s, from %d screened combinations, vectors=%s"
                        % (how, len(ordered), vec))
        lines.insert(5, "# screen AUROC %.4f; the best on that table was %.4f"
                        % (row["auroc"], best["auroc"]))
        lines.insert(6, "# NOT confirmed on the held-out slice -- the bottom of a "
                        "screened table is biased low.")
        # Only call it anti-correlated when it is below chance by more than
        # its own noise. The nearest-0.5 row lands a hair under half the time
        # and is the LEAST inverted row on the table, not an inverted one.
        margin = 2.0 * (row.get("auroc_se") or 0.0)
        if (0.5 - row["auroc"]) > margin:
            lines.insert(7, "# NOTE: below 0.5, i.e. ANTI-correlated -- true pairs "
                            "score HIGHER than")
            lines.insert(8, "#       shuffled ones. That is structure inverted, not "
                            "structure absent;")
            lines.insert(9, "#       --criterion uninformative picks the row nearest "
                            "0.5 instead.")
        path = os.path.join(args.out, "recommended_%s.env" % marker)
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        summary.append((marker, row["auroc"], best["auroc"], vec, label(setting)))
        print("%s -> %s" % (marker, path))

    if not summary:
        raise SystemExit("nothing written")
    print()
    print("%-6s %8s %8s %-10s %s" % ("marker", "worst", "best", "vectors", "setting"))
    for marker, worst, best, vec, lab in summary:
        print("%-6s %8.4f %8.4f %-10s %s" % (marker, worst, best, vec, lab))
    print()
    print("Train the control with:  AUDIT_DIR=%s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
