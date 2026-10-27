"""Tests for the detector-independent parts of the nuclei tools. Run: pytest -q

The detectors themselves need their own environments and weights; these cover
what every detector's results pass through."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from nuclei_detectors import rescaled  # noqa: E402
from nuclei_pilot import iou_match, variants  # noqa: E402
from select_config import Model  # noqa: E402


def boxes(spec, shape=(40, 40)):
    """Label image from (id, y0, y1, x0, x1) boxes."""
    lab = np.zeros(shape, np.int32)
    for i, y0, y1, x0, x1 in spec:
        lab[y0:y1, x0:x1] = i
    return lab


def test_iou_match_counts_pairs_at_and_above_half():
    ref = boxes([(1, 0, 10, 0, 10), (2, 20, 30, 20, 30), (3, 0, 4, 30, 34)])
    pred = boxes([(7, 0, 10, 0, 10),        # identical: IoU 1
                  (8, 20, 30, 25, 35),      # 50 / 150: IoU 1/3, no match
                  (9, 30, 40, 0, 10)])      # nothing under it
    tp, fp, fn, mean_iou = iou_match(ref, pred)
    assert (tp, fp, fn) == (1, 2, 2) and mean_iou == pytest.approx(1.0)
    half = boxes([(1, 0, 10, 0, 10)])
    shifted = boxes([(5, 0, 10, 0, 15)])    # 100 / 150 = 2/3
    assert iou_match(half, shifted)[0] == 1
    exactly = boxes([(5, 0, 10, 0, 20)])    # 100 / 200 = 0.5, counted, as stardist does
    assert iou_match(half, exactly)[:3] == (1, 0, 0)


def test_iou_match_on_empty_masks():
    empty = np.zeros((8, 8), np.int32)
    one = boxes([(1, 0, 4, 0, 4)], (8, 8))
    assert iou_match(empty, one) == (0, 1, 0, 0.0)
    assert iou_match(one, empty) == (0, 0, 1, 0.0)


def test_rescaled_returns_labels_at_the_tile_size():
    rgb = np.zeros((30, 50, 3), np.uint8)
    seen = {}

    def detector(x):
        seen["shape"] = x.shape
        lab = np.zeros(x.shape[:2], np.int32)
        lab[: x.shape[0] // 2, : x.shape[1] // 2] = 3
        return lab

    lab = rescaled(detector, rgb, 2.0)
    assert seen["shape"] == (60, 100, 3)              # the model saw it scaled
    assert lab.shape == (30, 50) and lab.dtype == np.int32
    assert set(np.unique(lab)) == {0, 3}              # nearest: ids survive
    assert (lab[:15, :25] == 3).all() and (lab[15:, :] == 0).all()


def test_variants_per_detector():
    assert variants("stardist")["ihc"] == ["he_rgb", "fluo_hdab", "fluo_h"]
    assert variants("instanseg") == {"he": ["instanseg"], "ihc": ["instanseg"]}


def test_model_reads_another_detectors_folder(tmp_path):
    def tiles_csv(d, tp):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "nuclei_tiles.csv"), "w") as f:
            f.write("tile,n_ref,n_pred,tp_r5\nt0,10,10,%d\n" % tp)
    tiles_csv(tmp_path, 4)
    tiles_csv(tmp_path / "nuclei_instanseg", 8)
    split = {"t0": "test"}
    assert Model(str(tmp_path), split, "5").nuclei("test")["recall"] == pytest.approx(0.4)
    assert Model(str(tmp_path), split, "5", "nuclei_instanseg").nuclei("test")["recall"] \
        == pytest.approx(0.8)


def _fake_eval(root, arm, names, recall_of, detector_dir=""):
    """An eval root with a 2-tile split (one sel, one test tile) per marker and
    one model per name, whose sel-half recall is recall_of(marker, name)."""
    for m in ("Ki67", "ER", "HER2", "PR"):
        d = root / m / "valA"
        d.mkdir(parents=True, exist_ok=True)
        (d / "val_test_split.csv").write_text("tile,slide,half\ns1_t0,s1,sel\ns2_t0,s2,test\n")
        for name in names:
            md = d / arm / (name % m) / detector_dir
            md.mkdir(parents=True)
            tp = int(round(100 * recall_of(m, name)))
            (md / "nuclei_tiles.csv").write_text(
                "tile,n_ref,n_pred,tp_r5\ns1_t0,100,100,%d\ns2_t0,100,100,50\n" % tp)


def test_select_config_ranks_the_rerun_on_its_detector(tmp_path):
    import subprocess
    cells = ["%s_lt0.0001_cyc0_trans1", "%s_lt0.001_cyc0_trans1",
             "%s_lt0.01_cyc0_trans1", "%s_lt0.1_cyc0_trans1"]
    best = {"%s_lt0.01_cyc0_trans1": 0.9}
    _fake_eval(tmp_path, "sweep", cells, lambda m, n: best.get(n, 0.7), "nuclei_cpsam")
    tool = os.path.join(os.path.dirname(__file__), "..", "tools", "select_config.py")
    out = subprocess.run([sys.executable, tool, "--eval-root", str(tmp_path), "--arms", "sweep",
                          "--detector", "cpsam", "--report-markers", "Ki67"],
                         capture_output=True, text=True, check=True).stdout
    assert "rerun   lt=0.01  cyc0 trans1" in out and "<- chosen" in out
    assert "(arm=sweep, task 2)" in out
    assert "CycleGAN small" not in out                  # no un-layouted baselines
    import json
    chosen = json.load(open(tmp_path / "selection_sweep_cpsam" / "selection.json"))["chosen"]
    assert chosen["lambda_topo"] == "0.01" and abs(chosen["mean_recall"] - 0.9) < 1e-9
    assert not (tmp_path / "selection").exists()       # the original selection untouched
