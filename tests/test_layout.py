"""Tests for the layout-consistency term. Run: pytest -q"""

import json

import pytest
import torch

from topo_i2i import layout
from topo_i2i.baseline_train import _output_of, _take_layout


def tiles(n=4, size=64, seed=0):
    """Blobs of lumen (bright) on tissue (dark), as [-1,1] RGB."""
    g = torch.Generator().manual_seed(seed)
    lay = torch.nn.functional.avg_pool2d(torch.rand(n, 1, size, size, generator=g), 9, 1, 4)
    lay = (lay > lay.flatten(1).median(1).values.view(-1, 1, 1, 1)).float()
    return (lay * 1.2 - 0.6).repeat(1, 3, 1, 1)


def test_upright_costs_nothing_inverted_costs_most():
    x = tiles()
    upright = x * 0.5 + 0.2                          # other contrast and brightness
    assert layout.layout_loss(x, upright).item() == pytest.approx(0.0, abs=1e-4)
    assert layout.layout_loss(x, -x).item() == pytest.approx(2.0, abs=1e-4)


def test_flat_source_tiles_are_skipped():
    x = tiles()
    x[:2] = 0.3                                      # no layout at all
    r, valid = layout.layout_corr(x, -x)
    assert valid.tolist() == [False, False, True, True]
    flat = torch.full_like(x, 0.3)
    gen = torch.rand_like(x, requires_grad=True)
    loss = layout.layout_loss(flat, gen)
    loss.backward()                                  # zero, but still differentiable
    assert loss.item() == 0.0


def test_gradient_pushes_a_partial_inversion_upright():
    x = tiles()
    noise = tiles(seed=1)
    gen = (-0.5 * x + noise).requires_grad_(True)    # leaning inverted, r < 0
    r0, _ = layout.layout_corr(x, gen.detach())
    assert (r0 < 0).all()
    layout.layout_loss(x, gen).backward()
    r1, _ = layout.layout_corr(x, gen.detach() - 1.0 * gen.grad)
    assert (r1 > r0).all()                           # one descent step: more upright


def test_a_perfect_inversion_is_a_flat_maximum():
    # r = -1 is the loss maximum, where the gradient vanishes -- why the term
    # is on from step 0, while runs are near r = 0, rather than a rescue later.
    x = tiles()
    gen = (-x).clone().requires_grad_(True)
    layout.layout_loss(x, gen).backward()
    assert gen.grad.abs().max() < 1e-6


def test_attach_adds_the_term_and_logs_it(tmp_path):
    x = tiles()

    class Model:
        def compute_generator_loss(self, batch):
            vis = {"real_A": batch["A"], "fake_B": -batch["A"],
                   "real_B": batch["B"], "fake_A": batch["B"]}
            return torch.tensor(1.0), {"loss_G": 1.0}, vis

    m = layout.attach(Model(), 0.5, str(tmp_path))
    loss, logs, _ = m.compute_generator_loss({"A": x, "B": x})
    assert logs["loss_layout"] == pytest.approx(2.0, abs=1e-4)
    assert loss.item() == pytest.approx(2.0, abs=1e-4)          # 1 + 0.5 * 2
    assert json.loads((tmp_path / "layout.json").read_text())["lambda_layout"] == 0.5


def test_weight_zero_leaves_the_model_alone(tmp_path):
    class Model:
        def compute_generator_loss(self, batch):
            return "untouched"
    m = Model()
    layout.attach(m, 0.0, str(tmp_path))
    assert m.compute_generator_loss(None) == "untouched"
    assert not (tmp_path / "layout.json").exists()


def test_a_run_cannot_resume_with_another_weight(tmp_path):
    layout.check_record(str(tmp_path), 0.5)
    layout.check_record(str(tmp_path), 0.5)          # same weight: fine
    for other in (0.0, 1.0):
        with pytest.raises(SystemExit):
            layout.check_record(str(tmp_path), other)


def test_baseline_cli_takes_the_flag_out_for_the_zoo():
    w, rest = _take_layout(["--model", "cyclegan", "--lambda-layout", "0.5", "--output", "o"])
    assert w == 0.5 and rest == ["--model", "cyclegan", "--output", "o"]
    w, rest = _take_layout(["--lambda-layout=2", "--steps", "10"])
    assert w == 2.0 and rest == ["--steps", "10"]
    assert _take_layout(["--steps", "10"]) == (0.0, ["--steps", "10"])
    assert _output_of(["--output", "runs/x"]) == "runs/x"
    assert _output_of(["--steps", "1"]) is None
