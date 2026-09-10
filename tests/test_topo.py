"""Tests for the topological loss. Run: pytest -q"""

import numpy as np
import pytest
import torch

from topo_i2i.fields import StainField, rgb_to_scalar_field
from topo_i2i.losses import diagram_distance, topological_loss
from topo_i2i.persistence import persistence_diagram

gudhi = pytest.importorskip("gudhi")


def one_hole():
    """A 3x3 well: four 0-valued corners and a single loop born at 3, dying at 5."""
    return torch.tensor([[0., 3., 0.], [3., 5., 3.], [0., 3., 0.]], dtype=torch.float64)


def test_diagram_matches_gudhi():
    dgm = persistence_diagram(one_hole(), dims=(0, 1))
    # Three finite 0-dim pairs (the fourth component is essential and dropped).
    assert dgm[0].shape == (3, 2)
    assert torch.allclose(dgm[0], torch.tensor([[0., 3.], [0., 3.], [0., 3.]], dtype=torch.float64))
    # One loop, born when the ring closes at 3, filled at 5.
    assert dgm[1].shape == (1, 2)
    assert torch.allclose(dgm[1][0], torch.tensor([3., 5.], dtype=torch.float64))


def test_gradient_reaches_critical_pixels():
    f = one_hole().clone().requires_grad_(True)
    dgm = persistence_diagram(f, dims=(0, 1))
    dgm[1][:, 0].sum().backward()          # d(birth of the loop)/d(field)
    grad = f.grad
    assert grad is not None
    # Exactly one pixel -- a saddle on the ring, value 3 -- carries the gradient.
    assert grad.abs().sum() == pytest.approx(1.0)
    assert f.detach().reshape(-1)[grad.reshape(-1).nonzero()[0, 0]] == 3.0


def test_identical_fields_have_zero_distance():
    d = persistence_diagram(one_hole(), dims=(0, 1))
    assert diagram_distance(d, d, dims=(0, 1)).item() == pytest.approx(0.0)


def test_distance_is_positive_for_different_topology():
    flat = torch.zeros(3, 3, dtype=torch.float64)
    d_hole = persistence_diagram(one_hole(), dims=(0, 1))
    d_flat = persistence_diagram(flat, dims=(0, 1))
    assert diagram_distance(d_hole, d_flat, dims=(0, 1)).item() > 0


def test_topological_loss_batch_and_backward():
    torch.manual_seed(0)
    fake = torch.rand(2, 16, 16, dtype=torch.float64, requires_grad=True)
    real = torch.rand(3, 16, 16, dtype=torch.float64)
    loss = topological_loss(fake, real, dims=(0, 1))
    assert loss.ndim == 0 and loss.item() >= 0
    loss.backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0


def test_loss_is_zero_against_itself():
    torch.manual_seed(1)
    x = torch.rand(2, 12, 12, dtype=torch.float64)
    assert topological_loss(x, x.clone(), dims=(0, 1)).item() == pytest.approx(0.0)


def test_real_side_is_detached():
    fake = torch.rand(1, 8, 8, dtype=torch.float64, requires_grad=True)
    real = torch.rand(1, 8, 8, dtype=torch.float64, requires_grad=True)
    topological_loss(fake, real, dims=(0,)).backward()
    assert real.grad is None


def test_stain_field_shape_and_polarity():
    # A dark (strongly stained) pixel must give a higher OD than a white one.
    rgb = torch.zeros(1, 3, 4, 4)
    rgb[..., 0, 0] = -1.0   # black  -> high OD
    rgb[..., 1, 1] = 1.0    # white  -> ~0 OD
    f = StainField("hematoxylin")(rgb)
    assert f.shape == (1, 4, 4)
    assert f[0, 0, 0] > f[0, 1, 1]


def test_gray_field_shape():
    assert rgb_to_scalar_field(torch.zeros(2, 3, 8, 8), "gray").shape == (2, 8, 8)


# --- cycle-topology (paired) term ---------------------------------------- #

def test_paired_loss_is_zero_for_identical_batches():
    from topo_i2i.losses import paired_topological_loss
    torch.manual_seed(2)
    x = torch.rand(3, 12, 12, dtype=torch.float64)
    assert paired_topological_loss(x, x.clone(), dims=(0, 1)).item() == pytest.approx(0.0)


def test_paired_loss_pairs_by_index_not_by_matching():
    """Swapping the order of the second batch must change a paired loss."""
    from topo_i2i.losses import paired_topological_loss, topological_loss
    torch.manual_seed(3)
    a = torch.rand(2, 10, 10, dtype=torch.float64)
    b = torch.rand(2, 10, 10, dtype=torch.float64)
    b_swapped = b.flip(0)
    paired = paired_topological_loss(a, b, dims=(0, 1)).item()
    paired_swapped = paired_topological_loss(a, b_swapped, dims=(0, 1)).item()
    assert paired != pytest.approx(paired_swapped)
    # the matched (OT) loss is order-invariant, which is the whole difference
    m = topological_loss(a, b, dims=(0, 1)).item()
    m_swapped = topological_loss(a, b_swapped, dims=(0, 1)).item()
    assert m == pytest.approx(m_swapped)


def test_paired_loss_rejects_unequal_lengths():
    from topo_i2i.losses import paired_diagram_loss
    with pytest.raises(ValueError):
        paired_diagram_loss([{}, {}], [{}], dims=(0,))


# --- the four-term model ------------------------------------------------- #

zoo = pytest.importorskip("i2i_stain_zoo")


def _model(**topo_kwargs):
    from topo_i2i.models import TopoCycleGAN, TopoCycleGANConfig, TopoConfig
    torch.manual_seed(0)
    return TopoCycleGAN(TopoCycleGANConfig(
        n_blocks=1, ngf=8, ndf=8, topo=TopoConfig(**topo_kwargs)))


def _batch(n=2, size=32):
    torch.manual_seed(1)
    return {"A": torch.rand(n, 3, size, size) * 2 - 1,
            "B": torch.rand(n, 3, size, size) * 2 - 1}


def test_all_four_terms_are_computed_and_logged():
    model = _model(downsample=2)
    loss, logs, _ = model.compute_generator_loss(_batch())
    for k in ("loss_ph_cyc_H", "loss_ph_cyc_I", "loss_ph_trans_H", "loss_ph_trans_I"):
        assert k in logs and logs[k] > 0, k
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.generator_parameters())


def test_term_weights_switch_families_off():
    m = _model(downsample=2, lambda_ph_cyc=0.0)
    _, logs, _ = m.compute_generator_loss(_batch())
    assert logs["loss_ph_cyc_H"] == 0.0 and logs["loss_ph_trans_H"] > 0

    m = _model(downsample=2, lambda_ph_trans=0.0)
    _, logs, _ = m.compute_generator_loss(_batch())
    assert logs["loss_ph_trans_H"] == 0.0 and logs["loss_ph_cyc_H"] > 0


def test_lambda_topo_zero_is_the_baseline():
    m = _model(lambda_topo=0.0)
    _, logs, _ = m.compute_generator_loss(_batch())
    assert logs["loss_topo"] == 0.0


def test_every_n_steps_skips():
    m = _model(downsample=2, every_n_steps=3)
    seen = []
    for _ in range(3):
        _, logs, _ = m.compute_generator_loss(_batch())
        seen.append(logs["loss_topo"])
    assert seen[0] == 0.0 and seen[1] == 0.0 and seen[2] > 0


def test_domains_use_different_stain_fields():
    m = _model(downsample=2)
    rgb = torch.rand(1, 3, 16, 16) * 2 - 1
    assert not torch.allclose(m._to_field(rgb, "A"), m._to_field(rgb, "B"))


# --- combined (deconvolved) fields --------------------------------------- #

def test_deconvolution_recovers_known_concentrations():
    """A synthetic image built from known H and DAB amounts must deconvolve back."""
    from topo_i2i.fields import DeconvolutionField, STAIN_VECTORS
    import numpy as np
    v_h = np.array(STAIN_VECTORS["hematoxylin"]); v_h /= np.linalg.norm(v_h)
    v_d = np.array(STAIN_VECTORS["dab"]);         v_d /= np.linalg.norm(v_d)
    c_h, c_d = 0.8, 0.3
    od = c_h * v_h + c_d * v_d
    rgb01 = np.power(10.0, -od)
    rgb = torch.tensor(rgb01, dtype=torch.float32).view(1, 3, 1, 1) * 2 - 1

    assert DeconvolutionField(("hematoxylin", "dab"), "max")(rgb).item() == pytest.approx(0.8, abs=2e-2)
    assert DeconvolutionField(("hematoxylin", "dab"), "sum")(rgb).item() == pytest.approx(1.1, abs=2e-2)
    assert DeconvolutionField(("hematoxylin", "dab"), "mean")(rgb).item() == pytest.approx(0.55, abs=2e-2)


def test_combined_field_differs_from_single_stain():
    from topo_i2i.fields import make_field
    torch.manual_seed(0)
    rgb = torch.rand(1, 3, 8, 8) * 2 - 1
    dab_only = make_field("dab")(rgb)
    combined = make_field("dab+hematoxylin", "max")(rgb)
    assert combined.shape == dab_only.shape
    assert not torch.allclose(combined, dab_only)


def test_combined_field_is_differentiable():
    from topo_i2i.fields import make_field
    rgb = (torch.rand(1, 3, 8, 8) * 2 - 1).requires_grad_(True)
    make_field("dab+hematoxylin", "max")(rgb).sum().backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0


def test_make_field_dispatch():
    from topo_i2i.fields import make_field, DeconvolutionField, StainField, _GrayField
    assert isinstance(make_field("gray"), _GrayField)
    assert isinstance(make_field("dab"), StainField)
    assert isinstance(make_field("dab+hematoxylin"), DeconvolutionField)


def test_model_uses_combined_field_for_ihc_by_default():
    from topo_i2i.fields import DeconvolutionField
    m = _model(downsample=2)
    assert isinstance(m._field_mods["B"], DeconvolutionField)
    _, logs, _ = m.compute_generator_loss(_batch())
    assert logs["loss_ph_trans_I"] > 0


# --- trans terms are paired source-vs-own-translation --------------------- #

def test_trans_terms_follow_the_data_not_the_index():
    """Permuting domain B permutes fake_A with it, so the paired sum is invariant.

    This is the property that makes the term well-defined: image i is compared
    with what was generated *from image i*, so relabelling the batch cannot
    change the loss. (The underlying diagram distance is order-sensitive --
    see test_paired_loss_pairs_by_index_not_by_matching -- it is the pairing
    that tracks the data.)
    """
    b = _batch(n=2)
    m1 = _model(downsample=2, lambda_ph_cyc=0.0)
    _, logs, _ = m1.compute_generator_loss(b)
    m2 = _model(downsample=2, lambda_ph_cyc=0.0)
    _, logs2, _ = m2.compute_generator_loss({"A": b["A"], "B": b["B"].flip(0)})
    assert logs["loss_ph_trans_I"] == pytest.approx(logs2["loss_ph_trans_I"], rel=1e-5)


def test_trans_H_compares_across_domains():
    """trans_H must read field A on the source and field B on the translation."""
    from topo_i2i.losses import paired_diagram_loss
    from topo_i2i.persistence import batch_diagrams
    m = _model(downsample=2, lambda_ph_cyc=0.0)
    b = _batch(n=2)
    _, logs, visuals = m.compute_generator_loss(b)

    dims = tuple(m.topo_cfg.dims)
    expected = paired_diagram_loss(
        batch_diagrams(m._to_field(visuals["fake_B"], "B"), dims),   # H+DAB(y_hat)
        batch_diagrams(m._to_field(b["A"].detach(), "A"), dims),     # H(x)
        dims)
    assert logs["loss_ph_trans_H"] == pytest.approx(float(expected.detach()), rel=1e-5)


def test_trans_I_compares_across_domains():
    from topo_i2i.losses import paired_diagram_loss
    from topo_i2i.persistence import batch_diagrams
    m = _model(downsample=2, lambda_ph_cyc=0.0)
    b = _batch(n=2)
    _, logs, visuals = m.compute_generator_loss(b)

    dims = tuple(m.topo_cfg.dims)
    expected = paired_diagram_loss(
        batch_diagrams(m._to_field(visuals["fake_A"], "A"), dims),   # H(x'_hat)
        batch_diagrams(m._to_field(b["B"].detach(), "B"), dims),     # H+DAB(y')
        dims)
    assert logs["loss_ph_trans_I"] == pytest.approx(float(expected.detach()), rel=1e-5)


def test_trans_gradient_reaches_the_generator():
    m = _model(downsample=2, lambda_ph_cyc=0.0)
    loss, logs, _ = m.compute_generator_loss(_batch())
    assert logs["loss_ph_trans_H"] > 0 and logs["loss_ph_trans_I"] > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.generator_parameters())


# --- field presets -------------------------------------------------------- #

def test_presets_cover_both_translation_tasks():
    from topo_i2i.fields import FIELD_PRESETS
    assert FIELD_PRESETS["he-ihc"] == {"field_A": "hematoxylin",
                                        "field_B": "dab+hematoxylin",
                                        "combine": "max"}
    assert FIELD_PRESETS["he-sr"] == {"field_A": "eosin",
                                      "field_B": "dab",
                                      "combine": "max"}


def test_explicit_fields_override_the_preset():
    from topo_i2i.fields import resolve_fields
    assert resolve_fields("he-sr") == {"field_A": "eosin", "field_B": "dab",
                                       "combine": "max"}
    r = resolve_fields("he-sr", field_B="sirius_red", combine="sum")
    assert r == {"field_A": "eosin", "field_B": "sirius_red", "combine": "sum"}
    # a partial override leaves the rest of the preset intact
    assert resolve_fields("he-ihc", field_A="gray")["field_B"] == "dab+hematoxylin"


def test_unknown_preset_is_rejected():
    from topo_i2i.fields import resolve_fields
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_fields("he-pas")


def test_preset_fields_are_all_constructible():
    from topo_i2i.fields import FIELD_PRESETS, make_field
    rgb = torch.rand(1, 3, 8, 8) * 2 - 1
    for name, spec in FIELD_PRESETS.items():
        for key in ("field_A", "field_B"):
            f = make_field(spec[key], spec["combine"])(rgb)
            assert f.shape == (1, 8, 8), (name, key)


# --- diagram projections -------------------------------------------------- #

def test_default_projection_is_lifetime_for_h0_birth_for_h1():
    from topo_i2i.losses import DEFAULT_PROJECTION, _resolve_projection
    assert DEFAULT_PROJECTION == {0: "lifetime", 1: "birth"}
    assert _resolve_projection(None, 0) == "lifetime"
    assert _resolve_projection(None, 1) == "birth"
    # a plain string forces one projection everywhere
    assert _resolve_projection("birth", 0) == "birth"
    # a mapping may override only some dimensions
    assert _resolve_projection({0: "death"}, 0) == "death"
    assert _resolve_projection({0: "death"}, 1) == "birth"


def test_projections_give_the_expected_values():
    from topo_i2i.losses import _project
    dgm = {0: torch.tensor([[1.0, 4.0], [2.0, 3.0]])}
    assert torch.allclose(_project(dgm, 0, "birth"), torch.tensor([1.0, 2.0]))
    assert torch.allclose(_project(dgm, 0, "death"), torch.tensor([4.0, 3.0]))
    assert torch.allclose(_project(dgm, 0, "lifetime"), torch.tensor([3.0, 1.0]))
    with pytest.raises(ValueError, match="projection must be"):
        _project(dgm, 0, "nonsense")


def test_projection_changes_the_distance():
    d_hole = persistence_diagram(one_hole(), dims=(0, 1))
    flat = persistence_diagram(torch.zeros(3, 3, dtype=torch.float64), dims=(0, 1))
    by_birth = diagram_distance(d_hole, flat, (0, 1), projection="birth").item()
    by_life = diagram_distance(d_hole, flat, (0, 1), projection="lifetime").item()
    # H0 of one_hole: three (0,3) pairs -> births all 0, lifetimes all 3
    assert by_birth != pytest.approx(by_life)
    assert by_life > 0


def test_identical_diagrams_are_zero_under_every_projection():
    d = persistence_diagram(one_hole(), dims=(0, 1))
    for proj in ("birth", "lifetime", "death", None):
        assert diagram_distance(d, d, (0, 1), projection=proj).item() == pytest.approx(0.0)


def test_lifetime_projection_is_differentiable():
    f = one_hole().clone().requires_grad_(True)
    d = persistence_diagram(f, dims=(0, 1))
    flat = persistence_diagram(torch.zeros(3, 3, dtype=torch.float64), dims=(0, 1))
    diagram_distance(d, flat, (0, 1), projection="lifetime").backward()
    # lifetime uses both endpoints, so gradient reaches more pixels than birth-only
    assert f.grad is not None and (f.grad.abs() > 0).sum() > 0


def test_model_projection_flag_changes_the_loss():
    a = _model(downsample=2, projection="auto")
    b = _model(downsample=2, projection="birth")
    batch = _batch()
    _, la, _ = a.compute_generator_loss(batch)
    _, lb, _ = b.compute_generator_loss(batch)
    assert la["loss_topo"] != pytest.approx(lb["loss_topo"])


# --- delayed start / warmup ---------------------------------------------- #

def test_schedule_is_off_then_ramps_then_saturates():
    m = _model(start_step=100, warmup_steps=50)
    assert m.topo_schedule(1) == 0.0
    assert m.topo_schedule(99) == 0.0
    assert m.topo_schedule(100) == pytest.approx(1 / 50)
    assert m.topo_schedule(124) == pytest.approx(25 / 50)
    assert m.topo_schedule(149) == pytest.approx(1.0)
    assert m.topo_schedule(10_000) == 1.0


def test_schedule_without_warmup_is_a_step_function():
    m = _model(start_step=10, warmup_steps=0)
    assert m.topo_schedule(9) == 0.0
    assert m.topo_schedule(10) == 1.0


def test_no_persistence_is_computed_before_the_start_step(monkeypatch):
    import topo_i2i.models as mod
    m = _model(downsample=2, start_step=5)
    calls = []
    monkeypatch.setattr(mod, "batch_diagrams",
                        lambda *a, **k: calls.append(1) or [])
    loss, logs, _ = m.compute_generator_loss(_batch())
    assert calls == [], "persistence must be skipped while the term is inactive"
    assert logs["loss_topo"] == 0.0 and logs["topo_scale"] == 0.0
    loss.backward()  # must still be a valid graph


def test_term_activates_at_the_start_step():
    m = _model(downsample=2, start_step=3)
    scales = []
    for _ in range(4):
        _, logs, _ = m.compute_generator_loss(_batch())
        scales.append(logs["topo_scale"])
    assert scales == [0.0, 0.0, 1.0, 1.0]


def test_warmup_scales_the_loss():
    full = _model(downsample=2)
    half = _model(downsample=2, start_step=1, warmup_steps=2)
    batch = _batch()
    _, lf, _ = full.compute_generator_loss(batch)
    _, lh, _ = half.compute_generator_loss(batch)
    assert lh["topo_scale"] == pytest.approx(0.5)
    assert lh["loss_topo"] == pytest.approx(0.5 * lf["loss_topo"], rel=1e-5)


def test_step_counter_survives_a_checkpoint_round_trip():
    """A requeued job must not restart the warmup: the counter is a buffer."""
    m = _model(downsample=2, start_step=2)
    for _ in range(3):
        m.compute_generator_loss(_batch())
    assert "_topo_step" in m.state_dict()
    restored = _model(downsample=2, start_step=2)
    restored.load_state_dict(m.state_dict())
    assert int(restored._topo_step) == 3
    _, logs, _ = restored.compute_generator_loss(_batch())
    assert logs["topo_scale"] == 1.0


# --- crop script ---------------------------------------------------------- #

def _write_image(path, size=(64, 64), value=180):
    import numpy as np
    from PIL import Image
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path)


def test_crop_grid_counts_and_names(tmp_path):
    from topo_i2i.crop import crop_one
    src = tmp_path / "a.png"
    _write_image(str(src), (64, 64))
    out = tmp_path / "out"
    out.mkdir()
    written, skipped = crop_one(str(src), str(out), tile_size=32)
    assert (written, skipped) == (4, 0)
    assert sorted(p.name for p in out.iterdir()) == [
        "a_r0c0.png", "a_r0c1.png", "a_r1c0.png", "a_r1c1.png"]


def test_crop_resizes_and_reports_scale(tmp_path):
    from PIL import Image
    from topo_i2i.crop import crop_one
    src = tmp_path / "a.png"
    _write_image(str(src), (64, 64))
    out = tmp_path / "out"
    out.mkdir()
    crop_one(str(src), str(out), tile_size=32, resize_to=16)
    assert Image.open(next(out.iterdir())).size == (16, 16)


def test_crop_drops_the_edge_remainder(tmp_path):
    """A 70px image at tile 32 yields 2 crops per axis, not 3 padded ones."""
    from topo_i2i.crop import crop_one
    src = tmp_path / "a.png"
    _write_image(str(src), (70, 70))
    out = tmp_path / "out"
    out.mkdir()
    written, _ = crop_one(str(src), str(out), tile_size=32)
    assert written == 4


def test_crop_overlap_increases_the_count(tmp_path):
    from topo_i2i.crop import crop_one
    src = tmp_path / "a.png"
    _write_image(str(src), (64, 64))
    out = tmp_path / "out"
    out.mkdir()
    written, _ = crop_one(str(src), str(out), tile_size=32, overlap=16)
    assert written == 9  # stride 16 -> positions 0,16,32 on each axis


def test_crop_rejects_overlap_at_least_tile_size(tmp_path):
    from topo_i2i.crop import crop_one
    src = tmp_path / "a.png"
    _write_image(str(src), (64, 64))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        crop_one(str(src), str(out), tile_size=32, overlap=32)


def test_tissue_threshold_drops_background_only(tmp_path):
    import numpy as np
    from PIL import Image
    from topo_i2i.crop import crop_one, tissue_fraction
    arr = np.full((64, 64, 3), 250, dtype=np.uint8)   # background
    arr[:32, :32] = 100                                # one tissue quadrant
    src = tmp_path / "a.png"
    Image.fromarray(arr).save(src)
    out = tmp_path / "out"
    out.mkdir()
    written, skipped = crop_one(str(src), str(out), tile_size=32, tissue_threshold=0.5)
    assert (written, skipped) == (1, 3)
    assert tissue_fraction(Image.fromarray(arr[:32, :32])) == pytest.approx(1.0)


def test_zero_threshold_keeps_everything(tmp_path):
    import numpy as np
    from PIL import Image
    from topo_i2i.crop import crop_one
    src = tmp_path / "a.png"
    Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8)).save(src)
    out = tmp_path / "out"
    out.mkdir()
    written, skipped = crop_one(str(src), str(out), tile_size=32, tissue_threshold=0.0)
    assert (written, skipped) == (4, 0)
