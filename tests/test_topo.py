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
    assert FIELD_PRESETS["he-ihc"] == {"field_A": "hematoxylin/eosin",
                                       "field_B": "dab+hematoxylin",
                                       "combine": "max"}
    assert FIELD_PRESETS["he-sr"] == {"field_A": "eosin/hematoxylin",
                                      "field_B": "dab/hematoxylin",
                                      "combine": "max"}


def test_explicit_fields_override_the_preset():
    from topo_i2i.fields import resolve_fields
    assert resolve_fields("he-sr") == {"field_A": "eosin/hematoxylin",
                                       "field_B": "dab/hematoxylin",
                                       "combine": "max"}
    r = resolve_fields("he-sr", field_B="sirius_red/hematoxylin", combine="sum")
    assert r == {"field_A": "eosin/hematoxylin",
                 "field_B": "sirius_red/hematoxylin", "combine": "sum"}
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


# --- single-channel deconvolution ('a/b' specs) --------------------------- #

def _stain_pixel(name, amt=1.0):
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    v = np.array(STAIN_VECTORS[name]); v /= np.linalg.norm(v)
    return torch.tensor(np.power(10.0, -(v*amt)), dtype=torch.float32).view(1,3,1,1)*2-1


def test_slash_spec_isolates_the_first_stain():
    """'a/b' must return a's concentration with b solved for and removed."""
    from topo_i2i.fields import make_field
    f = make_field("hematoxylin/eosin")
    assert f(_stain_pixel("hematoxylin")).item() == pytest.approx(1.0, abs=1e-3)
    assert f(_stain_pixel("eosin")).item() == pytest.approx(0.0, abs=1e-3)

    g = make_field("dab/hematoxylin")
    assert g(_stain_pixel("dab")).item() == pytest.approx(1.0, abs=1e-3)
    assert g(_stain_pixel("hematoxylin")).item() == pytest.approx(0.0, abs=1e-3)


def test_slash_spec_recovers_known_mixtures():
    import numpy as np
    from topo_i2i.fields import make_field, STAIN_VECTORS
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    f = make_field("hematoxylin/eosin")
    for aH, aE in ((0.8, 0.2), (0.2, 0.8), (0.5, 0.5)):
        rgb = np.power(10.0, -(aH*H + aE*E))
        px = torch.tensor(rgb, dtype=torch.float32).view(1,3,1,1)*2-1
        assert f(px).item() == pytest.approx(aH, abs=1e-3)


def test_projection_does_not_separate_stains():
    """Documents why the bare name is wrong: it answers ~0.8 for the wrong stain."""
    from topo_i2i.fields import make_field
    f = make_field("hematoxylin")
    assert f(_stain_pixel("dab")).item() > 0.7
    assert f(_stain_pixel("eosin")).item() > 0.7


def test_slash_spec_is_differentiable():
    from topo_i2i.fields import make_field
    rgb = (torch.rand(1, 3, 8, 8)*2-1).requires_grad_(True)
    make_field("hematoxylin/eosin")(rgb).sum().backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0


def test_presets_use_deconvolved_channels():
    from topo_i2i.fields import FIELD_PRESETS, DeconvolutionField, make_field
    assert FIELD_PRESETS["he-ihc"]["field_A"] == "hematoxylin/eosin"
    assert FIELD_PRESETS["he-sr"]["field_A"] == "eosin/hematoxylin"
    for preset in FIELD_PRESETS.values():
        for key in ("field_A", "field_B"):
            f = make_field(preset[key], preset["combine"])
            assert isinstance(f, DeconvolutionField), (preset, key)


def test_channel_argument_is_validated():
    from topo_i2i.fields import DeconvolutionField
    with pytest.raises(ValueError, match="channel must be"):
        DeconvolutionField(("hematoxylin", "dab"), channel=2)


# --- inference ------------------------------------------------------------ #

def test_zoo_inference_cannot_load_our_checkpoints():
    """Documents why topo_i2i.inference exists rather than reusing i2i-inference."""
    from dataclasses import asdict
    from i2i_stain_zoo.models import CycleGAN, CycleGANConfig
    from topo_i2i.models import TopoCycleGAN, TopoCycleGANConfig
    m = TopoCycleGAN(TopoCycleGANConfig(n_blocks=1, ngf=8, ndf=8))
    with pytest.raises(TypeError, match="topo"):
        CycleGANConfig(**asdict(m.cfg))
    plain = CycleGAN(CycleGANConfig(n_blocks=1, ngf=8, ndf=8))
    with pytest.raises(RuntimeError):
        plain.load_state_dict(m.state_dict(), strict=True)


def test_inference_round_trips_the_topo_config(tmp_path):
    from dataclasses import asdict
    from topo_i2i.inference import load_model
    from topo_i2i.models import TopoCycleGAN, TopoCycleGANConfig, TopoConfig

    cfg = TopoCycleGANConfig(n_blocks=1, ngf=8, ndf=8, lambda_cycle=3.0,
                             topo=TopoConfig(lambda_topo=0.002, field_B="hematoxylin/dab",
                                             start_step=123))
    m = TopoCycleGAN(cfg)
    path = tmp_path / "ckpt.pt"
    torch.save({"global_step": 7, "model": m.state_dict(), "config": asdict(cfg)}, path)

    restored = load_model(str(path), torch.device("cpu"))
    assert restored.cfg.lambda_cycle == 3.0
    assert restored.topo_cfg.lambda_topo == 0.002
    assert restored.topo_cfg.field_B == "hematoxylin/dab"
    assert restored.topo_cfg.start_step == 123
    assert not restored.training


def test_inference_forward_produces_a_valid_tile(tmp_path):
    from dataclasses import asdict
    from topo_i2i.inference import load_model, save_tile
    from topo_i2i.models import TopoCycleGAN, TopoCycleGANConfig
    cfg = TopoCycleGANConfig(n_blocks=1, ngf=8, ndf=8)
    m = TopoCycleGAN(cfg)
    path = tmp_path / "c.pt"
    torch.save({"model": m.state_dict(), "config": asdict(cfg)}, path)

    model = load_model(str(path), torch.device("cpu"))
    with torch.no_grad():
        y = model.forward_A2B(torch.rand(1, 3, 32, 32)*2-1)
    assert y.shape == (1, 3, 32, 32)
    out = tmp_path / "t.tif"
    save_tile(y, str(out))
    from PIL import Image
    assert Image.open(out).size == (32, 32)


def test_subdir_filter_keeps_tiles_and_drops_masks(tmp_path):
    """A raw per-case tiling carries masks beside the tiles; the loader walks both."""
    import os
    from PIL import Image
    from i2i_stain_zoo.datasets.common import list_images
    from topo_i2i.inference import filter_subdir

    for case in ("001", "002"):
        for sub in ("images", "masks"):
            d = tmp_path / case / sub
            d.mkdir(parents=True)
            for tile in ("0001248", "0001249"):
                Image.new("RGB", (4, 4)).save(d / (tile + ".tif"))

    found = list_images(str(tmp_path))
    assert len(found) == 8                      # the walk takes the masks too
    kept = filter_subdir(found, "images")
    assert len(kept) == 4
    assert not any(os.sep + "masks" + os.sep in p for p in kept)

    # Names stay unique across cases: inference writes the path relative to
    # --data, so a tile id repeated in another case does not overwrite it.
    stems = [os.path.relpath(p, str(tmp_path)) for p in kept]
    assert len(set(stems)) == len(stems)
    assert filter_subdir(found, "nope") == []


# --- comparison sheets ---------------------------------------------------- #

def test_compare_label_may_contain_an_equals_sign():
    """The cells want captioning as `lt=0.02 ...`, so the split is on the last =."""
    from topo_i2i.compare import parse_pred
    assert parse_pred("lt=0.02 cyc1 trans0=/preds/x") == ("lt=0.02 cyc1 trans0",
                                                          "/preds/x")
    assert parse_pred("baseline=/preds/b") == ("baseline", "/preds/b")
    with pytest.raises(Exception):
        parse_pred("no-separator-at-all")


def test_compare_keeps_a_slot_for_a_cell_that_has_not_been_inferred(tmp_path):
    """A cell still training must not shift every panel after it out of place."""
    import os
    from PIL import Image
    from topo_i2i.compare import (compose, find_match, index_by_stem, list_tiles)

    a = tmp_path / "valA"
    a.mkdir()
    for name in ("s1_r0c0.tif", "s1_r0c1.tif"):
        Image.new("RGB", (8, 8)).save(a / name)
    trained = tmp_path / "preds" / "cell0"
    trained.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(trained / "s1_r0c0.tif")
    untrained = tmp_path / "preds" / "cell1"
    untrained.mkdir()

    assert list_tiles(str(a)) == ["s1_r0c0.tif", "s1_r0c1.tif"]
    done = index_by_stem(str(trained))
    assert find_match("s1_r0c0", *done) is not None
    assert find_match("s1_r0c1", *done) is None          # inferred, but not this tile
    assert find_match("s1_r0c0", *index_by_stem(str(untrained))) is None

    # Every panel keeps its place whether or not it has an image: 15 slots at
    # three columns is five rows either way.
    full = compose([("c%d" % i, Image.new("RGB", (64, 64)), i < 2)
                    for i in range(15)], cols=3, size=64, title="t")
    holes = compose([("c%d" % i, None, i < 2) for i in range(15)],
                    cols=3, size=64, title="t")
    assert full.size == holes.size
    assert full.size[0] == 3 * 64 + 4 * 4                # cols * size + pad


def test_compare_subdir_keeps_tiles_and_drops_masks(tmp_path):
    """The same per-case layout topo-infer has to filter, filtered the same way."""
    import os
    from PIL import Image
    from topo_i2i.compare import list_tiles

    for case in ("001", "012"):
        for sub in ("images", "masks"):
            d = tmp_path / case / sub
            d.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(d / "0001248.tif")

    assert len(list_tiles(str(tmp_path))) == 4
    kept = list_tiles(str(tmp_path), "images")
    assert kept == [os.path.join("001", "images", "0001248.tif"),
                    os.path.join("012", "images", "0001248.tif")]


# --- per-channel cycle topology ------------------------------------------ #

def test_split_specs():
    from topo_i2i.fields import split_specs
    assert split_specs("dab+hematoxylin") == ("dab/hematoxylin", "hematoxylin/dab")
    assert split_specs("hematoxylin/dab") == ("hematoxylin/dab", "dab/hematoxylin")
    with pytest.raises(ValueError, match="cannot split"):
        split_specs("hematoxylin")


def test_split_changes_only_the_ihc_cycle_term():
    batch = _batch(n=1)
    plain = _model(downsample=2, field_B="dab+hematoxylin", combine="sum")
    split = _model(downsample=2, field_B="dab+hematoxylin", combine="sum",
                   ph_cyc_split=True)
    _, lp, _ = plain.compute_generator_loss(batch)
    _, ls, _ = split.compute_generator_loss(batch)
    assert ls["loss_ph_cyc_I"] != pytest.approx(lp["loss_ph_cyc_I"])
    for k in ("loss_ph_cyc_H", "loss_ph_trans_H", "loss_ph_trans_I"):
        assert ls[k] == pytest.approx(lp[k], rel=1e-5), k


def test_split_sees_swapped_positivity_that_the_merge_misses():
    """The error mode the split exists for: same positive count, different nuclei."""
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS, make_field
    from topo_i2i.persistence import persistence_diagram
    from topo_i2i.losses import diagram_distance

    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    n, N = 96, 16
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.random.default_rng(0)
    centres = r.integers(8, n-8, (N, 2))

    def render(pos):
        od = np.zeros((n, n, 3))
        for (cy, cx), p in zip(centres, pos):
            od += np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*3.0**2)))[..., None] * (D if p else H)
        return torch.tensor(np.clip(10**(-od), 0, 1),
                            dtype=torch.float32).permute(2, 0, 1)[None]*2-1

    truth = np.array([i % 2 == 0 for i in range(N)])
    real, swapped = render(truth), render(~truth)

    def dist(spec, a, b):
        f = make_field(spec, "sum")
        return float(diagram_distance(persistence_diagram(-f(a)[0].double(), (0, 1)),
                                      persistence_diagram(-f(b)[0].double(), (0, 1)), (0, 1)))

    merged = dist("dab+hematoxylin", real, swapped)
    split = (dist("hematoxylin/dab", real, swapped) + dist("dab/hematoxylin", real, swapped))
    assert split > 5 * merged, (merged, split)


def test_split_is_off_by_default():
    from topo_i2i.models import TopoConfig
    assert TopoConfig().ph_cyc_split is False
    from topo_i2i.train import build_parser
    assert build_parser().parse_args("--dataA a --dataB b".split()).ph_cyc_split is False


# --- field validation against registered pairs ---------------------------- #

def test_auroc_endpoints():
    from topo_i2i.validate_fields import auroc
    assert auroc([1, 2], [3, 4]) == pytest.approx(1.0)      # true always lower
    assert auroc([3, 4], [1, 2]) == pytest.approx(0.0)      # always higher
    assert auroc([1, 1], [1, 1]) == pytest.approx(0.5)      # all ties
    assert auroc([1, 3], [2, 4]) == pytest.approx(0.75)


def test_matched_pairs_uses_filenames(tmp_path):
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for name in ("t1.png", "t2.png", "t3.png"):
        _write_image(str(a/name)); _write_image(str(b/name))
    _write_image(str(a/"only_in_a.png"))
    pairs, how = matched_pairs(str(a), str(b))
    assert len(pairs) == 3 and "by filename" in how
    assert all(x == y for x, y in pairs)


def test_matched_pairs_warns_when_names_differ(tmp_path):
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    _write_image(str(a/"x1.png")); _write_image(str(b/"y1.png"))
    pairs, how = matched_pairs(str(a), str(b))
    assert len(pairs) == 1 and "WARNING" in how


def test_matched_pairs_respects_limit(tmp_path):
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(5):
        _write_image(str(a/("t%d.png" % i))); _write_image(str(b/("t%d.png" % i)))
    assert len(matched_pairs(str(a), str(b), limit=2)[0]) == 2


def test_registered_pairs_score_below_shuffled():
    """The property the whole check rests on, on data where truth is known."""
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.persistence import persistence_diagram
    from topo_i2i.losses import diagram_distance
    from topo_i2i.fields import make_field
    from topo_i2i.validate_fields import auroc

    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]

    def tile(seed):
        r = np.random.default_rng(seed)
        cs = r.integers(6, n-6, (r.integers(6, 16), 2))
        pos = r.random(len(cs)) < 0.4
        he, ihc = np.zeros((n, n, 3)), np.zeros((n, n, 3))
        for (cy, cx), p in zip(cs, pos):
            b = np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*2.5**2)))[..., None]
            he += b*H
            ihc += b*(D if p else H)
        to = lambda od: torch.tensor(np.clip(10**(-od), 0, 1),
                                     dtype=torch.float32).permute(2, 0, 1)[None]*2-1
        return to(he), to(ihc)

    fa, fb = make_field("hematoxylin/eosin"), make_field("dab+hematoxylin", "sum")
    tiles = [tile(s) for s in range(8)]
    da = [persistence_diagram(-fa(h)[0].double(), (0, 1)) for h, _ in tiles]
    db = [persistence_diagram(-fb(i)[0].double(), (0, 1)) for _, i in tiles]

    true = [float(diagram_distance(da[i], db[i], (0, 1))) for i in range(8)]
    shuf = [float(diagram_distance(da[i], db[j], (0, 1)))
            for i in range(8) for j in range(8) if i != j]
    assert np.mean(true) < np.mean(shuf)
    assert auroc(true, shuf) > 0.75


def test_validate_fields_sweeps_downsample():
    from topo_i2i.validate_fields import build_parser
    ns = build_parser().parse_args(
        "--dataA a --dataB b --downsample 1 2 4".split())
    assert ns.downsample == [1, 2, 4]
    assert build_parser().parse_args("--dataA a --dataB b".split()).downsample == [1]


def test_validate_fields_sweeps_dims_and_projection():
    from topo_i2i.validate_fields import build_parser
    ns = build_parser().parse_args(
        "--dataA a --dataB b --dims-set 0 1 0,1 --topo-projection auto birth".split())
    assert ns.dims_set == ["0", "1", "0,1"]
    assert ns.topo_projection == ["auto", "birth"]
    d = build_parser().parse_args("--dataA a --dataB b".split())
    assert d.dims_set == ["0,1"] and d.topo_projection == ["auto"]


def test_matched_pairs_offset_gives_a_disjoint_slice(tmp_path):
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(10):
        _write_image(str(a/("t%02d.png" % i))); _write_image(str(b/("t%02d.png" % i)))
    first, _ = matched_pairs(str(a), str(b), limit=4)
    second, how = matched_pairs(str(a), str(b), limit=4, offset=4)
    assert len(first) == len(second) == 4
    assert not (set(f for f, _ in first) & set(f for f, _ in second))
    assert "skipping the first 4" in how


def test_random_sampling_spans_the_directory(tmp_path):
    """head takes all crops of the first slides; random spreads across them."""
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for slide in range(10):
        for crop in range(4):
            name = "s%02d_r%dc0.png" % (slide, crop)
            _write_image(str(a/name)); _write_image(str(b/name))

    slides = lambda pairs: {f.split("_")[0] for f, _ in pairs}
    head, _ = matched_pairs(str(a), str(b), limit=8, sample="head")
    rand, _ = matched_pairs(str(a), str(b), limit=8, sample="random", seed=0)
    assert len(slides(head)) == 2          # 8 crops = only the first 2 slides
    assert len(slides(rand)) > 2           # spread over more


def test_offset_is_disjoint_only_with_the_same_seed(tmp_path):
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(20):
        _write_image(str(a/("t%02d.png" % i))); _write_image(str(b/("t%02d.png" % i)))
    names = lambda p: {f for f, _ in p}
    s1 = names(matched_pairs(str(a), str(b), limit=5, offset=0, seed=0)[0])
    s2 = names(matched_pairs(str(a), str(b), limit=5, offset=5, seed=0)[0])
    assert not (s1 & s2)
    s3 = names(matched_pairs(str(a), str(b), limit=5, offset=5, seed=99)[0])
    assert s1 & s3          # a different seed re-permutes, so slices overlap


def test_sample_mode_is_validated(tmp_path):
    from topo_i2i.validate_fields import matched_pairs
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    _write_image(str(a/"t.png")); _write_image(str(b/"t.png"))
    with pytest.raises(ValueError, match="sample must be"):
        matched_pairs(str(a), str(b), sample="nonsense")


def test_auroc_se_matches_a_bootstrap():
    """Hanley-McNeil SE should track the empirical spread, not sqrt(0.5/n)."""
    import numpy as np
    from topo_i2i.validate_fields import auroc, auroc_se
    rng = np.random.default_rng(0)
    draws = [auroc(rng.normal(0, 1, 128), rng.normal(0.36, 1, 640)) for _ in range(400)]
    empirical = float(np.std(draws))
    predicted = auroc_se(float(np.mean(draws)), 128, 640)
    assert predicted == pytest.approx(empirical, abs=0.01)
    assert predicted < 0.5 * (0.5 / 128) ** 0.5      # far tighter than the naive bound


# --- within-slide shuffling control --------------------------------------- #

def test_slide_of_strips_the_crop_suffix():
    from topo_i2i.validate_fields import slide_of
    assert slide_of("t00_r0c0.png") == "t00"
    assert slide_of("img_5_r10c3.tif") == "img_5"
    assert slide_of("no_grid_suffix.png") == "no_grid_suffix"


def test_within_slide_shuffle_is_a_same_slide_derangement():
    import numpy as np
    from topo_i2i.validate_fields import shuffled_partners, slide_of
    pairs = [("%s_r%dc%d.png" % (s, r, c),) * 2
             for s in ("A", "B", "C") for r in (0, 1) for c in (0, 1)]
    perm, usable = shuffled_partners(pairs, np.random.default_rng(0), True)
    assert len(usable) == len(pairs)
    for i, j in enumerate(perm):
        assert i != j, "a tile kept its own partner"
        assert slide_of(pairs[i][0]) == slide_of(pairs[j][0])


def test_within_slide_shuffle_leaves_lone_tiles_alone():
    """A slide with one tile has no alternative partner; it must not be mangled."""
    import numpy as np
    from topo_i2i.validate_fields import shuffled_partners
    pairs = [("A_r0c0.png",)*2, ("A_r0c1.png",)*2, ("LONE_r0c0.png",)*2]
    perm, usable = shuffled_partners(pairs, np.random.default_rng(0), True)
    assert usable == [0, 1]
    assert perm[2] == 2          # left in place, and excluded from the comparison


def test_unstratified_shuffle_is_still_a_derangement():
    import numpy as np
    from topo_i2i.validate_fields import shuffled_partners
    pairs = [("t%02d.png" % i,)*2 for i in range(12)]
    perm = shuffled_partners(pairs, np.random.default_rng(0), False)
    assert all(perm[i] != i for i in range(len(pairs)))


def test_within_slide_shuffles_differ_between_calls():
    """A fixed rotation would give identical permutations, inflating the sample."""
    import numpy as np
    from topo_i2i.validate_fields import shuffled_partners
    pairs = [("%s_r%dc%d.png" % (s, r, c),)*2
             for s in ("A", "B", "C", "D") for r in (0, 1) for c in (0, 1)]
    rng = np.random.default_rng(0)
    perms = {tuple(int(x) for x in shuffled_partners(pairs, rng, True)[0])
             for _ in range(10)}
    assert len(perms) > 1


def test_tiles_without_an_alternative_are_excluded_not_self_paired():
    """Self-paired tiles would put true distances into the shuffled set."""
    import numpy as np
    from topo_i2i.validate_fields import shuffled_partners
    pairs = [("A_r0c0.png",)*2, ("A_r0c1.png",)*2, ("LONE_r0c0.png",)*2]
    perm, usable = shuffled_partners(pairs, np.random.default_rng(0), True)
    assert usable == [0, 1]              # the lone tile is not in the comparison
    assert 2 not in usable
    for i in usable:
        assert perm[i] != i


def test_slide_regex_controls_the_grouping():
    from topo_i2i.validate_fields import slide_of
    assert slide_of("caseA_tile03_r1c0.png") == "caseA_tile03"
    assert slide_of("caseA_tile03_r1c0.png", r"_tile\d+_r\d+c\d+$") == "caseA"


# --- Macenko stain estimation --------------------------------------------- #

def _stain_tile(path, v1, v2, seed, n=128):
    import numpy as np
    from PIL import Image
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.random.default_rng(seed)
    od = np.zeros((n, n, 3))
    for _ in range(40):
        cy, cx = r.integers(5, n-5, 2)
        od += np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*3.0**2)))[..., None]*v1*r.uniform(.6, 1.3)
    for _ in range(15):
        cy, cx = r.integers(5, n-5, 2)
        od += np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*8.0**2)))[..., None]*v2*r.uniform(.3, .8)
    Image.fromarray((np.clip(10**(-od), 0, 1)*255).astype("uint8")).save(path)


def test_estimation_recovers_known_vectors(tmp_path):
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.stains import estimate_for_run, angle_between
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(12):
        _stain_tile(str(a/("t%02d.png" % i)), H, E, i)
        _stain_tile(str(b/("t%02d.png" % i)), H, D, i+500)

    res = estimate_for_run(str(a), str(b), limit=12)
    assert angle_between(res["A"]["stain1"], H) < 3.0
    assert angle_between(res["A"]["stain2"], E) < 3.0
    assert angle_between(res["B"]["stain2"], D) < 3.0
    # both domains' channel 1 must mean the same thing
    assert res["meta"]["shared_stain_angle_deg"] < 3.0


def test_order_like_uses_the_reference():
    import numpy as np
    from topo_i2i.stains import order_like
    a = np.array([1.0, 0.0, 0.0]); b = np.array([0.0, 1.0, 0.0])
    assert np.allclose(order_like((a, b), reference=b)[0], b)
    assert np.allclose(order_like((b, a), reference=b)[0], b)
    # with no reference, the larger red component leads
    assert np.allclose(order_like((b, a))[0], a)


def test_pin_shared_forces_the_domains_to_agree(tmp_path):
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.stains import estimate_for_run
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(12):
        _stain_tile(str(a/("t%02d.png" % i)), H, E, i)
        _stain_tile(str(b/("t%02d.png" % i)), H*0.97 + 0.03, D, i+500)
    res = estimate_for_run(str(a), str(b), limit=12, pin_shared=True)
    assert res["A"]["stain1"] == res["B"]["stain1"]
    assert res["meta"]["shared_stain_angle_deg"] == pytest.approx(0.0, abs=1e-9)


def test_estimated_vectors_reach_the_field(tmp_path):
    """make_field must use caller-supplied vectors, not the built-in table."""
    import numpy as np
    from topo_i2i.fields import make_field, STAIN_VECTORS
    custom = {"stain1": (0.1, 0.2, 0.97), "stain2": (0.9, 0.3, 0.3)}
    f = make_field("stain1/stain2", vectors=custom)
    v = np.array(custom["stain1"]); v /= np.linalg.norm(v)
    px = torch.tensor(np.power(10.0, -v), dtype=torch.float32).view(1, 3, 1, 1)*2-1
    assert f(px).item() == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(KeyError):
        make_field("stain1/stain2")          # unknown without the vectors


def test_low_separation_is_warned(tmp_path):
    import numpy as np
    from topo_i2i.stains import estimate_for_run
    v1 = np.array([0.60, 0.70, 0.38]); v1 /= np.linalg.norm(v1)
    v2 = np.array([0.62, 0.70, 0.35]); v2 /= np.linalg.norm(v2)   # nearly parallel
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(10):
        _stain_tile(str(a/("t%02d.png" % i)), v1, v2, i)
        _stain_tile(str(b/("t%02d.png" % i)), v1, v2, i+500)
    res = estimate_for_run(str(a), str(b), limit=10, min_separation=15.0)
    assert any("deg apart" in w for w in res["meta"]["warnings"])


def test_pixels_per_tile_caps_the_pooled_sample(tmp_path):
    """Coverage across slides should scale without the memory scaling with it."""
    import numpy as np
    from topo_i2i.stains import sample_od
    d = tmp_path / "A"
    d.mkdir()
    for i in range(6):
        _write_image(str(d / ("t%02d.png" % i)), (64, 64))
    full, n = sample_od(str(d), limit=6, pixels_per_tile=0)
    capped, _ = sample_od(str(d), limit=6, pixels_per_tile=100)
    assert n == 6
    assert full.shape[0] == 6 * 64 * 64
    assert capped.shape[0] == 6 * 100
    assert capped.shape[1] == 3


def test_capped_sampling_still_recovers_the_vectors(tmp_path):
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.stains import estimate_for_run, angle_between
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path/"A", tmp_path/"B"
    a.mkdir(); b.mkdir()
    for i in range(12):
        _stain_tile(str(a/("t%02d.png" % i)), H, E, i)
        _stain_tile(str(b/("t%02d.png" % i)), H, D, i+500)
    res = estimate_for_run(str(a), str(b), limit=12, pixels_per_tile=4000)
    assert angle_between(res["A"]["stain1"], H) < 4.0
    assert angle_between(res["B"]["stain2"], D) < 4.0
    assert res["meta"]["pixels_per_tile"] == 4000


# --- per-pair inspection --------------------------------------------------- #

def test_inspect_writes_every_intermediate(tmp_path):
    import json
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.inspect import build_parser, main
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path/"a.png", tmp_path/"b.png"
    _stain_tile(str(a), H, E, 0)
    _stain_tile(str(b), H, D, 1)
    stains = tmp_path/"s.json"
    stains.write_text(json.dumps({
        "A": {"stain1": list(H), "stain2": list(E), "separation_deg": 30.0},
        "B": {"stain1": list(H), "stain2": list(D), "separation_deg": 37.0},
        "meta": {}}))
    out = tmp_path/"out"

    import sys
    argv = sys.argv
    sys.argv = ["topo-inspect", "--imageA", str(a), "--imageB", str(b),
                "--stains", str(stains), "--image-size", "64", "--downsample", "1",
                "--outdir", str(out)]
    try:
        main()
    finally:
        sys.argv = argv

    for name in ("A_diagram.csv", "B_diagram.csv", "A_field.png", "B_field.png",
                 "A_stain1.png", "A_stain2.png", "summary.json"):
        assert (out/name).exists(), name

    s = json.loads((out/"summary.json").read_text())
    assert s["distance_total"] > 0
    assert set(s["distance_per_dim"]) == {"0", "1"}
    # the per-dimension distances must add up to the total
    assert sum(s["distance_per_dim"].values()) == pytest.approx(s["distance_total"], rel=1e-6)

    header, *rows = (out/"A_diagram.csv").read_text().strip().split("\n")
    assert header == "dim,birth,death,lifetime"
    dim, birth, death, life = rows[0].split(",")
    assert float(death) - float(birth) == pytest.approx(float(life), abs=1e-5)


def test_inspect_estimates_vectors_when_none_given(tmp_path):
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.inspect import estimate_from_images
    from topo_i2i.stains import angle_between
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path/"a.png", tmp_path/"b.png"
    _stain_tile(str(a), H, E, 3, n=192)
    _stain_tile(str(b), H, D, 4, n=192)
    v = estimate_from_images(str(a), str(b), 192)
    assert set(v) == {"A", "B"}
    assert angle_between(v["A"]["stain1"], H) < 10.0


def _run_inspect(tmp_path, field_b, outname="out"):
    import json, sys
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    from topo_i2i.inspect import main
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path/"a.png", tmp_path/"b.png"
    _stain_tile(str(a), H, E, 0); _stain_tile(str(b), H, D, 1)
    stains = tmp_path/"s.json"
    stains.write_text(json.dumps({
        "A": {"stain1": list(H), "stain2": list(E), "separation_deg": 30.0},
        "B": {"stain1": list(H), "stain2": list(D), "separation_deg": 37.0},
        "meta": {}}))
    out = tmp_path/outname
    argv = sys.argv
    sys.argv = ["topo-inspect", "--imageA", str(a), "--imageB", str(b),
                "--stains", str(stains), "--field-A", "stain1/stain2",
                "--field-B", field_b, "--image-size", "64", "--downsample", "1",
                "--outdir", str(out)]
    try:
        main()
    finally:
        sys.argv = argv
    return out


def test_merged_field_is_saved_when_the_spec_merges(tmp_path):
    out = _run_inspect(tmp_path, "stain1+stain2", "merged")
    assert (out/"B_combined.png").exists(), "the merged image the diagram uses must be saved"
    assert not (out/"A_combined.png").exists(), "field_A is a single channel here"


def test_no_combined_image_for_a_single_channel_spec(tmp_path):
    out = _run_inspect(tmp_path, "stain2/stain1", "single")
    assert not (out/"B_combined.png").exists()


def test_saved_field_array_reproduces_the_diagram(tmp_path):
    """The .npy must be the exact values the diagram came from."""
    import numpy as np
    import torch
    from topo_i2i.persistence import persistence_diagram
    out = _run_inspect(tmp_path, "stain1+stain2", "exact")
    arr = np.load(out/"B_field.npy")
    d = persistence_diagram(torch.from_numpy(arr), (0, 1))
    rows = (out/"B_diagram.csv").read_text().strip().split("\n")[1:]
    assert len(rows) == d[0].shape[0] + d[1].shape[0]


# --------------------------------------------------------------------------
# topo-audit: the pre-registered field-selection rule
# --------------------------------------------------------------------------
def _row(fa="stain1/stain2", fb="stain1+stain2", ds=2, dims="0,1", proj="birth",
         auroc=0.6, se=0.02):
    return {"field_A": fa, "field_B": fb, "downsample": ds, "dims": dims,
            "projection": proj, "true": 1.0, "shuffled": 1.1, "ratio": 1.1,
            "auroc": auroc, "auroc_se": se}


def _write_cell(d, name, rows, **meta):
    import json, os
    payload = {"rows": rows, "n_tiles": 512, "n_pairs": 512,
               "combinations": len(rows), "within_slide": True,
               "slide_regex": r"_r\d+c\d+$", "groups": 128, "stains": "/s.json",
               "limit": 512, "offset": 0, "seed": 0, "sample": "random",
               "shuffles": 5, "image_size": 256, "field_combine": "sum",
               "invert": True, "dataA": "/a", "dataB": "/b"}
    payload.update(meta)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as fh:
        json.dump(payload, fh)


def test_bonferroni_bar_rises_with_more_candidates():
    from topo_i2i.audit import z_threshold
    assert z_threshold(1) < z_threshold(5) < z_threshold(50)


def test_a_clear_signal_is_supported_and_noise_is_not():
    from topo_i2i.audit import supported
    assert supported(0.60, 512, 5)           # far above chance on 512 tiles
    assert not supported(0.52, 512, 5)       # inside the noise band


def test_a_perfect_auroc_on_a_tiny_sample_is_not_certified():
    """The SE at the observed value collapses to 0 at 1.0; the null SE does not."""
    from topo_i2i.audit import null_se, supported
    assert null_se(4) > 0
    assert not supported(1.0, 4, 5)
    assert supported(1.0, 512, 5)


def test_the_bar_is_the_band_the_reports_quote():
    """1.96 * null SE at n=1560 is the +/-0.020 the BCI report printed."""
    from topo_i2i.audit import null_se
    assert 1.96 * null_se(1560) == pytest.approx(0.020, abs=0.001)


def test_screening_ties_break_toward_the_cheaper_setting():
    from topo_i2i.audit import top_candidates
    screen = {"rows": [_row(ds=1, auroc=0.6), _row(ds=4, auroc=0.6)]}
    assert top_candidates(screen, 1)[0]["downsample"] == 4


def test_verdict_is_ph_cyc_only_when_nothing_survives(tmp_path):
    from topo_i2i.audit import decide_marker
    d = tmp_path / "BCI"
    _write_cell(str(d), "screen_estimated_strict.json", [_row(auroc=0.547)])
    _write_cell(str(d), "confirm_estimated_00.json", [_row(auroc=0.512, se=0.02)])
    out = decide_marker(str(d), ["estimated", "fixed"])
    assert out["verdict"] == "ph_cyc_only"


def test_verdict_is_supported_when_a_candidate_clears_the_bar(tmp_path):
    from topo_i2i.audit import decide_marker
    d = tmp_path / "Ki67"
    _write_cell(str(d), "screen_estimated_strict.json", [_row(auroc=0.60)])
    _write_cell(str(d), "confirm_estimated_00.json", [_row(auroc=0.60, se=0.01)])
    out = decide_marker(str(d), ["estimated", "fixed"])
    assert out["verdict"] == "ph_trans_supported"
    assert out["recommended"]["setting"][0] == "stain1/stain2"


def test_the_winner_is_chosen_on_the_held_out_slice_not_the_screen(tmp_path):
    """The whole point: the screen leader is the number selection inflated."""
    from topo_i2i.audit import decide_marker
    d = tmp_path / "ER"
    # Candidate 0 screened best; candidate 1 confirms best. 1 must win.
    _write_cell(str(d), "screen_estimated_strict.json",
                [_row(proj="birth", auroc=0.80), _row(proj="lifetime", auroc=0.70)])
    _write_cell(str(d), "confirm_estimated_00.json",
                [_row(proj="birth", auroc=0.58, se=0.01)])
    _write_cell(str(d), "confirm_estimated_01.json",
                [_row(proj="lifetime", auroc=0.64, se=0.01)])
    out = decide_marker(str(d), ["estimated"])
    assert out["verdict"] == "ph_trans_supported"
    assert out["recommended"]["setting"][4] == "lifetime"


def test_specimen_recognition_delta_is_reported(tmp_path):
    from topo_i2i.audit import decide_marker
    d = tmp_path / "BCI"
    _write_cell(str(d), "screen_estimated_strict.json", [_row(auroc=0.512)])
    _write_cell(str(d), "screen_estimated_unstratified.json", [_row(auroc=0.702)],
                within_slide=False, slide_regex=None, groups=None)
    _write_cell(str(d), "confirm_estimated_00.json", [_row(auroc=0.512, se=0.02)])
    out = decide_marker(str(d), ["estimated"])
    assert out["arms"]["estimated"]["specimen_recognition_delta"] == pytest.approx(0.19)


def test_recommendation_env_defers_to_an_explicit_override(tmp_path):
    """Sourcing must fill in blanks, never overwrite what the submitter set."""
    import subprocess
    from topo_i2i.audit import decide_marker, env_lines
    d = tmp_path / "Ki67"
    _write_cell(str(d), "screen_estimated_strict.json", [_row(auroc=0.60)])
    _write_cell(str(d), "confirm_estimated_00.json", [_row(auroc=0.60, se=0.01)])
    env = tmp_path / "rec.env"
    env.write_text("\n".join(env_lines(decide_marker(str(d), ["estimated"]))) + "\n")
    script = 'FIELD_A=mine; . "%s"; echo "$FIELD_A $FIELD_B $TOPO_PROJECTION"' % env
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.stdout.split() == ["mine", "stain1+stain2", "birth"]


def test_validate_fields_json_has_one_row_per_combination(tmp_path):
    import json, sys
    import numpy as np
    from PIL import Image
    from topo_i2i.validate_fields import build_parser, main
    a, b = tmp_path / "A", tmp_path / "B"
    for d in (a, b):
        d.mkdir()
        for i in range(4):
            arr = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / ("s0_r0c%d.png" % i))
    out = tmp_path / "j.json"
    argv = sys.argv
    sys.argv = ["topo-validate-fields", "--dataA", str(a), "--dataB", str(b),
                "--field-A", "gray", "--field-B", "gray", "--downsample", "1", "2",
                "--dims-set", "0", "--topo-projection", "birth", "lifetime",
                "--image-size", "32", "--limit", "4", "--shuffles", "1",
                "--json", str(out)]
    try:
        main()
    finally:
        sys.argv = argv
    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == 4 == payload["combinations"]
    assert all(0.0 <= r["auroc"] <= 1.0 for r in payload["rows"])
    # The per-row SE is estimated at the observed AUROC, so it may legitimately
    # be 0 on four noise tiles that separate perfectly; the audit tests against
    # the null SE instead, which cannot.
    assert all(r["auroc_se"] >= 0 for r in payload["rows"])


# --------------------------------------------------------------------------
# Regression: gudhi indexes its critical cells in FORTRAN order
# --------------------------------------------------------------------------
def _gudhi_intervals(field, dim):
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=field)
    cc.compute_persistence(homology_coeff_field=2)
    ref = cc.persistence_intervals_in_dimension(dim)
    return ref[np.isfinite(ref[:, 1])]


def _lex(a):
    return a[np.lexsort((a[:, 1], a[:, 0]))]


@pytest.mark.parametrize("shape", [(5, 7), (13, 29), (32, 32), (64, 64)])
def test_diagram_values_match_gudhis_own_intervals(shape):
    """The values we gather must BE the persistence diagram, not merely look
    like one. gudhi's cofaces_of_persistence_pairs returns indices into the
    column-major flattening; gathering them row-major reads the transposed
    pixel, which on a square field is silently wrong rather than an error."""
    from topo_i2i.persistence import persistence_diagram
    field = np.random.default_rng(0).random(shape)
    ours = persistence_diagram(torch.from_numpy(field).double(), (0, 1))
    for dim in (0, 1):
        got, ref = ours[dim].numpy(), _gudhi_intervals(field, dim)
        assert len(got) == len(ref), "dim %d: %d points vs %d" % (dim, len(got), len(ref))
        assert np.allclose(_lex(got), _lex(ref)), "dim %d values differ" % dim


@pytest.mark.parametrize("shape", [(17, 41), (48, 48)])
def test_birth_never_exceeds_death(shape):
    """In a sublevel filtration a feature cannot die before it is born; when it
    appears to, the critical indices were read in the wrong memory order."""
    from topo_i2i.persistence import persistence_diagram
    field = torch.from_numpy(np.random.default_rng(1).random(shape)).double()
    d = persistence_diagram(field, (0, 1))
    for dim in (0, 1):
        b, death = d[dim][:, 0], d[dim][:, 1]
        assert torch.all(death >= b), "dim %d has %d inverted pairs" % (
            dim, int((death < b).sum()))


def test_gradient_lands_on_the_pixel_that_set_the_value():
    """Non-square on purpose: a transposed gather would move the gradient to a
    different pixel, and on a square field that pixel still exists."""
    from topo_i2i.persistence import persistence_diagram
    field = torch.from_numpy(np.random.default_rng(2).random((9, 23))).double()
    field.requires_grad_(True)
    d = persistence_diagram(field, (0,))
    b = d[0][:, 0]
    b.sum().backward()
    touched = field.grad.nonzero()
    # every pixel the gradient reached must carry one of the birth values
    vals = field.detach()[touched[:, 0], touched[:, 1]]
    assert len(touched) > 0
    for v in vals:
        assert torch.isclose(b.detach(), v).any(), "gradient on a non-critical pixel"


# --------------------------------------------------------------------------
# topo-inspect --literature: look at what training actually runs when the
# audit's fixed-vector arm wins
# --------------------------------------------------------------------------
def _inspect_argv(tmp_path, out, *extra, field_a="eosin/hematoxylin",
                  field_b="hematoxylin+dab"):
    import numpy as np
    from topo_i2i.fields import STAIN_VECTORS
    H = np.array(STAIN_VECTORS["hematoxylin"]); H /= np.linalg.norm(H)
    E = np.array(STAIN_VECTORS["eosin"]);       E /= np.linalg.norm(E)
    D = np.array(STAIN_VECTORS["dab"]);         D /= np.linalg.norm(D)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _stain_tile(str(a), H, E, 0)
    _stain_tile(str(b), H, D, 1)
    return ["topo-inspect", "--imageA", str(a), "--imageB", str(b),
            "--field-A", field_a, "--field-B", field_b,
            "--image-size", "64", "--downsample", "1",
            "--outdir", str(out)] + list(extra)


def _run_argv(argv):
    import sys
    from topo_i2i.inspect import main
    saved = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = saved


def test_literature_uses_the_builtin_table_and_records_it(tmp_path):
    import json
    out = tmp_path / "lit"
    _run_argv(_inspect_argv(tmp_path, out, "--literature"))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["vector_source"] == "the built-in literature table"
    assert summary["vectors"]["A"] is None, "no estimated vectors should be recorded"
    assert (out / "A_diagram.csv").exists()


def test_literature_rejects_positional_stain_names(tmp_path):
    """stain1/stain2 only mean something with estimated vectors."""
    out = tmp_path / "bad"
    with pytest.raises(SystemExit) as e:
        _run_argv(_inspect_argv(tmp_path, out, "--literature",
                                field_a="stain1/stain2", field_b="stain1+stain2"))
    msg = str(e.value)
    assert "stain1" in msg and "--stains" in msg


def test_literature_and_stains_are_mutually_exclusive(tmp_path):
    out = tmp_path / "both"
    with pytest.raises(SystemExit):
        _run_argv(_inspect_argv(tmp_path, out, "--literature",
                                "--stains", str(tmp_path / "nope.json")))


def test_the_overview_projection_panel_equals_the_distance(tmp_path):
    """The area between the two plotted curves must BE the diagram distance.

    The panel pads then sorts, exactly as diagram_distance does. Sorting first
    and padding afterwards -- the obvious reading -- draws a different picture
    on a negated field, where every value lies below the padding zeros.
    """
    import numpy as np
    import torch
    from topo_i2i.losses import _project, _resolve_projection, diagram_distance
    from topo_i2i.persistence import persistence_diagram

    rng = np.random.default_rng(0)
    # different sizes on purpose, so the padding actually does something
    a = persistence_diagram(torch.from_numpy(-rng.random((40, 40))).double(), (0, 1))
    b = persistence_diagram(torch.from_numpy(-rng.random((24, 24))).double(), (0, 1))
    for dim in (0, 1):
        assert a[dim].shape[0] != b[dim].shape[0], "sizes must differ to test padding"

    total = 0.0
    for dim in (0, 1):
        how = _resolve_projection(None, dim)
        pa, pb = _project(a, dim, how).numpy(), _project(b, dim, how).numpy()
        n = max(len(pa), len(pb))
        pa = np.sort(np.concatenate([pa, np.zeros(n - len(pa))]))
        pb = np.sort(np.concatenate([pb, np.zeros(n - len(pb))]))
        total += np.abs(pa - pb).sum()
    assert total == pytest.approx(float(diagram_distance(a, b, (0, 1), None)), rel=1e-9)


def test_threaded_diagrams_match_serial_and_keep_order(tmp_path):
    """Threading the persistence loop must not change a single value, and must
    preserve order -- index i has to be the same tile in every cached spec or
    the true/shuffled pairing silently shifts."""
    import numpy as np
    from PIL import Image
    from topo_i2i.validate_fields import diagrams_for
    paths = []
    for i in range(6):
        p = tmp_path / ("t%d.png" % i)
        arr = (np.random.default_rng(i).random((64, 64, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    serial = diagrams_for(paths, "gray", "sum", 64, 1, (0, 1), True, workers=1)
    threaded = diagrams_for(paths, "gray", "sum", 64, 1, (0, 1), True, workers=4)
    assert len(serial) == len(threaded) == len(paths)
    for a, b in zip(serial, threaded):
        for dim in (0, 1):
            assert torch.equal(a[dim], b[dim])


def test_worker_count_comes_from_the_slurm_allocation(monkeypatch):
    from topo_i2i.validate_fields import _workers
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("TOPO_WORKERS", raising=False)
    assert _workers() == 8
    monkeypatch.setenv("TOPO_WORKERS", "3")
    assert _workers() == 3, "TOPO_WORKERS must win, for a local run"
    monkeypatch.setenv("TOPO_WORKERS", "nonsense")
    assert _workers() >= 1, "a bad value must not crash the job"
