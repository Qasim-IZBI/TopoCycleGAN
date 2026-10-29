"""Tests for the loss-figure tool's own logic. Run: pytest -q"""

import numpy as np
import pytest
import torch

from topo_i2i.figure import find_checkpoint, sorted_pair
from topo_i2i.losses import diagram_distance


def test_find_checkpoint_takes_the_newest_step(tmp_path):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for s in (5000, 40000, 200000):
        (ck / ("step_%d.pt" % s)).write_bytes(b"")
    (ck / "step_latest_copy.pt").write_bytes(b"")
    assert find_checkpoint(str(tmp_path)).endswith("step_200000.pt")     # the run dir
    assert find_checkpoint(str(ck)).endswith("step_200000.pt")           # checkpoints/
    assert find_checkpoint(str(tmp_path), 40000).endswith("step_40000.pt")
    assert find_checkpoint(str(ck / "step_5000.pt")).endswith("step_5000.pt")
    with pytest.raises(FileNotFoundError):
        find_checkpoint(str(tmp_path), 123)


@pytest.mark.parametrize("how", ["lifetime", "birth"])
def test_plotted_pair_is_the_distance(how):
    rng = np.random.default_rng(0)
    a = {0: torch.as_tensor(np.sort(rng.normal(-1, .3, (7, 2)), axis=1))}
    b = {0: torch.as_tensor(np.sort(rng.normal(-1, .3, (11, 2)), axis=1))}
    pa, pb = sorted_pair(a, b, 0, how)
    assert len(pa) == len(pb) == 11
    assert np.abs(pa - pb).sum() == pytest.approx(float(diagram_distance(a, b, (0,), how)))
