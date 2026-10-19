"""Open-weights nuclei detectors besides StarDist, for the nuclei evaluation.

StarDist ships no model for brightfield IHC, so the IHC side of the nuclei
evaluation runs its FLUORESCENCE model on colour-deconvolved haematoxylin
(+ DAB) -- out of domain, and it finds ~1.6-2x the nuclei the H&E side finds on
real adjacent sections. These two read both stains as RGB, with one model:

    instanseg  InstanSeg "brightfield_nuclei" (instanseg-torch; Apache-2.0),
               trained on H&E AND IHC brightfield. Fast; fine on CPU.
    cpsam      Cellpose-SAM, "cpsam_v2" (cellpose >= 4), a generalist built to
               be robust to stain and channel order. SAM backbone: needs a GPU
               (~2 min per tile on CPU).

Each is a method with StarDist's variant signature, fn(rgb, scale) -> labels at
the tile's own size, so nuclei_pilot.py and nuclei_eval.py use them the way
they use StarDist's. `scale` is applied here, by resizing the tile before the
model and the labels back after (nearest), not through either library's own
pixel-size or diameter handling -- so a scale means the same for every
detector, and the pilot chooses it the same way.

Both live in a PyTorch environment of their own (StarDist's is TensorFlow):

    conda create -n nuclei-torch python=3.11 -y
    P=~/.conda/envs/nuclei-torch/bin/python
    $P -m pip install torch instanseg-torch "cellpose>=4" scikit-image matplotlib scipy

Weights are fetched once, on the login node, into a fixed directory the jobs
only read (compute nodes have no internet, and parallel first-use downloads
would race):

    $P tools/nuclei_pilot.py --detector instanseg --fetch-models \\
        --models-dir /work2/bz66izin-TopoCG/nuclei_models
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

DETECTORS = ("instanseg", "cpsam")
INSTANSEG_MODEL = "brightfield_nuclei"
CELLPOSE_MODEL = "cpsam_v2"


def use_models_dir(models_dir: Optional[str]) -> None:
    """Point both libraries at the fixed weights copy. Must run before cellpose
    is imported: it reads its model directory at import time."""
    if not models_dir:
        return
    os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = os.path.join(models_dir, "instanseg")
    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.path.join(models_dir, "cellpose")


def weights_present(models_dir: str, detector: str) -> bool:
    if detector == "instanseg":
        d = os.path.join(models_dir, "instanseg", INSTANSEG_MODEL)
        return any("instanseg.pt" in files for _, _, files in os.walk(d))
    if detector == "cpsam":
        return os.path.isfile(os.path.join(models_dir, "cellpose", CELLPOSE_MODEL))
    raise ValueError(detector)


def fetch(models_dir: str, detector: str) -> None:
    """Download one detector's weights into models_dir and load them once."""
    use_models_dir(models_dir)
    TorchSegmenter([detector], models_dir, device="cpu")
    print("[models] %s -> %s" % (detector, models_dir))


def rescaled(fn, rgb: np.ndarray, scale: float) -> np.ndarray:
    """fn on the tile resized by `scale`; its labels back at the tile's size."""
    h, w = rgb.shape[:2]
    if scale != 1:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))
    labels = np.asarray(fn(rgb)).astype(np.int32)
    if labels.shape != (h, w):
        labels = np.asarray(Image.fromarray(labels).resize((w, h), Image.NEAREST))
    return labels.astype(np.int32)


class TorchSegmenter:
    """InstanSeg and / or Cellpose-SAM on a tile; a blank tile gets an empty mask
    without being run. Same interface as nuclei_pilot.Segmenter."""

    def __init__(self, detectors, models_dir: Optional[str] = None,
                 device: Optional[str] = None):
        use_models_dir(models_dir)
        import torch
        self.blank = 0
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._instanseg = self._cellpose = None
        if "instanseg" in detectors:
            from instanseg import InstanSeg
            self._instanseg = InstanSeg(INSTANSEG_MODEL, device=self.device, verbosity=0)
        if "cpsam" in detectors:
            from cellpose import models
            self._cellpose = models.CellposeModel(pretrained_model=CELLPOSE_MODEL,
                                                  device=torch.device(self.device))
        print("[detector] %s on %s" % (", ".join(detectors), self.device))

    def _empty_if_blank(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        from nuclei_pilot import is_blank
        if is_blank(rgb):
            self.blank += 1
            return np.zeros(rgb.shape[:2], dtype=np.int32)
        return None

    def instanseg(self, rgb: np.ndarray, scale: float) -> np.ndarray:
        empty = self._empty_if_blank(rgb)
        if empty is not None:
            return empty

        def run(x):
            import torch
            chw = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
            out = self._instanseg.eval_small_image(chw, pixel_size=None,
                                                   return_image_tensor=False, target="nuclei")
            return out.detach().cpu().numpy().reshape(-1, *x.shape[:2])[0]
        return rescaled(run, rgb, scale)

    def cpsam(self, rgb: np.ndarray, scale: float) -> np.ndarray:
        empty = self._empty_if_blank(rgb)
        if empty is not None:
            return empty

        def run(x):
            masks, _, _ = self._cellpose.eval(x, channel_axis=2)
            return masks
        return rescaled(run, rgb, scale)
