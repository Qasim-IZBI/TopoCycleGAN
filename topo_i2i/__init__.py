"""TopoGAN-style topological loss for unpaired image-to-image virtual staining.

Implements the loss of

    Wang, Liu, Samaras, Chen. "TopoGAN: A Topology-Aware Generative
    Adversarial Network." ECCV 2020.

on top of the I2I-Stain-Zoo package, without modifying it: the models here
subclass the zoo's models and add one term to `compute_generator_loss`.

The loss itself (fields, persistence, losses) has no dependency on the zoo, so
it can be used standalone -- e.g. to score how topologically different two
images are. Only `topo_i2i.models` and `topo_i2i.train` need the zoo, and they
are imported lazily so the rest works without it.
"""

from topo_i2i.fields import rgb_to_scalar_field, StainField
from topo_i2i.losses import diagram_distance, topological_loss
from topo_i2i.persistence import persistence_diagram

__all__ = [
    "rgb_to_scalar_field", "StainField",
    "persistence_diagram", "diagram_distance", "topological_loss",
    # lazy, require i2i-stain-zoo:
    "TopoLossMixin", "TopoCycleGAN", "TopoCycleGANConfig", "TopoConfig",
]

_LAZY = {
    "TopoLossMixin": "topo_i2i.models",
    "TopoCycleGAN": "topo_i2i.models",
    "TopoCycleGANConfig": "topo_i2i.models",
    "TopoConfig": "topo_i2i.models",
}


def __getattr__(name):
    """Import the zoo-dependent symbols only when they are actually used."""
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
