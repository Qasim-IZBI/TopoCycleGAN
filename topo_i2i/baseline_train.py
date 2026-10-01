"""Run the zoo's trainer with resume turned on, for the baseline models.

`i2i-train` builds the model, the loader and the trainer and then calls
`trainer.train(...)` -- but never `trainer.resume_if_exists()`. `topo-train`
does call it (train.py), which is why a TopoCycleGAN cell that hits the wall
continues where it stopped while a CycleGAN or DCLGAN baseline silently
restarts from step 0, on top of the checkpoints it already wrote.

Nothing is missing from the checkpoints themselves. BaseTrainer.save_checkpoint
stores the model, opt_G, opt_D, any aux optimisers, the global step and the
accumulated training time, and resume_if_exists restores all of it after
checking the saved config against the current one. The baselines were losing
that for want of a single call.

This supplies the call and changes nothing else. It takes the same arguments as
`i2i-train`, because it IS `i2i-train`: the CLI, the model construction, the
dataset and the trainer wiring are the zoo's, reached by delegating to its
`main()`. Forking any of that to insert one line would leave a copy to drift
against the original.

    topo-baseline --model dclgan --dataA trainA/ --dataB trainB/ \\
        --output runs/ER_dclgan --steps 400000 --dclgan_ngf 136

Resuming is the whole point, so it is unconditional: point it at a directory
holding checkpoints and it continues from the furthest-ahead one, preferring
step_latest.pt when that is ahead of the last numbered save. To start over,
train into a new --output, or delete the checkpoints first. A checkpoint whose
config disagrees with the current arguments makes BaseTrainer refuse to resume
rather than quietly load mismatched weights -- that error is the zoo's and it
is the right behaviour, so it is left alone.
"""

from __future__ import annotations

import sys

from i2i_stain_zoo.trainer.base_trainer import BaseTrainer


def _train_resuming(self, total_steps: int):
    """BaseTrainer.train, preceded by the resume the zoo's CLI leaves out."""
    self.resume_if_exists()
    return _train_resuming.__wrapped__(self, total_steps)


_train_resuming.__wrapped__ = BaseTrainer.train


def main() -> None:
    # Patch the method rather than reimplement the CLI around it. i2i-train
    # constructs BaseTrainer itself, deep inside its main(), so there is no
    # argument or subclass hook to pass a resuming trainer through -- and the
    # alternative, copying its ~40 lines of model, dataset and trainer setup
    # into this repo, is a fork that would drift the moment the zoo changes a
    # default. The patch is process-local and applies to the call below only.
    BaseTrainer.train = _train_resuming
    try:
        import i2i_stain_zoo.train as zoo_train
        zoo_train.main()
    finally:
        BaseTrainer.train = _train_resuming.__wrapped__


if __name__ == "__main__":
    sys.exit(main())
